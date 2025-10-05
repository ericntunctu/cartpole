# train_cartpole_terrain.py  -- terrain = 0.1 * sin(8x), force_mag = 20N, 有 500 步上限且不出框
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

import warnings, os, math
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings('ignore')

# tqdm (optional pretty progress bars)
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # fallback to prints

# ===== 基本訓練參數 =====
tau = 0.02                 
TOTAL_SECONDS_PER_EP = 10.0   # 每個 episode 的牆鐘秒數
RECORD_SECONDS = 120.0        # 記錄用的總秒數（畫 PSD 會用到）

total_episodes = 80           # 多跑幾集（原本 50）
timesteps_per_episode = int(TOTAL_SECONDS_PER_EP / tau)  # 每集步數 = 秒數 / tau
max_step_count = int(RECORD_SECONDS / tau)               # 錄製最多步數 = 秒數 / tau

rng = np.random.default_rng(0)

# ===== 參數輸入 =====
Length = []
Mass = []
Friction_coef = [0]

print("Length: ")
while True:
    x = float(input())
    if x != 0: Length.append(x)
    else: break

print("Mass: ")
while True:
    x = float(input())
    if x != 0: Mass.append(x)
    else: break

print("Friction_coef: ")
while True:
    x = float(input())
    if x != 0: Friction_coef.append(x)
    else: break

# ===== 地形：0.1 * sin(8x) =====
TERRAIN_TAG = "sin_A0.1_k8.0"
TERRAIN_FN  = lambda x: 0.1 * np.sin(8.0 * x)

PHI_MAX_DEG = None  # 可選：限制地形傾角（例如 10）

# ===== 自訂環境（θ → θ+φ），加上 500 步上限與 x 夾界 =====
class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, mass=0.1, friction_coef=0.0,
                 render_mode=None, diff_dx=5e-4, terrain_name=TERRAIN_TAG, terrain_fn=TERRAIN_FN,
                 max_episode_steps: int = 500, force_mag: float = 20.0):
        super().__init__(render_mode=render_mode)

        # 推力調成 20N
        self.force_mag = float(force_mag)

        # 物理參數（半長制，對齊父類）
        self.length = length / 2.0
        self.masspole = mass
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart
        self.friction_coef = friction_coef
        self.tau = tau

        # 地形
        self.diff_dx = float(diff_dx)
        self.terrain_name = terrain_name
        self.terrain_fn = terrain_fn
        self.phi_max = None if PHI_MAX_DEG is None else np.radians(PHI_MAX_DEG)

        # 限制回合步數（模擬 TimeLimit wrapper）
        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0

        # observation space（與父類一致）
        high = np.array([self.x_threshold*2, np.inf, self.theta_threshold_radians*2, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # φ：由地形斜率取 atan（中心差分）
    def _phi_from_x(self, x: float):
        dx = self.diff_dx
        h  = self.terrain_fn
        slope = (h(x + dx) - h(x - dx)) / (2.0 * dx)  # dh/dx
        phi = np.arctan(slope)
        if self.phi_max is not None:
            phi = np.clip(phi, -self.phi_max, self.phi_max)
        return slope, phi

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._step_count = 0
        return obs, info

    def step(self, action):
        action = int(np.asarray(action).squeeze())
        assert self.state is not None, "Call reset() first."
        x, x_dot, theta, theta_dot = self.state

        # 指令力 ±F 並加上黏滯摩擦 -b*x_dot（作用在車）
        force = self.force_mag if action == 1 else -self.force_mag
        force += -self.friction_coef * x_dot

        # 地形角 φ
        _, phi = self._phi_from_x(x)

        # 便利記號
        M = self.masscart
        m = self.masspole
        l = self.length
        g = self.gravity

        # ======= 只改「兩條方程」：把 θ 換成 θ+φ =======
        th_eff = theta + phi           # θ → θ + φ
        cost = np.cos(th_eff)
        sint = np.sin(th_eff)

        denom = (M + m) * l - m * l * cost * cost

        # 1) θ¨
        thetaacc = (
            (M + m) * g * np.sin(theta)   # = sin((θ+φ)-φ)
            - cost * (force + m * l * theta_dot**2 * sint - (M + m) * g * np.sin(phi))
        ) / denom

        # 2) x¨
        xacc = (
            (force + m * l * (theta_dot**2 * sint - thetaacc * cost)) / (M + m)
            - g * np.sin(phi)
        )

        # 積分（x, θ 含二階項）
        x         = x      + self.tau * x_dot     + 0.5 * xacc     * self.tau**2
        x_dot     = x_dot  + self.tau * xacc
        theta     = theta  + self.tau * theta_dot  + 0.5 * thetaacc * self.tau**2
        theta_dot = theta_dot + self.tau * thetaacc

        # === 出框處理：仍判 terminated=True，但把 x 夾回界內，避免動畫或數據越界 ===
        terminated = (
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )
        if terminated:
            x = float(np.clip(x, -self.x_threshold, self.x_threshold))

        self.state = (x, x_dot, theta, theta_dot)

        # === TimeLimit：最多 500 步 ===
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        reward = 1.0
        return np.array(self.state, dtype=np.float32), reward, bool(terminated), bool(truncated), {}

def create_env(length, mass, friction_coef):
    return CartPoleWithVaryingParameters(length=length, mass=mass,
                                         friction_coef=friction_coef,
                                         terrain_name=TERRAIN_TAG, terrain_fn=TERRAIN_FN,
                                         max_episode_steps=500, force_mag=20.0)

episode_rewards = []

# ====== 進度條用 Callback：每 learn() 步就更新 ======
class TqdmStepCallback(BaseCallback):
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar
        self.prev = 0

    def _on_step(self) -> bool:
        cur = self.num_timesteps
        delta = cur - self.prev
        if delta > 0 and self.pbar is not None:
            self.pbar.update(delta)
            self.prev = cur
        return True

def Model_Training(length, mass, friction_coef):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    header = f"L={length}, M={mass}, F={friction_coef}, Terrain={TERRAIN_TAG}"
    if tqdm: print(f"\n== Training [{header}] on {device}")
    else:    print(f"\n== Training [{header}] on {device}")

    env = DummyVecEnv([lambda: create_env(length, mass, friction_coef)])
    model = PPO("MlpPolicy", env, verbose=0, device=device)

    total_steps = total_episodes * timesteps_per_episode
    pbar = tqdm(total=total_steps, desc=f"Train steps ({header})", leave=False) if tqdm else None

    for ep in range(1, total_episodes + 1):
        cb = TqdmStepCallback(pbar)
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False, callback=cb)
        if tqdm:
            # quick eval：這裡也會因為有 TimeLimit 而在 ≤500 步內結束
            test_env = create_env(length, mass, friction_coef)
            obs, _ = test_env.reset(seed=0)
            done, ep_reward = False, 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, _ = test_env.step(action)
                ep_reward += float(r)
                done = bool(terminated or truncated)
            test_env.close()
            pbar.set_postfix(ep=ep, evalR=f"{ep_reward:.0f}")
        elif ep % 10 == 0:
            print(f"  episode {ep}/{total_episodes}")

    if pbar: pbar.close()
    return model

# ===== 輸出資料夾 =====
output_dir = "cartpole_data"
model_dir  = "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 外層參數組合進度
grid_total = max(1, len(Length) * len(Mass) * len(Friction_coef))
outer_bar = tqdm(total=grid_total, desc="Parameter grid", leave=True) if tqdm else None

combo_idx = 0
for length in Length:
    for mass in Mass:
        for friction_coef in Friction_coef:
            combo_idx += 1
            model = Model_Training(length, mass, friction_coef)

            # 存模型（動畫腳本會載入）
            model_path = os.path.join(
                model_dir,
                f"PPO_Terrain_{TERRAIN_TAG}_L{length}_M{mass}_F{friction_coef}.zip"
            )
            model.save(model_path)
            if tqdm: tqdm.write(f"Saved model: {model_path}")
            else:    print(f"Saved model: {model_path}")

            test_env = create_env(length, mass, friction_coef)

            # ===== 資料紀錄（因為環境內有 TimeLimit，這裡每集也會 ≤500 步結束）=====
            cart_positions = []
            cart_velocities = []
            pole_angles = []
            pole_angular_velocities = []
            actions_taken = []
            rewards_received = []
            timestamps = []
            pole_tip_offsets = []
            phis = []  # 記 φ

            obs, info = test_env.reset(seed=0)
            done = False
            step_count = 0
            total_reward = 0.0

            rec_bar = tqdm(total=min(max_step_count, 500), desc=f"Record ({length},{mass},{friction_coef})", leave=False) if tqdm else None

            while not done and step_count < max_step_count:
                x, x_dot, theta, theta_dot = obs
                cart_positions.append(x)
                cart_velocities.append(x_dot)
                pole_angles.append(theta)
                pole_angular_velocities.append(theta_dot)
                timestamps.append(step_count * tau)
                pole_tip_offsets.append(test_env.unwrapped.length * np.sin(theta))  # 半長 * sinθ

                # 記 φ
                _, phi_now = test_env.unwrapped._phi_from_x(float(x))
                phis.append(phi_now)

                action, _ = model.predict(obs, deterministic=True)
                actions_taken.append(action)

                obs, reward, terminated, truncated, info = test_env.step(action)
                rewards_received.append(reward)
                total_reward += float(reward)
                done = bool(terminated or truncated)

                step_count += 1
                if rec_bar: rec_bar.update(1)
                if step_count >= 500:  # 再加一道保險：最多 500 步
                    break

            test_env.close()
            if rec_bar: rec_bar.close()

            # ===== 存檔（檔名含地形）=====
            recorded_data = {
                'cart_positions': np.array(cart_positions),
                'cart_velocities': np.array(cart_velocities),
                'pole_angles': np.array(pole_angles),
                'pole_angular_velocities': np.array(pole_angular_velocities),
                'actions_taken': np.array(actions_taken),
                'rewards_received': np.array(rewards_received),
                'timestamps': np.array(timestamps),
                'total_steps': step_count,
                'total_reward': total_reward,
                'pole_tip_offsets': np.array(pole_tip_offsets),
                'episode_rewards': np.array([]),
                'phis': np.array(phis),
                'terrain_name': np.array([TERRAIN_TAG])
            }

            data_filename = f"Terrain_{TERRAIN_TAG}_PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz"
            full_path = os.path.join(output_dir, data_filename)
            np.savez(full_path, **recorded_data)
            if tqdm:
                tqdm.write(f"Saved data: {full_path}")
                outer_bar.update(1)
            else:
                print(f"Saved data: {full_path} ({combo_idx}/{grid_total})")

if outer_bar: outer_bar.close()