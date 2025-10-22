# train_cartpole_terrain_no_episode_truncation.py
# 變更重點：移除「每回合 500 步截斷」，與 CODE1 一致 → 回合只在 terminated 時結束
# 仍保留：tau=0.02, total_episodes=50, timesteps_per_episode=100, max_step_count=500000
#        自動偵測 CUDA、記錄訓練時間、直接輸出 .npz 到 target_dir、地形/phi
# 修正：整合 'Time' 參數為重複運行次數（每個地形配置跑 times 次，檔名加 _runN）

import time
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

import warnings, os, math
import numpy as np
from typing import Optional

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ===== 基本訓練/收集參數（與你的既定設定一致）=====
tau = 0.02                   # 時間步長
total_episodes = 50          # 訓練回合數
timesteps_per_episode = 100  # 每回合訓練步數
max_step_count = 500000      # 測試/資料蒐集步數上限（全域）

rng = np.random.default_rng(0)

# ===== 互動式輸入 =====
Length = []
Mass = []
Friction_coef = [0]

print("Length: ")
while True:
    try:
        x = float(input())
        if x != 0:
            Length.append(x)
        else:
            break
    except ValueError:
        print("無效輸入，請輸入數字或 0")

print("Mass: ")
while True:
    try:
        x = float(input())
        if x != 0:
            Mass.append(x)
        else:
            break
    except ValueError:
        print("無效輸入，請輸入數字或 0")

print("Friction_coef: ")
while True:
    try:
        x = float(input())
        if x != 0:
            Friction_coef.append(x)
        else:
            break
    except ValueError:
        print("無效輸入，請輸入數字或 0")

# ===== 地形輸入 =====
# 需提供同目錄下的 Terrain_Setting.py（你給的版本）
from Terrain_Setting import Terrain
terrain = Terrain()
terrain.ask_user()

PHI_MAX_DEG = None  # 若想限制坡角，可改數字（度）

class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, mass=0.1, friction_coef=0.0,
                 render_mode=None, diff_dx=5e-4, terrain_name=None, terrain_fn=None,
                 # ← 注意：不再使用 max_episode_steps；只靠 terminated 結束
                 force_mag: float = 20.0):
        super().__init__(render_mode=render_mode)

        self.force_mag = float(force_mag)

        self.length = length / 2.0
        self.masspole = mass
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart
        self.friction_coef = friction_coef
        self.tau = tau

        self.diff_dx = float(diff_dx)
        self.terrain_name = terrain_name
        self.terrain_fn = terrain_fn
        self.phi_max = None if PHI_MAX_DEG is None else np.radians(PHI_MAX_DEG)

        # 不使用每回合步數截斷 → 不需要 _step_count / max_episode_steps
        # self._step_count = 0

        high = np.array(
            [self.x_threshold*2, np.inf, self.theta_threshold_radians*2, np.inf],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _phi_from_x(self, x: float):
        dx = self.diff_dx
        h  = self.terrain_fn
        slope = (h(x + dx) - h(x - dx)) / (2.0 * dx)
        phi = np.arctan(slope)
        if self.phi_max is not None:
            phi = np.clip(phi, -self.phi_max, self.phi_max)
        return slope, phi

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        # 不再重置 _step_count
        return obs, info

    def step(self, action):
        action = int(np.asarray(action).squeeze())
        assert self.state is not None, "Call reset() first."
        x, x_dot, theta, theta_dot = self.state

        # 推力 + 地面速度型阻尼（與你地形版一致）
        force = self.force_mag if action == 1 else -self.force_mag
        force += -self.friction_coef * x_dot

        _, phi = self._phi_from_x(x)

        M = self.masscart
        m = self.masspole
        l = self.length
        g = self.gravity

        th_eff = theta + phi
        cost = np.cos(th_eff)
        sint = np.sin(th_eff)

        denom = (M + m) * l - m * l * cost * cost

        thetaacc = (
            (M + m) * g * np.sin(theta)
            - cost * (force + m * l * theta_dot**2 * sint - (M + m) * g * np.sin(phi))
        ) / denom

        xacc = (
            (force + m * l * (theta_dot**2 * sint - thetaacc * cost)) / (M + m)
            - g * np.sin(phi)
        )

        x         = x      + self.tau * x_dot     + 0.5 * xacc     * self.tau**2
        x_dot     = x_dot  + self.tau * xacc
        theta     = theta  + self.tau * theta_dot  + 0.5 * thetaacc * self.tau**2
        theta_dot = theta_dot + self.tau * thetaacc

        terminated = (
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )
        if terminated:
            x = float(np.clip(x, -self.x_threshold, self.x_threshold))

        self.state = (x, x_dot, theta, theta_dot)

        # 關鍵：不做回合步數截斷
        truncated = False

        reward = 1.0
        return np.array(self.state, dtype=np.float32), reward, bool(terminated), bool(truncated), {}

def create_env(length, mass, friction_coef, terrain_name, terrain_fn):
    # 關鍵：不再傳 max_episode_steps=500
    return CartPoleWithVaryingParameters(length=length, mass=mass,
                                         friction_coef=friction_coef,
                                         terrain_name=terrain_name, terrain_fn=terrain_fn,
                                         force_mag=20.0)

episode_rewards = []

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

def Model_Training(length, mass, friction_coef, terrain_name, terrain_fn, header_tag):
    # 自動偵測 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    header = f"L={length}, M={mass}, F={friction_coef}, Terrain={header_tag}"
    if tqdm: print(f"\n== Training [{header}] on {device}")
    else:    print(f"\n== Training [{header}] on {device}")

    env = DummyVecEnv([lambda: create_env(length, mass, friction_coef, terrain_name, terrain_fn)])
    model = PPO("MlpPolicy", env, verbose=0, device=device)

    # 訓練計時
    t0 = time.time()
    pbar = tqdm(total=total_episodes * timesteps_per_episode,
                desc=f"Train steps ({header})", leave=False) if tqdm else None

    for episode in range(total_episodes):
        cb = TqdmStepCallback(pbar)
        model.learn(total_timesteps=timesteps_per_episode,
                    reset_num_timesteps=False, callback=cb)
        if (episode + 1) % 10 == 0:
            print("episode：", episode + 1)

    if pbar: pbar.close()
    train_secs = time.time() - t0
    return model, device, train_secs

# ===== 輸出資料夾 =====
output_dir = "cartpole_data"
os.makedirs(output_dir, exist_ok=True)

target_dir = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Cartpole_Terrain\cartpole_data"
os.makedirs(target_dir, exist_ok=True)

# ==== Grid 訓練（修正：加入 'Time' 重複運行）====
combo_idx = 0
# 修正：grid_total 包含每個地形的 'Time'
grid_total = 0
for i in range(len(terrain.terrain)):
    times = max(1, terrain.terrain[i]['Time'])  # 至少跑 1 次，避免 0
    grid_total += times * len(Length) * len(Mass) * len(Friction_coef)

outer_bar = tqdm(total=grid_total, desc="Parameter grid", leave=True) if tqdm else None

for i in range(len(terrain.terrain)):
    TERRAIN_TAG   = terrain.get_tag(i)
    TERRAIN_FN    = terrain.get_fn(i)
    TERRAIN_NAME  = terrain.terrain[i]['Terrain Name']
    TERRAIN_PARAM = terrain.terrain[i]['Parameter']
    times = max(1, terrain.terrain[i]['Time'])  # 至少 1 次

    # 新增：重複運行次數迴圈
    for run in range(times):
        run_suffix = f"_run{run+1}" if times > 1 else ""  # 檔名加 _runN，若 >1 次

        for length in Length:
            for mass in Mass:
                for friction_coef in Friction_coef:
                    combo_idx += 1

                    # ===== 訓練（含計時）=====
                    model, device_used, train_secs = Model_Training(
                        length, mass, friction_coef,
                        TERRAIN_NAME, TERRAIN_FN, TERRAIN_TAG
                    )

                    # ===== 測試/資料蒐集 =====
                    test_env = create_env(length, mass, friction_coef, TERRAIN_NAME, TERRAIN_FN)

                    cart_positions = []
                    cart_velocities = []
                    pole_angles = []
                    pole_angular_velocities = []
                    actions_taken = []
                    rewards_received = []
                    timestamps = []
                    pole_tip_offsets = []
                    phis = []

                    obs, info = test_env.reset(seed=run)  # 改用 run 作為 seed，增加多樣性
                    done = False
                    step_count = 0
                    total_reward = 0.0

                    rec_bar = tqdm(total=max_step_count,
                                   desc=f"Record (L={length},M={mass},F={friction_coef}, Run={run+1})",
                                   leave=False) if tqdm else None

                    while not done and step_count < max_step_count:
                        x, x_dot, theta, theta_dot = obs

                        cart_positions.append(float(x))
                        cart_velocities.append(float(x_dot))
                        pole_angles.append(float(theta))
                        pole_angular_velocities.append(float(theta_dot))
                        timestamps.append(step_count * tau)
                        pole_tip_offsets.append(test_env.unwrapped.length * np.sin(theta))

                        _, phi_now = test_env.unwrapped._phi_from_x(float(x))
                        phis.append(float(phi_now))

                        action, _ = model.predict(obs, deterministic=True)
                        actions_taken.append(int(action))

                        obs, reward, terminated, truncated, info = test_env.step(action)
                        rewards_received.append(float(reward))
                        total_reward += float(reward)
                        done = bool(terminated or truncated)

                        step_count += 1
                        if rec_bar: rec_bar.update(1)

                    test_env.close()
                    if rec_bar: rec_bar.close()

                    # ===== 組裝輸出（加 run_suffix）=====
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
                        'episode_rewards': np.array(episode_rewards),
                        'phis': np.array(phis),

                        # 地形資訊
                        'terrain_name': np.array([TERRAIN_NAME]),
                        'terrain_parameter': np.array([TERRAIN_PARAM], dtype=object),

                        # 訓練 meta（加 run 資訊）
                        'training_seconds': np.array(train_secs, dtype=np.float64),
                        'training_device': np.array([device_used]),
                        'training_total_steps': np.array(total_episodes * timesteps_per_episode, dtype=np.int64),
                        'run_number': np.array([run + 1]),  # 新增：記錄這是第幾次運行
                    }

                    base_name = f"Terrain_{TERRAIN_TAG}_PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}{run_suffix}"
                    npz_name  = base_name + ".npz"

                    # 本地備份
                    local_full_path = os.path.join(output_dir, npz_name)
                    np.savez(local_full_path, **recorded_data)

                    # 指定資料夾輸出
                    target_full_path = os.path.join(target_dir, npz_name)
                    np.savez(target_full_path, **recorded_data)

                    msg = f"Saved data: {target_full_path}  (local backup: {local_full_path})"
                    if tqdm:
                        tqdm.write(msg)
                        outer_bar.update(1)
                    else:
                        print(msg)

if outer_bar: outer_bar.close()