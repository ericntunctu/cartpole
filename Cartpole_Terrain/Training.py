# train_cartpole_terrain.py  -- terrain 可由使用者輸入（Sine / Slope），force_mag = 20N，500 步上限且不出框
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
import zipfile

warnings.filterwarnings('ignore')

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# ===== 基本訓練參數 =====
tau = 0.02
TOTAL_SECONDS_PER_EP = 10.0
RECORD_SECONDS = 120.0

total_episodes = 80
timesteps_per_episode = int(TOTAL_SECONDS_PER_EP / tau)
max_step_count = int(RECORD_SECONDS / tau)

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

# ===== 地形由使用者輸入 =====
from terrain import Terrain
terrain = Terrain()
terrain.ask_user()

PHI_MAX_DEG = None

class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, mass=0.1, friction_coef=0.0,
                 render_mode=None, diff_dx=5e-4, terrain_name=None, terrain_fn=None,
                 max_episode_steps: int = 500, force_mag: float = 20.0):
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

        self.max_episode_steps = int(max_episode_steps)
        self._step_count = 0

        high = np.array([self.x_threshold*2, np.inf, self.theta_threshold_radians*2, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def _phi_from_x(self, x: float):
        dx = self.diff_dx
        h  = self.terrain_fn
        slope = (h(x + dx) - h(x - dx)) / (2.0 * dx)
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

        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        reward = 1.0
        return np.array(self.state, dtype=np.float32), reward, bool(terminated), bool(truncated), {}

def create_env(length, mass, friction_coef, terrain_name, terrain_fn):
    return CartPoleWithVaryingParameters(length=length, mass=mass,
                                         friction_coef=friction_coef,
                                         terrain_name=terrain_name, terrain_fn=terrain_fn,
                                         max_episode_steps=500, force_mag=20.0)

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    header = f"L={length}, M={mass}, F={friction_coef}, Terrain={header_tag}"
    if tqdm: print(f"\n== Training [{header}] on {device}")
    else:    print(f"\n== Training [{header}] on {device}")

    env = DummyVecEnv([lambda: create_env(length, mass, friction_coef, terrain_name, terrain_fn)])
    model = PPO("MlpPolicy", env, verbose=0, device=device)

    total_steps = total_episodes * timesteps_per_episode
    pbar = tqdm(total=total_steps, desc=f"Train steps ({header})", leave=False) if tqdm else None

    for ep in range(1, total_episodes + 1):
        cb = TqdmStepCallback(pbar)
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False, callback=cb)
        if tqdm:
            test_env = create_env(length, mass, friction_coef, terrain_name, terrain_fn)
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

output_dir = "cartpole_data"
model_dir  = "models"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# ==== 逐一訓練使用者輸入的每一組地形 ====
combo_idx = 0
grid_total = max(1, len(terrain.terrain) * max(1, len(Length)) * max(1, len(Mass)) * max(1, len(Friction_coef)))
outer_bar = tqdm(total=grid_total, desc="Parameter grid", leave=True) if tqdm else None

for i in range(len(terrain.terrain)):
    TERRAIN_TAG   = terrain.get_tag(i)
    TERRAIN_FN    = terrain.get_fn(i)
    TERRAIN_NAME  = terrain.terrain[i]['Terrain Name']
    TERRAIN_PARAM = terrain.terrain[i]['Parameter']

    for length in Length:
        for mass in Mass:
            for friction_coef in Friction_coef:
                combo_idx += 1
                model = Model_Training(length, mass, friction_coef, TERRAIN_NAME, TERRAIN_FN, TERRAIN_TAG)

                model_path = os.path.join(
                    model_dir,
                    f"PPO_Terrain_{TERRAIN_TAG}_L{length}_M{mass}_F{friction_coef}.zip"
                )
                model.save(model_path)
                if tqdm: tqdm.write(f"Saved model: {model_path}")
                else:    print(f"Saved model: {model_path}")

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
                    pole_tip_offsets.append(test_env.unwrapped.length * np.sin(theta))

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
                    if step_count >= 500:
                        break

                test_env.close()
                if rec_bar: rec_bar.close()

                # === recorded_data（照你原本的鍵值）===
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

                    # 記錄正在訓練的那一組地形
                    'terrain_name': np.array([TERRAIN_NAME]),
                    'terrain_parameter': np.array([TERRAIN_PARAM], dtype=object)
                }

                # === 儲存 .npz 到原本 output_dir ===
                data_filename = f"Terrain_{TERRAIN_TAG}_PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz"
                full_path = os.path.join(output_dir, data_filename)
                np.savez(full_path, **recorded_data)

                # === 另外壓縮成 .zip 丟到指定路徑 ===
                import os, zipfile  # 就地匯入即可
                target_dir = r"C:\Users\kyleh\OneDrive\文件\HRV & CartPole\Cartpole_Terrain_Code\Sine_Terrain_Data"
                os.makedirs(target_dir, exist_ok=True)

                zip_filename = os.path.splitext(data_filename)[0] + ".zip"
                zip_path = os.path.join(target_dir, zip_filename)

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(full_path, os.path.basename(full_path))

                # === 訊息（沿用你原本的 tqdm/print 風格）===
                if tqdm:
                    tqdm.write(f"Saved data: {full_path}")
                    tqdm.write(f"Saved zip:  {zip_path}")
                    outer_bar.update(1)
                else:
                    print(f"Saved data: {full_path} ({combo_idx}/{grid_total})")
                    print(f"Saved zip:  {zip_path}")
