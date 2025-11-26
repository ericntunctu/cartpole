import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import warnings
warnings.filterwarnings("ignore")

# ==================== 參數設定 ====================
# ==================== 參數設定 ====================
total_episodes = 500            # ← 改這裡！從 10000 → 500
timesteps_per_episode = 500     # ← 改這裡！從 100000 → 500
max_step_count = 50000          # 測試要撐滿 50000 步不變
output_dir = "data"   # 就在你 .py 檔旁邊自動生一個 data 資料夾
# 或


# 彈簧參數（可以自行調整）
SPRING_K = 8.0          # 彈簧常數，越大耦合越強
SPRING_LENGTH = 2.0     # 平衡時兩車距離

# ==================== 環境本體 ====================
class CoupledCartPole(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, length=1.0, mass_pole=0.1, N=3, render_mode=None):
        super().__init__()
        self.N = N
        self.length = length / 2          # 標準 CartPole 用半長
        self.mass_pole = mass_pole
        self.mass_cart = 1.0
        self.g = 9.81
        self.tau = 0.02
        self.force_mag = 10.0             # 標準 CartPole 推力

        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length

        # 狀態： [x1, x_dot1, θ1, θ_dot1, x2, ...] → 4N 維
        high = np.array([np.finfo(np.float32).max] * 4 * N, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 動作：每個車獨立控制
        self.action_space = spaces.Box(-self.force_mag, self.force_mag, shape=(N,), dtype=np.float32)

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 初始位置等距排列 + 小擾動
        x0 = np.linspace(0, (self.N-1)*SPRING_LENGTH, self.N)
        x0 += self.np_random.uniform(-0.05, 0.05, size=self.N)

        state = np.zeros(4 * self.N)
        state[0::4] = x0
        state[1::4] = self.np_random.uniform(-0.5, 0.5, size=self.N)
        state[2::4] = self.np_random.uniform(-0.1, 0.1, size=self.N)   # θ 靠近直立
        state[3::4] = self.np_random.uniform(-0.5, 0.5, size=self.N)

        self.state = state.astype(np.float32)
        return self.state.copy(), {}

    def step(self, action):
        x         = self.state[0::4]
        x_dot     = self.state[1::4]
        theta     = self.state[2::4]
        theta_dot = self.state[3::4]

        force = np.clip(action, -self.force_mag, self.force_mag)

        # ============= 正確的彈簧力 =============
        spring_force = np.zeros(self.N)
        if self.N > 1:
            # 左鄰居拉力
            left_pull = SPRING_K * ((x[1:] - x[:-1]) - SPRING_LENGTH)
            spring_force[1:] += left_pull
            spring_force[:-1] -= left_pull
        # ======================================

        total_force = force + spring_force

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (total_force + self.pole_mass_length * theta_dot**2 * sintheta) / self.total_mass
        theta_acc = (self.g * sintheta - costheta * temp) / \
                    (self.length * (4/3 - self.mass_pole * costheta**2 / self.total_mass))
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass

        # Euler 積分
        x       += self.tau * x_dot
        x_dot   += self.tau * x_acc
        theta   += self.tau * theta_dot
        theta_dot += self.tau * theta_acc

        self.state = np.concatenate([x, x_dot, theta, theta_dot]).astype(np.float32)

        # ============= 失敗條件（標準 + 防穿模）=============
        angle_fail = np.any(np.abs(theta) > np.deg2rad(12))
        pos_fail   = np.any(np.abs(x) > 2.4)
        collision  = self.N > 1 and np.any(np.diff(x) <= 0.1)
        terminated = bool(angle_fail or pos_fail or collision)
        # =================================================

        reward = 1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), reward, terminated, False, {}

    def render(self):
        # 簡單文字視覺化（正式要畫圖再補）
        print(f"\rPositions: {x.round(2)}  Angles: {np.rad2deg(theta).round(1)}", end="")

# ==================== 工具函數 ====================
def create_env(length, mass, N):
    return lambda: CoupledCartPole(length=length, mass_pole=mass, N=N)

def train_model(length, mass, N):
    print(f"\n=== Training Length={length}, Mass={mass}, N={N} ===")
    env = DummyVecEnv([create_env(length, mass, N)])
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=total_episodes * timesteps_per_episode)
    print("Training finished!")
    return model

def get_next_run_number(length, mass, N, output_dir):
    prefix = f"L{length}_M{mass}_N{N}_"
    existing = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith(".npz")]
    nums = [int(f[len(prefix):-4]) for f in existing if f[len(prefix):-4].isdigit()]
    return max(nums, default=0) + 1

def test_and_save(length, mass, N, model, output_dir):
    env = CoupledCartPole(length=length, mass_pole=mass, N=N)
    obs, _ = env.reset(seed=42)

    data = {
        "cart_positions": [], "cart_velocities": [], "pole_angles": [], "pole_angular_velocities": [],
        "forces": [], "timestamps": [], "spring_forces": []
    }

    for step in range(max_step_count):
        t = step * env.tau
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        x      = obs[0::4]
        x_dot  = obs[1::4]
        theta  = obs[2::4]
        theta_dot = obs[3::4]

        # 計算當下彈簧力（方便後續分析）
        spring_f = np.zeros(N)
        if N > 1:
            left = SPRING_K * ((x[1:] - x[:-1]) - SPRING_LENGTH)
            spring_f[1:] += left
            spring_f[:-1] -= left

        data["cart_positions"].append(x.copy())
        data["cart_velocities"].append(x_dot.copy())
        data["pole_angles"].append(theta.copy())
        data["pole_angular_velocities"].append(theta_dot.copy())
        data["forces"].append(action.copy())
        data["spring_forces"].append(spring_f.copy())
        data["timestamps"].append(t)

        if terminated:
            print(f"  → 只撐了 {step+1} 步，失敗，重訓...")
            return False

    print(f"  → 撐滿 {max_step_count} 步！成功！正在存檔...")
    for key in data:
        data[key] = np.array(data[key])

    run_num = get_next_run_number(length, mass, N, output_dir)
    path = os.path.join(output_dir, f"L{length}_M{mass}_N{N}_{run_num}.npz")
    np.savez_compressed(path, **data)
    print(f"  → 存檔完成：{os.path.basename(path)}")
    return True

# ==================== 主程式 ====================
os.makedirs(output_dir, exist_ok=True)

print("輸入格式：length, mass, N, run_number  (輸入 0 結束)")
combinations = []
while True:
    s = input().strip()
    if s == "0":
        break
    l, m, n, runs = map(float, s.split(","))
    n = int(n)
    runs = int(runs)
    for _ in range(runs):
        combinations.append((l, m, n))

for length, mass, N in combinations:
    while True:
        model = train_model(length, mass, N)
        if test_and_save(length, mass, N, model, output_dir):
            break