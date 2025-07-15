import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from scipy.fft import fft, fftfreq
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')

#設定基本參數
total_episodes = 60 #訓練次數
timestep_per_episode = 60000 #每次訓練步數
max_step_count = 500000 #總共執行步數
tau = 0.02

#輸入Length
Length = []
while True:
    x = float(input("The Length of pole: "))
    if x != 0:
        Length.append(x)
    else:
        break

# ======= Training =======

# Create environment
class CartPoleWithVaryingPoleLength(CartPoleEnv):
    def __init__(self, length=1.0, render_mode=None):  # 目的：更改Length
        super().__init__(render_mode=render_mode)
        self.length = length / 2
        self.polemass_length = self.masspole * self.length
        self.tau = tau

    def step(self, action):
        # 與原本相同的力（左右推）
        force = self.force_mag if action == 1 else -self.force_mag
        x, x_dot, theta, theta_dot = self.state
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 狀態更新
        x = x + self.tau * x_dot + 1 / 2 * xacc * self.tau ** 2
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        self.theta_threshold_radians = 24 * 2 * math.pi / 360

        terminated = x < -self.x_threshold \
                     or x > self.x_threshold \
                     or theta < -self.theta_threshold_radians \
                     or theta > self.theta_threshold_radians
        terminated = bool(terminated)
        reward = 1.0
        self.steps_beyond_terminated = None if not terminated else self.steps_beyond_terminated
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

def make_env(Length):
    return CartPoleWithVaryingPoleLength(Length)

log_xf_list = []
log_power_list = []

def Model_Training(length):
    print(f"Training with Pole Length {length}")
    env = DummyVecEnv([lambda: make_env(length)])
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    # Training loop
    for episode in range(total_episodes):
        model.learn(total_timesteps=timestep_per_episode, reset_num_timesteps=False)
        if (episode + 1) % 10 == 0:
            print("episode：", episode + 1)

    return model

# ======= Analysis =======

for length in Length:
    print("\n======= Training =======\n")
    model = Model_Training(length)

    # Data recording arrays
    cart_positions = []
    cart_velocities = []
    pole_angles = []
    pole_angular_velocities = []
    actions_taken = []
    rewards_received = []
    timestamps = []
    pole_tip_offsets = []

    # Collect data for analysis
    test_env = CartPoleWithVaryingPoleLength(length=1.0, render_mode="rgb_array")
    obs, info = test_env.reset()
    done = False
    step_count = 20
    total_reward = 0

    while not done and step_count < max_step_count:
        # Record current state before action
        cart_position = obs[0]
        cart_velocity = obs[1]
        pole_angle = obs[2]
        pole_angular_velocity = obs[3]

        cart_positions.append(cart_position)
        cart_velocities.append(cart_velocity)
        pole_angles.append(pole_angle)
        pole_angular_velocities.append(pole_angular_velocity)
        timestamps.append(step_count * 0.02)
        pole_tip_offset = test_env.unwrapped.length * np.sin(pole_angle)
        pole_tip_offsets.append(pole_tip_offset)

        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)
        actions_taken.append(action)

        # Step environment
        obs, reward, terminated, truncated, info = test_env.step(action)
        rewards_received.append(reward)
        total_reward += reward
        done = terminated or truncated

        step_count += 1

    test_env.close()

    # Convert lists to numpy arrays
    cart_positions = np.array(cart_positions)
    cart_velocities = np.array(cart_velocities)
    pole_angles = np.array(pole_angles)
    pole_angular_velocities = np.array(pole_angular_velocities)
    actions_taken = np.array(actions_taken)
    rewards_received = np.array(rewards_received)
    timestamps = np.array(timestamps)
    pole_tip_offsets = np.array(pole_tip_offsets)

    # Performance analysis
    pole_half_length = test_env.unwrapped.length
    pole_full_length = pole_half_length * 2
    zero_crossing_indices = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]

    # 加上 1，因為 np.diff 產生的是 N-1 長度
    zero_crossing_times = timestamps[zero_crossing_indices + 1]

    # 假設你已有 zero_crossing_times，如：
    # zero_crossing_times = np.array([...])
    # Step 1: 取偶數 index 的過零時間點
    even_crossing_times = zero_crossing_times[::2]

    # Step 2: 計算每個週期的時間長度
    periods = np.diff(even_crossing_times)  # 相鄰時間差，應該接近週期
    N = len(periods)
    dt = 1  # 每個樣本對應一個週期，不需要真實時間尺度

    if len(periods) < 2:
        print(f"[Warning] Not enough zero crossings for length = {length}")
        break

    # Step 3: FFT 分析
    yf = fft(periods - np.mean(periods))  # 去除 DC 分量 (均值)
    xf = fftfreq(N, dt)

    # 只取正頻率部分
    xf_pos = xf[:N // 2]
    power_spectrum = np.abs(yf[:N // 2]) ** 2

    # 假設 you have periods = np.diff(even_crossing_times)
    # periods = periods - np.mean(periods)  # 去掉均值
    N = len(periods)
    dt = 1  # index 間隔
    xf = xf[:N // 2]
    power = np.abs(yf[:N // 2]) ** 2

    epsilon = 1e-12
    log_xf = np.log(xf[1:] + epsilon)
    log_power = np.log(power[1:] + epsilon)

    log_xf_list.append(log_xf.tolist())
    log_power_list.append(log_power.tolist())

# plt.figure(figsize=(8, 5))
#
# for i in range(len(Length)):  # 逐列畫圖
#     plt.scatter(log_xf_list[i], log_power_list[i], label=f"Length = {Length[i]}")
#
# plt.xlabel("log(frequency)")
# plt.ylabel("log(power)")
# plt.title("CartPole System with Varinging PoleLength")
# plt.legend()
# plt.grid(True)
# plt.show()

for i in range(len(Length)):
    plt.figure(figsize=(8, 5))
    plt.scatter(log_xf_list[i], log_power_list[i],
                s=10, alpha=0.7)

    plt.xlabel("log(frequency)")
    plt.ylabel("log(power)")
    plt.title(f"CartPole System – PoleLength = {Length[i]}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()