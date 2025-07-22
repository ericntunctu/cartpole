#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install stable_baselines3')


# In[1]:


print('A') # 防止jupyter notebook有問題


# In[2]:


get_ipython().system('pip install scipy')


# In[3]:


import numpy as np
print(np.__version__)


# In[4]:


get_ipython().system('pip install torch')


# In[2]:


import torch
print(torch.cuda.is_available())  # 應該顯示 True
print(torch.cuda.get_device_name(0))  # 顯示你的 GPU 名稱


# In[2]:


import warnings
import numpy as np
import os  # 用來處理檔案和目錄
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')

# 基本訓練參數設定
tau = 0.02  # 時間步長
total_episodes = 50  # 訓練次數
timesteps_per_episode = 100  # 訓練步數
max_step_count = 500000  # 總共執行步數

# 輸入參數
Length = []
Mass = []
Friction_coef = [0]

# 輸入 Length
print("Length: ")
while True:
    x = float(input())
    if x != 0:
        Length.append(x)
    else:
        break

# 輸入 Mass
print("Mass: ")
while True:
    x = float(input())
    if x != 0:
        Mass.append(x)
    else:
        break

# 輸入 Friction_coef
print("Friction_coef: ")
while True:
    x = float(input())
    if x != 0:
        Friction_coef.append(x)
    else:
        break

# Create Environment
class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, mass=0.1, friction_coef=0.1, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.length = length / 2  # 設定桿長
        self.masspole = mass  # 設定桿質量
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart
        self.friction_coef = friction_coef  # 設定摩擦係數
        self.tau = tau

    def step(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        x, x_dot, theta, theta_dot = self.state
        # 地面摩擦力
        friction_force = -self.friction_coef * x_dot
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        sgn = lambda x: abs(x) / x if x != 0 else 0  # 定義 sgn function，處理 x = 0 的情況

        thetaacc = (self.gravity * sintheta + costheta * ((-force - 2 * self.polemass_length * theta_dot ** 2 * (sintheta + self.friction_coef * sgn(x_dot) * costheta)) / self.total_mass + self.friction_coef * self.gravity * sgn(x_dot))) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * (costheta - self.friction_coef * sgn(x_dot)) / self.total_mass))
        N = self.total_mass * self.gravity - 2 * self.polemass_length * (thetaacc * sintheta + theta_dot ** 2 * costheta)
        xacc = (force + 2 * self.polemass_length * (theta_dot ** 2 * sintheta - thetaacc * costheta) - self.friction_coef * N * sgn(x_dot)) / self.total_mass
        x = x + self.tau * x_dot + 0.5 * xacc * self.tau ** 2
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot + 0.5 * thetaacc * self.tau ** 2
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        terminated = x < -self.x_threshold or x > self.x_threshold or \
                     theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        terminated = bool(terminated)
        reward = 1.0
        self.steps_beyond_terminated = None if not terminated else self.steps_beyond_terminated
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

def create_env(length, mass, friction_coef):
    return CartPoleWithVaryingParameters(length=length, mass=mass, friction_coef=friction_coef)

episode_rewards = []

def Model_Training(length, mass, friction_coef):
    print(f"\n== Pole Length: {length}, Pole Mass: {mass}, Friction Coefficient: {friction_coef}")
    env = DummyVecEnv([lambda: create_env(length, mass, friction_coef)])
    model = PPO("MlpPolicy", env, verbose=0, device="cuda")

    # Training loop
    for episode in range(total_episodes):
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
        if (episode + 1) % 10 == 0:
            print("episode：", episode + 1)

    return model

# 創建輸出資料夾
output_dir = "cartpole_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for length in Length:
    for mass in Mass:
        for friction_coef in Friction_coef:
            model = Model_Training(length, mass, friction_coef)
            test_env = CartPoleWithVaryingParameters(length, mass, friction_coef, render_mode="rgb_array")

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
            obs, info = test_env.reset()
            done = False
            step_count = 0  # 從 0 開始
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
                timestamps.append(step_count * tau)
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

            recorded_data = {
                'cart_positions': cart_positions,
                'cart_velocities': cart_velocities,
                'pole_angles': pole_angles,
                'pole_angular_velocities': pole_angular_velocities,
                'actions_taken': actions_taken,
                'rewards_received': rewards_received,
                'timestamps': timestamps,
                'total_steps': step_count,
                'total_reward': total_reward,
                'pole_tip_offsets': pole_tip_offsets,
                'episode_rewards': np.array(episode_rewards)
            }

            # Save data to numpy file in the cartpole_data directory
            data_filename = f"PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz"
            full_path = os.path.join(output_dir, data_filename)
            np.savez(full_path, **recorded_data)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq



def fourier_analysis(path):
    data = np.load(path)

    cart_pos = data["cart_positions"]
    cart_vel = data["cart_velocities"]
    pole_ang = data["pole_angles"]
    pole_ang_vel = data["pole_angular_velocities"]
    pole_tip_offsets = data["pole_tip_offsets"]
    timestamps = data["timestamps"]
    
    zero_crossing_indices = np.where(np.diff(np.sign(pole_tip_offsets)) != 0)[0]
    
    # 加上 1，因為 np.diff 產生的是 N-1 長度
    zero_crossing_times = timestamps[zero_crossing_indices + 1]
    plt.plot(timestamps[:500], pole_tip_offsets[:500], label='Pole Tip Offset')
    plt.scatter(zero_crossing_times[:500], np.zeros_like(zero_crossing_times)[:500], color='red', label='Zero Crossings')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Pole Tip Horizontal Offset")
    plt.legend()
    plt.title("Pole Tip Oscillation and Zero Crossings")
    plt.grid(True)
    plt.show()

    
    # 假設你已有 zero_crossing_times，如：
    # zero_crossing_times = np.array([...])
    
    # Step 1: 取偶數 index 的過零時間點
    even_crossing_times = zero_crossing_times[::2]
    # Step 2: 計算每個週期的時間長度
    periods = np.diff(even_crossing_times)  # 相鄰時間差，應該接近週期
    N = len(periods)
    dt = 1  # 每個樣本對應一個週期，不需要真實時間尺度
    
    # Step 3: FFT 分析
    yf = fft(periods)  # 去除 DC 分量 (均值)
    xf = fftfreq(N, dt)
    
    # 只取正頻率部分
    xf_pos = xf[:N//2]
    power_spectrum = np.abs(yf[:N//2])**2
    
    # Step 4: 繪圖
    plt.figure(figsize=(10, 5))
    plt.plot(xf_pos, power_spectrum)
    plt.xlabel("Frequency (1/crossings)")
    plt.ylabel("Power")
    plt.title("Fourier Spectrum of Zero-Crossing Periods")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    # 假設 you have `periods` = np.diff(even_crossing_times)
    periods = periods - np.mean(periods)  # 去掉均值
    N = len(periods)
    dt = 1  # index 間隔
    
    yf = fft(periods)
    xf = fftfreq(N, dt)
    xf = xf[:N//2]
    power = np.abs(yf[:N//2])**2
    Q = N
    # 避免 log(0)，加個 epsilon
    epsilon = 1e-12
    log_xf = np.log10(xf[2:Q] + epsilon)
    log_power = np.log10(power[2:Q] + epsilon)
    
    plt.figure(figsize=(8, 5))
    plt.plot(log_xf, log_power, label='Power Spectrum (log-log)')
    plt.xlabel('log10(Frequency)')
    plt.ylabel('log10(Power)')
    plt.title('Low-frequency behavior of Power Spectrum')
    plt.grid(True)
    plt.legend()
    plt.show()

# test
path = "PoleLength_0.5_PoleMass_0.05_Friction_0.npz"
fourier_analysis(path)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




