import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from scipy.fft import fft, fftfreq
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')

tau = 0.02

class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, pole_mass=0.1, friction_coef=0.1, render_mode=None):
        """
        整合的CartPole環境，同時支持可變桿長、桿質量和摩擦力
        
        Args:
            length: 桿的長度（米）
            pole_mass: 桿的質量（公斤）
            friction_coef: 摩擦係數
            render_mode: 渲染模式
        """
        super().__init__(render_mode=render_mode)
        
        # 設定桿長相關參數
        self.length = length / 2
        
        # 設定桿質量
        self.masspole = pole_mass
        self.polemass_length = self.masspole * self.length
        
        # 重新計算總質量
        self.total_mass = self.masspole + self.masscart
        
        # 設定摩擦係數
        self.friction_coef = friction_coef
        
        # 設定時間步長
        self.tau = tau

    def step(self, action):
        # 與原本相同的力（左右推）
        force = self.force_mag if action == 1 else -self.force_mag
        x, x_dot, theta, theta_dot = self.state

        # 添加地面摩擦力：對 cart 的速度做衰減
        friction_force = -self.friction_coef * x_dot
        force += friction_force

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # 計算加速度（使用更新後的桿長）
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # 狀態更新（使用第一段代碼的更新方式，包含加速度項）
        x = x + self.tau * x_dot + 1 / 2 * xacc * self.tau ** 2
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        # 設定角度閾值
        self.theta_threshold_radians = 24 * 2 * math.pi / 360

        # 檢查是否終止
        terminated = x < -self.x_threshold \
                     or x > self.x_threshold \
                     or theta < -self.theta_threshold_radians \
                     or theta > self.theta_threshold_radians
        terminated = bool(terminated)
        
        reward = 1.0
        self.steps_beyond_terminated = None if not terminated else self.steps_beyond_terminated
        
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

# ======= 使用範例 =======

def create_env(length, pole_mass, friction_coef):
    """創建整合的CartPole環境"""
    return CartPoleWithVaryingParameters(length=length, pole_mass=pole_mass, friction_coef=friction_coef)




env =  create_env(length=1.0, pole_mass=0.1, friction_coef=0.1)
model = PPO("MlpPolicy", env, verbose=0, device="cpu")
# Training parameters
total_episodes = 1
timesteps_per_episode = 500
episode_rewards = []

# Training loop
for episode in range(total_episodes):
    model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
    print("finish train")
    # Test current model performance
    test_env = env
    obs, info = test_env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        done = terminated or truncated
    
    episode_rewards.append(total_reward)
    test_env.close()
    
    # Show progress every 10 episodes
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}: Recent 10 episodes average reward = {avg_reward:.2f}")

print(f"\nTraining completed! Final average reward: {np.mean(episode_rewards[-10:]):.2f}")
