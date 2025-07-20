import warnings
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

warnings.filterwarnings('ignore')

#基本訓練參數設定
tau = 0.02 #時間步長
total_episodes = 50 #訓練次數
timesteps_per_episode = 100 #訓練步數
max_step_count = 500000 #總共執行步數

#輸入參數
Length = []
Mass = []
Friction_coef = [0]

#輸入Length
print("Length: ")
while True:
    x = float(input())
    if x != 0:
        Length.append(x)
    else:
        break

#輸入Mass
print("Mass: ")
while True:
    x = float(input())
    if x != 0:
        Mass.append(x)
    else:
        break

#輸入Friction_coef
print("Friction_coef: ")
while True:
    x = float(input())
    if x != 0:
        Friction_coef.append(x)
    else:
        break

#Creat Enviorment
class CartPoleWithVaryingParameters(CartPoleEnv):
    def __init__(self, length=1.0, mass=0.1, friction_coef=0.1, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.length = length / 2 # 設定桿長
        self.masspole = mass # 設定桿質量
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart
        self.friction_coef = friction_coef # 設定摩擦係數
        self.tau = tau

    def step(self, action):
        force = self.force_mag if action == 1 else -self.force_mag
        x, x_dot, theta, theta_dot = self.state
        # 地面摩擦力
        friction_force = -self.friction_coef * x_dot
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        sgn = lambda x: abs(x)/x #定義sgn function

        thetaacc = (self.gravity * sintheta + costheta * ((-force - 2 * self.polemass_length * theta_dot ** 2 * (sintheta + self.friction_coef * sgn(x_dot) * costheta)) / self.total_mass + self.friction_coef * self.gravity * sgn(x_dot))) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta (costheta - self.friction_coef * sgn(x_dot)) / self.total_mass))
        N = self.total_mass * self.gravity - 2 * self.polemass_length * (thetaacc * sintheta + theta_dot**2 * costheta)
        xacc = (force + 2 * self.polemass_length * (theta_dot**2 * sintheta - thetaacc * costheta) - self.friction_coef * N * sgn(x_dot)) / self.total_mass
        x = x + self.tau * x_dot + 1/2 * xacc * self.tau ** 2
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot + 1/2 * thetaacc * self.tau ** 2
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        terminated = x < -self.x_threshold \
                     or x > self.x_threshold \
                     or theta < -self.theta_threshold_radians \
                     or theta > self.theta_threshold_radians
        terminated = bool(terminated)
        reward = 1.0
        self.steps_beyond_terminated = None if not terminated else self.steps_beyond_terminated
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

def create_env(length, mass, friction_coef):
    return CartPoleWithVaryingParameters(length = length, mass = mass, friction_coef = friction_coef)

episode_rewards = []

def Model_Training(length, mass, friction_coef):
    print(f"\n== Pole Length: {length}, Pole Mass: {mass}, Friction Coefficient: {friction_coef}")
    env = DummyVecEnv([lambda: create_env(length, mass, friction_coef)])
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    # Training loop
    for episode in range(total_episodes):
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
        if (episode + 1) % 10 == 0:
            print("episode：", episode + 1)

    return model

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
                episode_rewards.append(reward)
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

            # Save data to numpy file
            data_filename = f"PoleLength_{length}_PoleMass_{mass}_Friction_{friction_coef}.npz"
            np.savez(data_filename, **recorded_data)