import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import warnings

warnings.filterwarnings("ignore")

total_episodes = 200
timesteps_per_episode = 1000
max_step_count = 500000

output_dir = r"C:\Users\511\Documents\Cartpole\cartpole_data\Varying_Force"

combinations = []
print("Enter [length, mass, Number] combinations:")
while True:
    x = input().strip(" ").split(",")
    if x[0] == "0":
        break

    length = float(x[0])
    mass = float(x[1])
    run_number = int(x[2])

    for i in range(run_number):
        combinations.append([length, mass])

class CartPole(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, length=1, mass_pole=0.1, render_mode=None,):
        super().__init__()
        self.render_mode = render_mode

        self.mass_cart = 1.0
        self.mass_pole = mass_pole
        self.length = length / 2
        self.force_mag = 100.0
        self.τ = 0.02
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length

        high = np.array([4.8, np.finfo(np.float32).max, np.pi / 2, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-self.force_mag], dtype=np.float32),
            high=np.array([self.force_mag], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )
        self.state = None
        self.steps_beyond_terminated = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_terminated = None
        observation = np.array(self.state, dtype=np.float32)
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):

        force_mag = self.force_mag
        pole_mass_length = self.pole_mass_length
        total_mass = self.total_mass
        mass_pole= self.mass_pole
        length = self.length
        gravity = 9.81
        τ = self.τ

        f = np.clip(action[0], -force_mag, force_mag)
        x, x_dot, θ, θ_dot = self.state
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)
        temp = (f + pole_mass_length * θ_dot ** 2 * sinθ) / total_mass
        θ_acc = (gravity * sinθ - cosθ * temp) / (length * (4.0 / 3.0 - mass_pole * cosθ ** 2 / total_mass))
        x_acc = temp - pole_mass_length * θ_acc * cosθ / total_mass

        x = x + τ * x_dot
        x_dot = x_dot + τ * x_acc
        θ = θ + τ * θ_dot
        θ_dot = θ_dot + τ * θ_acc

        self.state = (x, x_dot, θ, θ_dot)

        terminated = bool(abs(θ) > 0.2095 or abs(x) > 2.4)

        reward = 1.0 if not terminated else 0.0
        observation = np.array(self.state, dtype=np.float32)
        info = {}

        return observation, reward, terminated, False, info

def create_env(length, mass):
    return CartPole(length = length, mass_pole = mass)

def Model_Training(length, mass):
    print(f"\n== Pole Length: {length}, Pole Mass: {mass}")
    try:
        env = DummyVecEnv([lambda: create_env(length, mass)])
        model = PPO("MlpPolicy", env, verbose=0, device="cpu")
        for episode in range(total_episodes):
            model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
            if (episode + 1) % 10 == 0:
                print(f"episode: {episode + 1}")
        print("Finished Training")
        return model
    except Exception as e:
        print("Failure")
        raise

def get_next_run_number(length, mass, output_dir):
    max_number = 0
    expected_prefix = f"Length_{length}_Mass_{mass}_"
    for filename in os.listdir(output_dir):
        if filename.endswith(".npz") and filename.startswith(expected_prefix):
            try:
                run_number_str = filename[len(expected_prefix):-4]
                run_number = int(run_number_str)
                max_number = max(max_number, run_number)
            except ValueError:
                continue
    return max_number + 1

def Run(length, mass, model, output_dir):

    test_env = CartPole(length = length, mass_pole = mass)
    obs, info = test_env.reset(seed=42)

    cart_positions = []
    cart_velocities = []
    pole_angles = []
    pole_angular_velocities = []
    Force = []
    timestamps = []
    pole_tip_offsets = []
    L = test_env.length

    for step in range(max_step_count):
        timestamps.append(step * test_env.τ)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(action)
        x, x_dot, θ, θ_dot = obs
        tip_offset_x = x + L * np.sin(θ)
        pole_tip_offsets.append(tip_offset_x)
        cart_positions.append(x)
        cart_velocities.append(x_dot)
        pole_angles.append(θ)
        pole_angular_velocities.append(θ_dot)
        Force.append(action[0])

        if terminated:
            print("本資料訓練失敗")
            break

    test_env.close()

    if not terminated:
        data = {
            "cart_positions": np.array(cart_positions),
            "cart_velocities": np.array(cart_velocities),
            "pole_angles": np.array(pole_angles),
            "pole_angular_velocities": np.array(pole_angular_velocities),
            "actions_taken": np.array(Force),
            "timestamps": np.array(timestamps),
            "pole_tip_offsets": np.array(pole_tip_offsets)
        }
        return data
    else:
        return False

def Save_file(data, output_dir, length, mass):
    run_number = get_next_run_number(length, mass, output_dir)
    filename = f"Length_{length}_Mass_{mass}_{run_number}.npz"
    full_path = os.path.join(output_dir, filename)
    np.savez(full_path, **data)

for combination in combinations:
    length = combination[0]
    mass = combination[1]

    while True:
        model = Model_Training(length, mass)
        data = Run(length, mass, model, output_dir)
        if data != False:
            Save_file(data, output_dir, length, mass)
            break