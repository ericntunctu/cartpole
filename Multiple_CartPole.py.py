import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import warnings
import random

warnings.filterwarnings("ignore")

total_episodes = 10000
timesteps_per_episode = 100000
max_step_count = 50000

output_dir = r"C:\Users\511\Documents\Cartpole\cartpole_data\2CartPole"

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

class MultipleCartPole(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, length, mass_pole, N, spring_length = 2.0, k = 2.0, render_mode=None,):
        super().__init__()
        self.render_mode = render_mode

        self.spring_length = spring_length
        self.k = k
        self.mass_cart = 1.0
        self.mass_pole = mass_pole
        self.length = length / 2
        self.force_mag = 20
        self.τ = 0.02
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length
        self.N = N

        high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max, np.pi / 2, np.finfo(np.float32).max] * N, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        low = np.array([-self.force_mag] * N, dtype=np.float32)
        high = np.array([self.force_mag] * N, dtype=np.float32)

        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(N,),
            dtype=np.float32
        )

        self.state = None
        self.steps_beyond_terminated = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        spring_length = self.spring_length
        N = self.N
        v0 = random.uniform(-0.05, 0.05)
        ω0 = random.uniform(-0.05, 0.05)
        initial_state = [[i * spring_length, v0, np.pi/36, ω0] for i in range(N)]
        self.state = [i for j in initial_state for i in j]
        self.steps_beyond_terminated = None
        observation = np.array(self.state, dtype=np.float32)
        info = {}
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action):

        spring_Length = self.spring_length
        k = self.k
        N = self.N
        force_mag = self.force_mag
        pole_mass_length = self.pole_mass_length
        total_mass = self.total_mass
        mass_pole= self.mass_pole
        length = self.length
        gravity = 9.81
        τ = self.τ

        state = np.array(self.state)
        x = state[0::4]
        x_dot = state[1::4]
        θ = state[2::4]
        θ_dot = state[3::4]

        f_ex = np.clip(action, -force_mag, force_mag)

        ΔL = np.diff(x) - spring_Length
        ΔL = np.concatenate(([0] , ΔL , [0]))
        F_spring = k * ΔL
        f = f_ex - np.diff(F_spring)

        def Euler(x, x_dot, θ, θ_dot, f):

            cosθ = np.cos(θ)
            sinθ = np.sin(θ)
            temp = (f + pole_mass_length * θ_dot ** 2 * sinθ) / total_mass
            θ_acc = (gravity * sinθ - cosθ * temp) / (length * (4.0 / 3.0 - mass_pole * cosθ ** 2 / total_mass))
            x_acc = temp - pole_mass_length * θ_acc * cosθ / total_mass

            x = x + τ * x_dot
            x_dot = x_dot + τ * x_acc
            θ = θ + τ * θ_dot
            θ_dot = θ_dot + τ * θ_acc

            return [x, x_dot, θ, θ_dot]

        state_updated = [
            Euler(x[i], x_dot[i], θ[i], θ_dot[i], f[i]) for i in range(N)
        ]

        self.state = [i for j in state_updated for i in j]

        x_updated = [state_updated[i][0] for i in range(N)]
        θ_updated = [state_updated[i][2] for i in range(N)]

        if any(angle > 0.2095 * 2 for angle in θ_updated) or any((Distance <= 1e-6 or Distance < 0) for Distance in np.diff(x_updated)):
            terminated = True
        else:
            terminated = False

        reward = 1.0 if not terminated else 0.0
        observation = np.array(self.state, dtype=np.float32)
        info = {}

        return observation, reward, terminated, False, info

def create_env(length, mass):
    return MultipleCartPole(length = length, mass_pole = mass, N = 1, spring_length=2.0, k = 2.0)

def Model_Training(length, mass):
    print(f"\n== Pole Length: {length}, Pole Mass: {mass} ==")
    env = DummyVecEnv([lambda: create_env(length, mass)])
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    for episode in range(total_episodes):
        model.learn(total_timesteps=timesteps_per_episode, reset_num_timesteps=False)
        if (episode + 1) % 10 == 0:
            print(f"episode: {episode + 1}")
    print("Finished Training")
    return model

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

def Run(length, mass, model):

    test_env = MultipleCartPole(length = length, mass_pole = mass, N = 1, spring_length = 2.0, k = 2.0)
    obs, info = test_env.reset(seed=42)

    cart_positions = []
    cart_velocities = []
    pole_angles = []
    pole_angular_velocities = []
    Force = []
    timestamps = []
    L = test_env.length

    for step in range(max_step_count):
        timestamps.append(step * test_env.τ)
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = test_env.step(action)
        x = obs[0::4]
        x_dot = obs[1::4]
        θ = obs[2::4]
        θ_dot = obs[3::4]
        cart_positions.append(x)
        cart_velocities.append(x_dot)
        pole_angles.append(θ)
        pole_angular_velocities.append(θ_dot)
        Force.append(action)

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
            "force": np.array(Force),
            "timestamps": np.array(timestamps),
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
        data = Run(length, mass, model)
        if data != False:
            Save_file(data, output_dir, length, mass)
            break