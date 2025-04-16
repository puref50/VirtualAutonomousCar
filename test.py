# to test a model 
from stable_baselines3 import PPO
from test_environment import CarEnv
import numpy as np


models_dir = "E:\\project\\models\\1744255933"

env = CarEnv()
env.reset()

def evaluate_policy(env, model, episodes):
    total_rewards = []
    total_distances = []
    total_invasions = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        total_rewards.append(info["episode_reward"])
        total_distances.append(info["episode_distance"])
        total_invasions.append(info["lane_invasions"])

        print(f"[EP {ep+1}] Reward: {info['episode_reward']:.2f}, "
              f"Distance: {info['episode_distance']:.2f}m, "
              f"Invasions: {info['lane_invasions']}")

    print("\n=== Evaluation Summary ===")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Average distance: {np.mean(total_distances):.2f} m")
    print(f"Average lane invasions: {np.mean(total_invasions):.2f}")

model_path = f"{models_dir}\\5000000.zip"
model = PPO.load(model_path, env=env)
evaluate_policy(env, model, 100)