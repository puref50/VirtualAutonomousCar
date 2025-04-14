from stable_baselines3 import PPO
from test_environment import CarEnv

#update here
models_dir = "E:\\project\\models\\1744255933"

env = CarEnv()
env.reset()

#and update here
model_path = f"{models_dir}\\5000000.zip"
model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(reward)