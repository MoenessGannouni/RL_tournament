import os
from stable_baselines3 import PPO

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "ppo-LunarLander-v3.zip")
model = PPO.load(model_path)

def agent(observation):
    # Given an observation, predict the best action
    action, _states = model.predict(observation, deterministic=True)
    return int(action)
