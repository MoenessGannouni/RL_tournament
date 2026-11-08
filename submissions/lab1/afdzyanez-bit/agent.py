from pathlib import Path
from stable_baselines3 import PPO

model_path = Path(__file__).with_name("ppo-LunarLander-v3.zip")
model = PPO.load(str(model_path), device="cpu")

def agent(observation):
    action, _states = model.predict(observation, deterministic=True)
    return int(action)
