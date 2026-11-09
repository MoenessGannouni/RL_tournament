
import os
from stable_baselines3 import DQN

# Cargar el modelo DQN definitivo
model_path = os.path.join(
    os.path.dirname(__file__),
    "ppo-LunarLander-v3.zip"
)

model = DQN.load(model_path)

def agent(observation):
    action, _states = model.predict(
        observation,
        deterministic=True
    )
    return int(action)
