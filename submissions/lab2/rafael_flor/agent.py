import os
import numpy as np
import pickle

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "q-learning.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
    
Qtable = model["qtable"]

def agent(observation):
    # Select the action with the maximum expected reward
    action = np.argmax(Qtable[observation][:])
    return int(action)
