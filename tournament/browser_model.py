"""Container-only export of stateless, discrete policies to bounded declarative JSON.

Never import this module or decode student model files on the CI host.
"""
import json
from pathlib import Path


def export_policy(agent, lab):
    import numpy as np
    actions = 4 if lab == 1 else 6
    declared = Path('browser_policy.json')
    if declared.is_file():
        if declared.stat().st_size > 2_000_000:
            raise ValueError('Browser policy exceeds 2 MB')
        model = json.loads(declared.read_text())
    elif lab == 2:
        q = None
        # Check for pickle-based Q-tables first (as saved by Lab 2 course notebook), then raw numpy
        pkl_candidates = [Path('q-learning.pkl'), Path('qtable.pkl'), *Path('.').glob('*.pkl'), *Path('.').glob('*.pickle')]
        for pkl_file in pkl_candidates:
            if pkl_file.is_file():
                import pickle
                with open(pkl_file, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, dict) and 'qtable' in raw:
                    q = np.asarray(raw['qtable'], dtype=np.float64)
                elif isinstance(raw, np.ndarray):
                    q = np.asarray(raw, dtype=np.float64)
                if q is not None and q.shape == (500, 6):
                    break
                q = None
        if q is None and Path('qtable.npy').is_file():
            q = np.load('qtable.npy', allow_pickle=False)
        if q is None:
            raise ValueError('Add browser_policy.json, q-learning.pkl or qtable.npy')
        if q.shape != (500, 6) or not np.isfinite(q).all():
            raise ValueError('Expected finite Taxi Q-table with shape (500, 6)')
        model = {'kind': 'qtable', 'values': q.tolist()}
    elif lab == 1 and len(list(Path('.').glob('*.zip'))) == 1:
        import torch
        zip_path = next(Path('.').glob('*.zip'))
        custom_objects = {
            'learning_rate': 0.0,
            'lr_schedule': lambda _: 0.0,
            'clip_range': lambda _: 0.2,
        }
        policy = None
        raw_layers = None
        try:
            from stable_baselines3 import PPO
            model = PPO.load(
                zip_path,
                device='cpu',
                custom_objects=custom_objects,
            )
            policy = model.policy
            raw_layers = [*policy.mlp_extractor.policy_net, policy.action_net]
        except Exception:
            try:
                from stable_baselines3 import DQN
                model = DQN.load(
                    zip_path,
                    device='cpu',
                    custom_objects=custom_objects,
                )
                policy = model.policy
                raw_layers = list(policy.q_net.q_net)
            except Exception:
                try:
                    from stable_baselines3 import A2C
                    model = A2C.load(
                        zip_path,
                        device='cpu',
                        custom_objects=custom_objects,
                    )
                    policy = model.policy
                    raw_layers = [*policy.mlp_extractor.policy_net, policy.action_net]
                except Exception:
                    pass
        if policy is None or raw_layers is None:
            raise ValueError('Expected discrete LunarLander model (PPO, DQN, A2C)')
        if policy.observation_space.shape != (8,) or policy.action_space.n != 4:
            raise ValueError('Expected discrete LunarLander model')
        layers = []
        for layer in raw_layers:
            if isinstance(layer, torch.nn.Linear):
                layers.append({'weight': layer.weight.detach().cpu().tolist(),
                               'bias': layer.bias.detach().cpu().tolist(), 'activation': 'linear'})
            elif isinstance(layer, (torch.nn.Tanh, torch.nn.ReLU)) and layers:
                layers[-1]['activation'] = 'tanh' if isinstance(layer, torch.nn.Tanh) else 'relu'
            else:
                raise ValueError('Export supports Linear, Tanh and ReLU MLP layers')
        model = {'kind': 'mlp', 'layers': layers}
    else:
        raise ValueError('Add browser_policy.json, q-learning.pkl, qtable.npy or a discrete RL MLP ZIP')
    # Independently compare the exported forward pass with the submitted agent.
    # This catches preprocessing, different checkpoints and custom feature extractors.
    if model.get('kind') == 'random':
        # Stochastic intent must be explicit in the supplied browser_policy.json.
        if model != {'kind': 'random', 'actions': actions}:
            raise ValueError('Invalid random policy')
        return {'model': model, 'probes': []}
    observations = list(range(500)) if lab == 2 else [
        *np.random.default_rng(813).uniform(-1, 1, (128, 8)).astype(np.float32).tolist(),
        [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1]]
    probes = []
    for obs in observations:
        if model['kind'] == 'qtable':
            scores = model['values'][obs]
        else:
            x = np.asarray(obs, dtype=np.float32)
            for layer in model['layers']:
                x = np.asarray(layer['weight'], dtype=np.float32) @ x + np.asarray(layer['bias'], dtype=np.float32)
                if layer['activation'] == 'tanh':
                    x = np.tanh(x)
                elif layer['activation'] == 'relu':
                    x = np.maximum(x, 0)
            scores = x
        predicted = int(np.argmax(scores))
        actual = agent(np.asarray(obs, dtype=np.float32) if lab == 1 else obs)
        if isinstance(actual, (bool, np.bool_)) or not isinstance(actual, (int, np.integer)) or int(actual) != predicted:
            raise ValueError('Browser export disagrees with agent.py; supply a compatible browser_policy.json')
        probes.append({'observation': obs, 'action': predicted})
    return {'model': model, 'probes': probes}
