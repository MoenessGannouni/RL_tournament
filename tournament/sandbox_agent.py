#!/usr/bin/env python3
"""Container-only adapter: untrusted imports and model decoding happen here."""
import importlib.util
import json
import os
import sys


def main():
    # Preserve a protocol descriptor, discard student output (including native writes).
    protocol = os.fdopen(os.dup(1), 'w', buffering=1)
    null_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    sys.path.insert(0, '/submission')
    os.chdir('/submission')
    import numpy as np
    spec = importlib.util.spec_from_file_location('student_agent', '/submission/agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = getattr(module, 'agent')
    if not callable(agent):
        raise TypeError('agent must be callable')
    protocol.write('{"ready":true}\n')
    if sys.argv[1:] == ['--browser-export']:
        sys.path.insert(0, "/opt")
        from browser_model import export_policy
        request = json.loads(sys.stdin.buffer.readline(4096))
        try:
            result = export_policy(agent, request['lab'])
        except Exception:
            result = {'error': 'No compatible browser export. Supply browser_policy.json or a supported Q-table/PPO MLP.'}
        protocol.write(json.dumps(result, allow_nan=False) + '\n')
        return
    for line in sys.stdin.buffer:
        if len(line) > 4096:
            raise ValueError('Observation exceeds protocol limit')
        request = json.loads(line)
        observation = request['observation']
        if isinstance(observation, list):
            observation = np.asarray(observation, dtype=np.float32)
        action = agent(observation)
        # Do not silently truncate floats or accept booleans as actions.
        if isinstance(action, (bool, np.bool_)) or not isinstance(action, (int, np.integer)):
            raise TypeError('agent must return an integer action')
        protocol.write(json.dumps({'action': int(action)}) + '\n')


if __name__ == '__main__':
    main()
