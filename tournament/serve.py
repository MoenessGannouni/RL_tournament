"""Serve the tournament UI with real, locally controlled Gymnasium environments.

Run: python -m tournament.serve [--isolated-agents]
The server binds loopback only, serves web/ only, and never imports student code.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlsplit

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')

import cv2
import gymnasium as gym
import numpy as np

from tournament.sandbox import IsolatedAgent, preflight_sandbox, validate_submission

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / 'web'
BUILTINS = ('manual', 'heuristic', 'random')


def taxi_action(env, observation):
    """Shortest path to passenger/destination using the environment's transition table."""
    row, col, passenger, destination = tuple(env.unwrapped.decode(int(observation)))
    target = env.unwrapped.locs[destination if passenger == 4 else passenger]
    if (row, col) == tuple(target):
        return 5 if passenger == 4 else 4
    queue, seen = [(int(observation), None)], {int(observation)}
    for state, first in queue:
        r, c, _, _ = tuple(env.unwrapped.decode(state))
        if (r, c) == tuple(target):
            return first
        for action in range(4):
            next_state = env.unwrapped.P[state][action][0][1]
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, action if first is None else first))
    return 0


class Session:
    def __init__(self, lab, seed, policies, available):
        self.lab, self.seed, self.policies = lab, seed, policies
        self.envs, self.agents, self.states, self.observations = [], [], [], []
        self.updated = time.monotonic()
        try:
            for policy in policies:
                candidates = ['Taxi-v4', 'Taxi-v3'] if lab == 2 else ['LunarLander-v3', 'LunarLander-v2']
                env = None
                for candidate in candidates:
                    try:
                        env = gym.make(candidate, render_mode='rgb_array')
                        break
                    except Exception:
                        pass
                if env is None:
                    raise RuntimeError(f'Could not create environment for lab {lab}')
                self.envs.append(env)
                obs, _ = env.reset(seed=seed)
                env.action_space.seed(seed)
                self.observations.append(obs)
                agent = None
                if policy not in BUILTINS:
                    agent = IsolatedAgent(available[policy], int(env.action_space.n), total_timeout=1800)
                self.agents.append(agent)
                self.states.append(dict(score=0, steps=0, done=False, message='READY', action=None))
        except Exception:
            self.close()
            raise

    def close(self):
        for agent in self.agents:
            if agent is not None:
                agent.close()
        for env in self.envs:
            env.close()

    def step(self, actions, repeat=1):
        if type(repeat) is not int or not 1 <= repeat <= (8 if self.lab == 1 else 1):
            raise ValueError("Invalid frame batch")
        if not isinstance(actions, list) or len(actions) != len(self.envs):
            raise ValueError('One action per player is required')
        # Validate the entire request before either environment advances.
        for env, action in zip(self.envs, actions):
            if action is not None and (type(action) is not int or not env.action_space.contains(action)):
                raise ValueError('Invalid discrete action')
        for _ in range(repeat):
            for i, (env, policy, state) in enumerate(zip(self.envs, self.policies, self.states)):
                if state['done']:
                    continue
                if policy == 'manual':
                    action = actions[i]
                    if action is None:
                        continue  # Taxi waits for a discrete keypress.
                elif policy == 'random':
                    action = int(env.action_space.sample())
                elif policy == 'heuristic':
                    if self.lab == 2:
                        action = taxi_action(env, self.observations[i])
                    else:
                        from gymnasium.envs.box2d.lunar_lander import heuristic
                        action = heuristic(env.unwrapped, self.observations[i])
                else:
                    action = self.agents[i](self.observations[i])
                obs, reward, terminated, truncated, _ = env.step(action)
                self.observations[i] = obs
                state.update(score=state['score'] + float(reward), steps=state['steps'] + 1,
                             done=bool(terminated or truncated), action=int(action),
                             message='STEP LIMIT REACHED' if truncated else
                             ('DESTINATION REACHED' if self.lab == 2 else 'EPISODE COMPLETE')
                             if terminated else 'RUNNING')
        self.updated = time.monotonic()
        return self.snapshot()

    def snapshot(self):
        states = []
        for env, state in zip(self.envs, self.states):
            rgb = env.render()
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError('Frame encoding failed')
            states.append({**state, 'frame': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'),
                           'env_id': env.spec.id})
        return {'states': states, 'seed': self.seed}


class TournamentServer(HTTPServer):
    def __init__(self, address, isolated_agents=False):
        super().__init__(address, Handler)
        self.sessions = {}
        self.available = {1: {}, 2: {}}
        if isolated_agents:
            preflight_sandbox()
            for lab in (1, 2):
                for directory in sorted((ROOT / 'submissions' / f'lab{lab}').iterdir()):
                    if directory.is_dir() and (directory / 'agent.py').exists():
                        self.available[lab]['agent:' + directory.name] = validate_submission(directory)

    def expire_sessions(self):
        for key, session in list(self.sessions.items()):
            if time.monotonic() - session.updated > 300:
                self.sessions.pop(key).close()

    def service_actions(self):
        self.expire_sessions()

    def server_close(self):
        for session in getattr(self, 'sessions', {}).values():
            session.close()
        super().server_close()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, fmt, *args):
        pass

    def allowed(self, mutation=False):
        hosts = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host')
        if host not in hosts:
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin != f'http://{host}':
            return False
        if self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin', 'none'):
            return False
        return not mutation or self.headers.get_content_type() == 'application/json'

    def respond(self, status, data):
        encoded = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if not self.allowed():
            return self.respond(403, {'error': 'Same-origin loopback requests only'})
        if urlsplit(self.path).path == '/api/config':
            return self.respond(200, {'gymnasium': gym.__version__, 'agents': {
                str(lab): [{'value': key, 'label': path.name} for key, path in agents.items()]
                for lab, agents in self.server.available.items()}})
        if urlsplit(self.path).path.startswith('/api/'):
            return self.respond(404, {'error': 'Unknown endpoint'})
        super().do_GET()

    def send_head(self):
        # Shared by GET and HEAD; never list directories or follow links out of web/.
        if not self.allowed():
            self.send_error(403)
            return None
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(WEB_ROOT) or path.is_dir() and not (path / 'index.html').is_file():
            self.send_error(404)
            return None
        return super().send_head()

    def do_POST(self):
        if not self.allowed(mutation=True):
            return self.respond(403, {'error': 'Same-origin JSON requests only'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 4096:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError('Expected a JSON object')
            self.server.expire_sessions()
            route = urlsplit(self.path).path
            if route == '/api/session':
                lab, seed, policies = data.get('lab'), data.get('seed'), data.get('policies')
                if type(lab) is not int or lab not in (1, 2) or type(seed) is not int or not 0 <= seed <= 1000000:
                    raise ValueError('Invalid lab or seed')
                available = self.server.available[lab]
                if (not isinstance(policies, list) or not 1 <= len(policies) <= 2 or
                        any(not isinstance(p, str) or p not in (*BUILTINS, *available) for p in policies)):
                    raise ValueError('Unknown policy')
                if policies.count('manual') > 1:
                    raise ValueError('Only one human player per duel')
                if len(self.server.sessions) >= 8:
                    return self.respond(429, {'error': 'Too many sessions. Close an existing session or wait five minutes.'})
                session = Session(lab, seed, policies, available)
                try:
                    snapshot = session.snapshot()
                except Exception:
                    session.close()
                    raise
                key = secrets.token_urlsafe(24)
                self.server.sessions[key] = session
                return self.respond(200, {'id': key, **snapshot})
            if route not in ('/api/step', '/api/close'):
                return self.respond(404, {'error': 'Unknown endpoint'})
            key = data.get('id')
            if not isinstance(key, str) or key not in self.server.sessions:
                return self.respond(404, {'error': 'Session expired. Reset to start again.'})
            if route == '/api/close':
                self.server.sessions.pop(key).close()
                return self.respond(200, {'closed': True})
            try:
                snapshot = self.server.sessions[key].step(data.get('actions'), data.get('repeat', 1))
            except Exception:
                self.server.sessions.pop(key).close()
                raise
            return self.respond(200, snapshot)
        except (ValueError, TypeError, KeyError) as exc:
            self.respond(400, {'error': str(exc)})
        except Exception as exc:
            # No fallback to host imports or a substitute simulation.
            self.respond(503, {'error': f'Gymnasium session failed: {type(exc).__name__}. Check dependencies and Docker, then reset.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--isolated-agents', action='store_true', help='Enable repository agents in Docker (trusted image required)')
    args = parser.parse_args()
    with TournamentServer(('127.0.0.1', args.port), args.isolated_agents) as server:
        print(f'Gymnasium tournament: http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
