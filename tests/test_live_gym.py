"""Exercise the original Gymnasium transition/rendering and local API boundaries."""
import base64
import http.client
import json
import os
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from tournament.serve import ROOT, Session, TournamentServer


class GymSessionTests(unittest.TestCase):
    def session(self, lab=2, policies=('heuristic', 'heuristic')):
        session = Session(lab, 42, list(policies), {})
        self.addCleanup(session.close)
        return session

    def test_same_seed_uses_original_taxi_frames_and_finishes_identically(self):
        session = self.session()
        first = session.snapshot()['states']
        self.assertEqual(first[0]['frame'], first[1]['frame'])
        image = cv2.imdecode(np.frombuffer(base64.b64decode(first[0]['frame'].split(',')[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (350, 550, 3))
        for _ in range(200):
            states = session.step([None, None])['states']
            if all(state['done'] for state in states):
                break
        self.assertEqual(states[0], states[1])
        self.assertEqual(states[0]['message'], 'DESTINATION REACHED')
        self.assertIn(states[0]['env_id'], ('Taxi-v3', 'Taxi-v4'))
        self.assertLess(states[0]['steps'], 30)
        self.assertEqual(session.step([None, None])['states'], states)

    def test_manual_taxi_waits_and_invalid_action_does_not_advance_either_player(self):
        session = self.session(policies=('manual', 'random'))
        session.step([None, None])
        self.assertEqual(session.states[0]['steps'], 0)
        self.assertEqual(session.states[1]['steps'], 1)
        with self.assertRaises(ValueError):
            session.step([0, 6])
        self.assertEqual(session.states[0]['steps'], 0)
        session.step([0, None])
        self.assertEqual(session.states[0]['steps'], 1)
        self.assertEqual(session.states[0]['score'], -1)

    def test_lander_runs_box2d_and_original_renderer(self):
        session = self.session(1, ('manual', 'heuristic'))
        before = session.snapshot()['states'][0]['frame']
        states = session.step([2, None], repeat=4)['states']
        self.assertEqual(states[0]['env_id'], 'LunarLander-v3')
        self.assertEqual(states[0]['steps'], 4)
        self.assertNotEqual(states[0]['frame'], before)
        self.assertEqual(states[0]['action'], 2)

    def test_agent_failure_closes_all_environments(self):
        with patch('tournament.serve.IsolatedAgent', side_effect=RuntimeError('sandbox unavailable')):
            with patch('gymnasium.wrappers.TimeLimit.close', autospec=True) as close:
                with self.assertRaises(RuntimeError):
                    Session(2, 42, ['agent:test'], {'agent:test': ROOT / 'submissions/lab2/rafael_flor'})
                close.assert_called_once()

    @unittest.skipUnless(os.environ.get('RUN_SANDBOX_TESTS') == '1', 'Requires Docker agent image')
    def test_saved_taxi_agent_runs_live_in_isolation(self):
        session = Session(2, 42, ['agent:rafael_flor', 'random'], {
            'agent:rafael_flor': ROOT / 'submissions/lab2/rafael_flor'})
        self.addCleanup(session.close)
        for _ in range(200):
            states = session.step([None, None])['states']
            if states[0]['done']:
                break
        self.assertEqual(states[0]['message'], 'DESTINATION REACHED')
        self.assertGreater(states[0]['score'], 0)


class LocalAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = TournamentServer(('127.0.0.1', 0))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def request(self, path, data=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request('POST' if data is not None else 'GET', path,
                               json.dumps(data) if data is not None else None,
                               {'Content-Type': 'application/json', **(headers or {})})
            response = connection.getresponse()
            body = response.read()
            return response.status, json.loads(body) if response.getheader('Content-Type') == 'application/json' else body.decode()
        finally:
            connection.close()

    def test_host_origin_and_content_type_rejected(self):
        for headers in ({'Host': 'evil.test'}, {'Origin': 'https://evil.test'},
                        {'Sec-Fetch-Site': 'cross-site'}, {'Content-Type': 'text/plain'}):
            self.assertEqual(self.request('/api/session', {}, headers)[0], 403)

    def test_unknown_agents_paths_and_unbounded_input_rejected(self):
        for data in ({'lab': 2, 'seed': 42, 'policies': ['agent:../../foo']},
                     {'lab': 2, 'seed': -1, 'policies': ['manual']},
                     {'lab': 2, 'seed': 42, 'policies': ['manual', 'manual']},
                     {'lab': 2, 'seed': 42, 'policies': ['random'] * 3}):
            self.assertEqual(self.request('/api/session', data)[0], 400)
        self.assertEqual(self.request('/api/config')[1]['agents'], {'1': [], '2': []})
        self.assertEqual(self.request('/../submissions/')[0], 404)

    def test_create_step_close_and_expiry(self):
        status, result = self.request('/api/session', {'lab': 2, 'seed': 42, 'policies': ['manual']})
        self.assertEqual(status, 200)
        key = result['id']
        status, result = self.request('/api/step', {'id': key, 'actions': [0]})
        self.assertEqual(status, 200)
        self.assertEqual(result['states'][0]['steps'], 1)
        self.assertEqual(self.request('/api/close', {'id': key})[0], 200)
        self.assertEqual(self.request('/api/step', {'id': key, 'actions': [0]})[0], 404)
        _, result = self.request('/api/session', {'lab': 2, 'seed': 42, 'policies': ['manual']})
        self.server.sessions[result['id']].updated -= 301
        self.assertEqual(self.request('/api/step', {'id': result['id'], 'actions': [0]})[0], 404)
