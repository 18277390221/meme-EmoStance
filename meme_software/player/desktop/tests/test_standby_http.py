"""Standby controls retain the existing loopback Host/Origin boundary."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
import test_music_http as music_http


class StandbyHTTPTests(music_http.MusicHTTPTests):
    def setUp(self):
        super().setUp()
        del self.player.music_command
        self.player.standby_command = lambda body: self.calls.append(body) or {'state': 'awake'}

    def post(self, body, headers=None):
        from http.client import HTTPConnection
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        self.addCleanup(connection.close)
        connection.request('POST', '/api/standby', json.dumps(body),
                           {'Content-Type': 'application/json', **(headers or {})})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_local_music_callback_receives_validated_json(self):
        status, state = self.post({'action': 'sleep'})
        self.assertEqual(200, status)
        self.assertEqual('awake', state['state'])
        self.assertEqual([{'action': 'sleep'}], self.calls)

    def test_missing_feature_and_invalid_command_report_clear_errors(self):
        del self.player.standby_command
        self.assertEqual(400, self.post({'action': 'sleep'})[0])
        self.player.standby_command = lambda body: (_ for _ in ()).throw(ValueError('只接受action'))
        status, state = self.post({'action': 'sleep', 'path': '/tmp/test'})
        self.assertEqual(400, status)
        self.assertEqual('只接受action', state['error'])


if __name__ == '__main__':
    unittest.main()
