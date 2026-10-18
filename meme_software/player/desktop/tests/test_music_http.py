"""Music controls retain the player's loopback Host and Origin protections."""
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from launch import Handler


class QuietHandler(Handler):
    def log_message(self, *_args):
        pass


class MusicHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.calls = []
        self.player = SimpleNamespace(website=Path(self.temporary.name),
            music_command=lambda body: self.calls.append(body) or {'state': 'preparing'})
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, player=self.player))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def post(self, body, headers=None):
        connection = HTTPConnection('127.0.0.1', self.server.server_port, timeout=2)
        self.addCleanup(connection.close)
        supplied = {'Content-Type': 'application/json'}
        supplied.update(headers or {})
        connection.request('POST', '/api/music', json.dumps(body), supplied)
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_local_music_callback_receives_validated_json(self):
        status, body = self.post({'action': 'play', 'text': '想听音乐'})
        self.assertEqual(200, status)
        self.assertEqual('preparing', body['state'])
        self.assertEqual([{'action': 'play', 'text': '想听音乐'}], self.calls)

    def test_foreign_origin_and_wrong_host_cannot_control_music(self):
        for headers in ({'Origin': 'https://example.invalid'}, {'Host': 'example.invalid'}):
            with self.subTest(headers=headers):
                self.assertEqual(403, self.post({'action': 'play'}, headers)[0])
        self.assertEqual([], self.calls)

    def test_missing_feature_and_invalid_command_report_clear_errors(self):
        del self.player.music_command
        status, body = self.post({'action': 'play'})
        self.assertEqual(400, status)
        self.assertIn('不可用', body['error'])
        self.player.music_command = lambda body: (_ for _ in ()).throw(ValueError('不接受歌曲路径'))
        status, body = self.post({'action': 'play', 'path': '/tmp/file'})
        self.assertEqual(400, status)
        self.assertEqual('不接受歌曲路径', body['error'])


if __name__ == '__main__':
    unittest.main()
