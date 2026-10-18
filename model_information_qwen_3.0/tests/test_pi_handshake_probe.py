"""Verify the diagnostic helper without touching a Pi, sound device or model."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from model_information_qwen_v3.tools import verify_pi_handshake as probe


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.options = {'action': 'check', 'player_port': 8784, 'software_dir': '/installed/meme',
                        'server_host': '10.205.84.185', 'server_port': 9000, 'timeout': 10}

    def state(self, phase='listening', paused=False, arecord=True):
        return {'health_available': True, 'phase': phase, 'model_ready': True,
                'microphone_paused': paused, 'pending_requests': 0, 'media_active': phase == 'playing',
                'processes': {'app': [10], 'stt': [11], 'arecord': [12] if arecord else [], 'aplay': []}}

    def test_health_strips_all_nonstatus_content(self):
        payload = {'status': 'ready', 'token': 'secret-never-return', 'reply_text': 'private-user-text',
                   'communication': {'phase': 'connecting', 'model_ready': False, 'microphone_paused': True,
                                     'model_version': None, 'pending_requests': 0,
                                     'token': 'secret-never-return', 'text': 'private-user-text'}}
        result = probe.sanitized_health(payload)
        self.assertEqual({'health_available': True, 'phase': 'connecting', 'model_ready': False,
                          'microphone_paused': True, 'pending_requests': 0}, result)
        self.assertNotIn('secret-never-return', json.dumps(result))
        self.assertNotIn('private-user-text', json.dumps(result))

    def test_health_ready_does_not_imply_model_ready(self):
        result = probe.sanitized_health({'status': 'ready', 'communication': {'model_ready': False}})
        self.assertTrue(result['health_available'])
        self.assertFalse(result['model_ready'])
        self.assertNotIn('phase', result)

    def test_health_background_metadata_has_safe_types_and_no_paths(self):
        good = {'background_id': 10, 'conversation_id': 'conversation-1',
                'selected_visual': 'meme-0001-猫猫-10.gif'}
        result = probe.sanitized_health({'status': 'ready', 'communication': good})
        self.assertEqual({'health_available': True, **good}, result)
        for fields in ({'background_id': True}, {'background_id': 11},
                       {'conversation_id': '../private'},
                       {'selected_visual': '/private/meme-0001-cat-1.gif'},
                       {'selected_visual': 'meme-0001-cat.gif'}):
            self.assertEqual({'health_available': True}, probe.sanitized_health(
                {'status': 'ready', 'communication': fields}))

    def test_bad_status_types_and_phase_do_not_leak(self):
        result = probe.sanitized_health({'status': 'ready', 'communication':
                                        {'phase': 'private text', 'model_version': 'secret with spaces',
                                         'protocol_version': True, 'model_ready': 'true'}})
        self.assertEqual({'health_available': True}, result)

    def test_check_only_reads_health_and_relevant_processes(self):
        with patch.object(probe, 'http_json', return_value={'status': 'ready'}) as http, \
             patch.object(probe, 'process_ids', return_value={'app': [1], 'stt': [], 'arecord': [], 'aplay': []}):
            report = probe.remote_run(self.options)
        http.assert_called_once_with(8784, '/api/health')
        self.assertEqual('check', report['action'])
        self.assertEqual([1], report['processes']['app'])

    def test_process_selection_excludes_unrelated_audio_and_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            software = root / 'software'
            desktop = software / 'player/desktop'
            desktop.mkdir(parents=True)
            proc = root / 'proc'
            proc.mkdir()
            for pid, parent, comm, argv in (
                    (10, 1, 'python3', ['python3', 'app.py']),
                    (11, 10, 'python', ['python', str(desktop / 'stt_worker.py')]),
                    (12, 11, 'arecord', ['arecord', '-q']),
                    (13, 10, 'aplay', ['aplay', '-q']),
                    (99, 1, 'arecord', ['arecord', 'private-args']),
                    (98, 1, 'python3', ['python3', 'other.py', 'private-args'])):
                process = proc / str(pid)
                process.mkdir()
                (process / 'status').write_text(f'Uid: {probe.os.getuid()}\nPPid: {parent}\n')
                (process / 'comm').write_text(comm)
                (process / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
                (process / 'cwd').symlink_to(desktop, target_is_directory=True)
            result = probe.process_ids(software, proc)
        self.assertEqual({'app': [10], 'stt': [11], 'arecord': [12], 'aplay': [13]}, result)
        self.assertNotIn('private-args', json.dumps(result))

    def test_choose_preferred_asset_only_if_present(self):
        self.assertEqual('meme-0062', probe.select_asset([{'id': 'meme-0001'}, {'id': 'meme-0062'}]))
        self.assertEqual('meme-0001', probe.select_asset([{'id': '../invalid'}, {'id': 'meme-0001'}]))
        with self.assertRaises(ValueError):
            probe.select_asset([])

    def test_playback_refuses_while_model_not_ready(self):
        state = self.state()
        state['model_ready'] = False
        with patch.object(probe, 'snapshot', return_value=state), patch.object(probe, 'http_json') as http:
            report = probe.playback_exercise(self.options)
        http.assert_not_called()
        self.assertEqual('player_not_ready_for_controlled_playback', report['error'])

    def test_playback_observes_pause_and_restoration_without_model_request(self):
        samples = [self.state(), self.state('preparing', True, False),
                   self.state('playing', True, False), self.state()]
        with patch.object(probe, 'snapshot', side_effect=samples), patch.object(probe.time, 'sleep'), \
             patch.object(probe, 'http_json', side_effect=[[{'id': 'meme-0062'}], {'status': 'ok'}]) as http:
            report = probe.playback_exercise(self.options)
        self.assertTrue(report['playback_verified'])
        self.assertTrue(report['listening_restored'])
        self.assertEqual('/api/assets', http.call_args_list[0].args[1])
        body = http.call_args_list[1].args[2]
        self.assertEqual('/api/response', http.call_args_list[1].args[1])
        self.assertEqual('握手联调完成', body['reply_text'])
        self.assertTrue(body['request_id'].startswith('desktop-verify-'))
        self.assertNotIn('握手联调完成', json.dumps(report, ensure_ascii=False))

    def test_local_auth_is_parsed_as_data_and_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / 'zerotier_chat').mkdir()
            token = 'mock-only-local-pi-token'
            (home / 'zerotier_chat/.env').write_text('IGNORED=$(touch /not-executed)\nZT_CHAT_TOKEN=' + token)
            with patch.object(probe.Path, 'home', return_value=home), patch.dict(probe.os.environ, {}, clear=True):
                self.assertEqual(token, probe.local_token(home))

    def test_request_exchange_uses_end_and_separate_accepted_and_playback_ack(self):
        key, session, token = 'request-123', 'session-123', 'mock-only-local-pi-token'
        frames = [{'type': 'hello_ok', 'protocol_version': 2, 'model_ready': True,
                   'model_version': 'V2.0.0', 'session_id': session},
                  {'type': 'control_ack', 'control': 'speech_start'},
                  {'type': 'accepted', 'id': key, 'status': 'processing'},
                  {'type': 'result', 'id': key, 'meme_id': 'meme-0062', 'reply_text': 'private-model-reply'},
                  {'type': 'control_ack', 'control': 'playback_done', 'id': key}]
        connection = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        connection.makefile.return_value = io.BytesIO(b''.join((json.dumps(frame) + '\n').encode() for frame in frames))
        with patch.object(probe.socket, 'create_connection', return_value=connection), \
             patch.object(probe, 'local_token', return_value=token), \
             patch.object(probe.uuid, 'uuid4', side_effect=[Mock(hex=key), Mock(hex=session)]):
            report = probe.request_exercise(self.options)
        messages = [json.loads(call.args[0]) for call in connection.sendall.call_args_list]
        self.assertEqual(['hello', 'speech_start', 'request', 'playback_done'], [frame['type'] for frame in messages])
        self.assertIs(messages[2]['end'], True)
        self.assertEqual(session, messages[0]['session_id'])
        self.assertTrue(report['accepted_before_result'])
        self.assertTrue(report['reply_nonempty'])
        self.assertTrue(report['playback_acknowledged'])
        self.assertNotIn(token, json.dumps(report))
        self.assertNotIn('private-model-reply', json.dumps(report))

    def test_errors_never_reflect_sensitive_exception_content(self):
        with patch.object(probe, 'snapshot', side_effect=ValueError('private exception token')):
            report = probe.remote_run(self.options)
        self.assertEqual('probe_failed', report['error'])
        self.assertNotIn('private exception', json.dumps(report))

    def test_cli_uses_qwen_settings_and_selected_config_without_loading_model(self):
        settings = Mock(software_dir='/installed/meme', host='10.0.0.1', port=9000)
        config = Path('/local/selected_qwen.json')
        with patch('model_information_qwen_v3.launcher.load_settings', return_value=settings) as load, \
                patch.object(probe, 'ssh_probe', return_value={'action': 'check'}) as remote, \
                patch('sys.stdout', io.StringIO()):
            code = probe.main(['--check', '--config', str(config)])
        self.assertEqual(code, 0)
        self.assertEqual(load.call_args.args[0].config, config)
        self.assertIsNone(load.call_args.args[0].model_dir)
        self.assertEqual(remote.call_args.args[1]['action'], 'check')


if __name__ == '__main__':
    unittest.main()
