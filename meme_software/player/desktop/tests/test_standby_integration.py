"""Exercise real Desktop standby and audio recovery hooks without a Tk display."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(Path(__file__).parent))
from conversation_turn import ConversationTurn
from voice_recovery import VoiceRecovery
from test_voice_recovery_integration import desktop_class, FakeLink, FakeOutput, FakeSpeech


class StandbyDesktopTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        cls = desktop_class([True])
        self.d = d = cls.__new__(cls)
        d.closed = d.restarting = False
        d.voice_wanted = True
        d.standby_connection_lost = False
        d.waiting_audio_start = d.audio_gate_held = False
        d.speech = FakeSpeech()
        d.speech.alive = True
        d.link = FakeLink()
        d.link.connected = True
        d.audio_output = FakeOutput()
        d.audio = SimpleNamespace(process=None, is_playing=lambda: False)
        d.turn = ConversationTurn(lambda _: self.fail('standby recovery cannot send model requests'),
                                  self.pause, self.resume)
        d.turn.enable()
        d.turn.set_ready(True)
        self.assertTrue(d.turn.standby_prepare())
        self.assertTrue(d.turn.standby_started())
        d.standby = SimpleNamespace(active=True, screen=None, abort=self.abort, poll=self.poll_standby)
        d.music = SimpleNamespace(active=False, set_ready=self.music_ready, disable=self.disable_music,
                                  player=SimpleNamespace(process_ids=lambda: []))
        d.voice_recovery = VoiceRecovery(clock=lambda: 0.)
        d.recovery_ticket = None
        d.speech_state, d.speech_error, d.notice = 'listening', '', ''
        d.speech_pane = None
        d.stats = {}
        d.next_settings = None
        d.io_done = threading.Event()
        d.stop = lambda: self.calls.append('stop_media')
        self.calls.clear()

    def pause(self):
        self.calls.append('pause')
        self.d.speech.pause()

    def resume(self):
        self.calls.append('resume')
        self.d.speech.resume()

    def abort(self):
        self.calls.append('abort_standby')
        self.d.standby.active = False
        self.d.turn.standby_abort()

    def poll_standby(self):
        self.calls.append('poll_standby')

    def music_ready(self, ready):
        self.calls.append(('ready', ready))
        self.d.turn.set_ready(ready)

    def disable_music(self):
        self.calls.append('disable_music')
        self.d.turn.disable()

    def flap_server(self):
        self.d.server_ready(False)
        self.d.server_ready(True)

    def test_fast_server_disconnect_reconnect_aborts_before_normal_listening_resumes(self):
        self.flap_server()
        self.assertTrue(self.d.standby_connection_lost)
        self.assertTrue(self.d.standby.active)
        self.assertTrue(self.d.speech.paused)
        self.calls.clear()
        self.d.poll_standby()
        self.assertEqual(['abort_standby', 'pause', 'resume', 'poll_standby'], self.calls)
        self.assertFalse(self.d.standby_connection_lost)
        self.assertFalse(self.d.standby.active)
        self.assertEqual('listening', self.d.turn.phase)
        self.assertFalse(self.d.speech.paused)
        self.assertEqual(0, self.d.link.starts)
        self.assertEqual([], self.d.speech.starts)
        self.calls.clear()
        self.d.poll_standby()
        self.assertEqual(['poll_standby'], self.calls)

    def test_disconnect_without_reconnect_leaves_capture_paused(self):
        self.d.link.connected = False
        self.d.server_ready(False)
        self.calls.clear()
        self.d.poll_standby()
        self.assertEqual(['abort_standby', 'pause', 'poll_standby'], self.calls)
        self.assertEqual('connecting', self.d.turn.phase)
        self.assertTrue(self.d.speech.paused)

    def test_reconnected_standby_does_not_override_recovery_or_manual_pause_guards(self):
        for guard in ('voice_paused', 'restarting', 'asr_stopped', 'output_missing', 'socket_disconnected',
                      'asr_recovery'):
            with self.subTest(guard=guard):
                self.setUp()
                self.flap_server()
                if guard == 'voice_paused':
                    self.d.voice_wanted = False
                elif guard == 'restarting':
                    self.d.restarting = True
                elif guard == 'asr_stopped':
                    self.d.speech.alive = False
                elif guard == 'output_missing':
                    self.d.audio_output.available = False
                elif guard == 'socket_disconnected':
                    self.d.link.connected = False
                else:
                    self.d.voice_recovery.fail({'reason': 'audio_device_unavailable', 'recoverable': True})
                self.calls.clear()
                self.d.poll_standby()
                self.assertIn('abort_standby', self.calls)
                self.assertNotIn('resume', self.calls)
                self.assertTrue(self.d.speech.paused)

    def test_ready_notification_without_disconnect_does_not_abort_standby(self):
        self.d.server_ready(True)
        self.calls.clear()
        self.d.poll_standby()
        self.assertEqual(['poll_standby'], self.calls)
        self.assertTrue(self.d.standby.active)
        self.assertEqual('standby', self.d.turn.phase)

    def test_microphone_loss_aborts_standby_before_music_disable_and_asr_recovery(self):
        self.d.speech.alive = False
        self.d.speech.events.extend([
            {'type': 'error', 'reason': 'audio_device_unavailable', 'recoverable': True,
             'message': 'USB录音设备已断开'}, {'type': 'exited'}])
        with patch('builtins.print'):
            self.d.poll_voice()
        self.assertLess(self.calls.index('abort_standby'), self.calls.index('disable_music'))
        self.assertFalse(self.d.standby.active)
        self.assertEqual('recovering', self.d.speech_state)
        self.assertNotEqual('idle', self.d.voice_recovery.status()['state'])
        self.assertGreater(self.d.audio_output.requests, 0)
        self.assertEqual([], self.d.speech.starts)
        self.assertFalse(self.d.link.closed)

    def test_output_device_loss_aborts_standby_before_disabling_chat_and_stopping_media(self):
        self.d.audio_output.available = False
        self.d.poll_audio_output()
        self.assertLess(self.calls.index('abort_standby'), self.calls.index('disable_music'))
        self.assertLess(self.calls.index('abort_standby'), self.calls.index('stop_media'))
        self.assertFalse(self.d.standby.active)
        self.assertFalse(self.d.turn.enabled)
        self.assertTrue(self.d.speech.paused)
        self.assertTrue(self.d.audio_gate_held)
        self.assertEqual('recovering', self.d.speech_state)
        self.d.audio_output.available = True
        self.calls.clear()
        self.d.poll_audio_output()
        self.assertEqual(['resume'], self.calls)
        self.assertEqual('listening', self.d.turn.phase)
        self.assertFalse(self.d.speech.paused)
        self.assertEqual([], self.d.speech.starts)

    def test_unexpected_asr_exit_also_aborts_standby(self):
        self.d.speech.alive = False
        self.d.speech.events.append({'type': 'exited'})
        self.d.poll_voice()
        self.assertLess(self.calls.index('abort_standby'), self.calls.index('disable_music'))
        self.assertFalse(self.d.standby.active)
        self.assertEqual('paused', self.d.speech_state)

    def test_audio_service_repair_is_blocked_for_active_standby_even_with_capture_paused(self):
        self.d.speech.paused = True
        self.assertFalse(self.d.audio_repair_allowed())
        self.d.speech.alive = False
        self.assertFalse(self.d.audio_repair_allowed())
        self.d.standby.active = False
        self.assertTrue(self.d.audio_repair_allowed())

    def test_audio_service_repair_preserves_other_capture_and_media_guards(self):
        self.d.standby.active = False
        self.d.speech.paused = True
        self.assertTrue(self.d.audio_repair_allowed())
        self.d.speech.paused = False
        self.assertFalse(self.d.audio_repair_allowed())
        self.d.speech.paused = True
        self.d.music.active = True
        self.assertFalse(self.d.audio_repair_allowed())
        self.d.music.active = False
        self.d.audio.is_playing = lambda: True
        self.assertFalse(self.d.audio_repair_allowed())

    def test_audio_process_ids_include_standby_wav_and_other_owned_output(self):
        self.d.music.player.process_ids = lambda: [17, 19]
        self.d.audio.process = SimpleNamespace(pid=23, poll=lambda: None)
        self.d.standby.screen = SimpleNamespace(audio=SimpleNamespace(
            process=SimpleNamespace(pid=29, poll=lambda: None)))
        self.assertEqual({17, 19, 23, 29}, self.d.audio_process_ids())

    def test_audio_process_ids_filter_exited_standby_and_meme_output(self):
        self.d.music.player.process_ids = lambda: [17]
        self.d.audio.process = SimpleNamespace(pid=23, poll=lambda: 0)
        self.d.standby.screen = SimpleNamespace(audio=SimpleNamespace(
            process=SimpleNamespace(pid=29, poll=lambda: 1)))
        self.assertEqual({17}, self.d.audio_process_ids())

    def test_audio_process_ids_handle_absent_sleep_screen_and_duplicate_owners(self):
        self.assertEqual(set(), self.d.audio_process_ids())
        self.d.music.player.process_ids = lambda: [17]
        self.d.audio.process = SimpleNamespace(pid=17, poll=lambda: None)
        self.d.standby.screen = SimpleNamespace(audio=SimpleNamespace(
            process=SimpleNamespace(pid=17, poll=lambda: None)))
        self.assertEqual({17}, self.d.audio_process_ids())
        del self.d.standby
        self.assertEqual({17}, self.d.audio_process_ids())


if __name__ == '__main__':
    unittest.main()
