"""Exercise real desktop lifecycle methods without requiring a Tk display."""
import ast
from collections import deque
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
from conversation_turn import ConversationTurn
from stt_bridge import SpeechProcess
from voice_recovery import VoiceRecovery


def desktop_class(available):
    tree = ast.parse((DESKTOP / 'app.py').read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'Desktop')
    namespace = {'json': json, 'threading': threading,
                 'find_asr_root': lambda *_: '/existing/asr',
                 'configured_capture_available': lambda _: available[0],
                 'args': SimpleNamespace(root=Path('/player'), end_silence=.7, mute=False)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), namespace)
    return namespace['Desktop']


class FakeSpeech:
    def __init__(self):
        self.alive = False
        self.paused = True
        self.events = deque()
        self.starts = []
        self.close_error = None

    def running(self):
        return self.alive

    def drain(self):
        events = list(self.events)
        self.events.clear()
        return events

    def start(self, root, device, **options):
        if self.alive:
            raise AssertionError('duplicate ASR owner')
        self.events.clear()
        self.alive = True
        self.starts.append((root, device, options))

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def close(self):
        self.alive = False
        if self.close_error:
            self.events.extend([self.close_error, {'type': 'exited'}])


class FakeLink:
    def __init__(self):
        self.thread = SimpleNamespace(is_alive=lambda: True)
        self.starts = 0
        self.closed = False

    def start(self, *_):
        self.starts += 1

    def drain(self):
        return []

    def close(self):
        self.closed = True


class FakeOutput:
    def __init__(self):
        self.available = True
        self.requests = 0

    def ready(self):
        return self.available

    def request_recovery(self):
        self.requests += 1


class DesktopRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.]
        self.available = [False]
        cls = desktop_class(self.available)
        self.d = cls.__new__(cls)
        d = self.d
        d.closed = d.restarting = False
        d.voice_wanted = True
        d.waiting_audio_start = False
        d.audio_gate_held = False
        d.audio_output = FakeOutput()
        d.audio = SimpleNamespace(process=None, active=False)
        d.audio.is_playing = lambda: d.audio.active
        d.stop_calls = 0
        def stop_media():
            d.stop_calls += 1
            d.audio.active = False
            d.turn.media_cancelled()
        d.stop = stop_media
        d.voice_recovery = VoiceRecovery(clock=lambda: self.now[0])
        d.recovery_ticket = None
        d.standby = SimpleNamespace(abort=lambda: None, screen=None, active=False)
        d.speech = FakeSpeech()
        d.link = FakeLink()
        d.turn = ConversationTurn(lambda _: self.fail('recovery must not send chat'),
                                  d.speech.pause, d.speech.resume)
        d.turn.enable()
        d.turn.set_ready(True)
        d.music = SimpleNamespace(active=False, disable=lambda: d.turn.disable())
        d.settings = {'asr_root': '/existing/asr', 'device': 'plughw:CARD=USB,DEV=0',
                      'host': 'server', 'port': 9000, 'token_file': '/private/token'}
        d.speech_state = 'listening'
        d.speech_error = d.notice = ''
        d.stats = {}
        d.speech_pane = None
        d.next_settings = None
        d.io_done = threading.Event()
        self.error = {'type': 'error', 'reason': 'audio_device_unavailable',
                      'recoverable': True, 'message': 'USB录音设备已断开'}

    def fail_device(self):
        self.d.speech.alive = False
        self.d.speech.events.extend([self.error, {'type': 'exited'}])
        with patch('builtins.print'):
            self.d.poll_voice()

    def attempt(self):
        self.available[0] = True
        self.now[0] += 1
        self.d.poll_voice_recovery()
        self.now[0] += 1
        with patch('builtins.print'):
            self.d.poll_voice_recovery()

    def test_usb_returns_after_long_absence_restarts_one_worker_keeps_link(self):
        self.fail_device()
        for instant in (1, 3, 7, 15, 31, 120, 600):
            self.now[0] = instant
            self.d.poll_voice_recovery()
        self.assertEqual([], self.d.speech.starts)
        self.assertEqual(0, self.d.voice_recovery.status()['attempts'])
        self.now[0] = 620
        self.attempt()
        self.assertEqual(1, len(self.d.speech.starts))
        self.assertEqual(0, self.d.link.starts)
        self.assertFalse(self.d.link.closed)
        self.assertTrue(self.d.turn.enabled)
        self.assertEqual({'paused': True, 'end_silence': .7}, self.d.speech.starts[0][2])
        self.assertEqual('recovering', self.d.voice_recovery.status()['state'])
        self.d.speech.events.append({'type': 'listening'})
        self.d.poll_voice()
        self.assertEqual('idle', self.d.voice_recovery.status()['state'])
        self.assertEqual('', self.d.speech_error)

    def test_old_worker_and_music_must_exit_before_any_new_owner(self):
        self.fail_device()
        self.d.speech.alive = True
        self.available[0] = True
        self.now[0] = 50
        self.d.poll_voice_recovery()
        self.d.speech.alive = False
        self.d.music.active = True
        self.d.poll_voice_recovery()
        self.assertEqual([], self.d.speech.starts)
        self.d.music.active = False
        self.attempt()
        self.assertEqual(1, len(self.d.speech.starts))

    def test_manual_pause_with_late_device_error_never_reopens_microphone(self):
        self.fail_device()
        self.d.stop_voice()
        self.assertTrue(self.d.io_done.wait(1))
        # Cleanup can finish between two GUI drains. Its delayed hardware error
        # must not undo the user's explicit pause on the next GUI frame.
        self.d.poll_voice()
        self.assertFalse(self.d.restarting)
        self.d.speech.events.extend([self.error, {'type': 'exited'}])
        with patch('builtins.print'):
            self.d.poll_voice()
        self.assertFalse(self.d.voice_wanted)
        self.available[0] = True
        for instant in (10, 20, 100):
            self.now[0] = instant
            self.d.poll_voice_recovery()
        self.assertEqual([], self.d.speech.starts)
        self.assertEqual('idle', self.d.voice_recovery.status()['state'])
        self.assertEqual('paused', self.d.speech_state)

    def test_model_and_busy_errors_do_not_restart(self):
        for reason in ('asr_error', 'audio_device_busy', 'audio_overrun'):
            self.setUp()
            self.d.speech.events.append({**self.error, 'reason': reason, 'recoverable': False})
            with patch('builtins.print'):
                self.d.poll_voice()
            self.attempt()
            self.assertEqual([], self.d.speech.starts)
            self.assertEqual('error', self.d.speech_state)

    def test_loading_is_not_success_and_second_device_failure_keeps_budget(self):
        self.fail_device()
        self.attempt()
        self.d.speech.events.append({'type': 'loading'})
        self.d.poll_voice()
        self.assertEqual(1, self.d.voice_recovery.status()['attempts'])
        self.fail_device()
        self.now[0] += 2
        self.d.poll_voice_recovery()
        self.now[0] += 1
        with patch('builtins.print'):
            self.d.poll_voice_recovery()
        self.assertEqual(2, len(self.d.speech.starts))
        self.assertEqual(2, self.d.voice_recovery.status()['attempts'])

    def test_closed_desktop_cannot_restart_a_pending_attempt(self):
        self.fail_device()
        self.d.closed = True
        self.attempt()
        self.assertEqual([], self.d.speech.starts)

    def test_asr_recovery_waits_for_usb_output_before_spending_attempt(self):
        self.fail_device()
        self.available[0] = True
        self.d.audio_output.available = False
        for instant in (10, 20, 30):
            self.now[0] = instant
            self.d.poll_voice_recovery()
        self.assertEqual([], self.d.speech.starts)
        self.assertEqual(0, self.d.voice_recovery.status()['attempts'])
        self.d.audio_output.available = True
        self.attempt()
        self.assertEqual(1, len(self.d.speech.starts))
        self.assertGreater(self.d.audio_output.requests, 0)

    def test_initial_start_waits_for_audio_service_and_keeps_model_link(self):
        self.d.audio_output.available = False
        self.d.start_voice()
        self.assertTrue(self.d.waiting_audio_start)
        self.assertEqual([], self.d.speech.starts)
        self.d.poll_audio_output()
        self.assertFalse(self.d.turn.enabled)
        self.d.audio_output.available = True
        self.d.poll_audio_output()
        self.d.poll_voice_recovery()
        self.assertEqual(1, len(self.d.speech.starts))
        self.assertFalse(self.d.waiting_audio_start)
        self.assertEqual(0, self.d.link.starts)
        self.assertFalse(self.d.link.closed)

    def test_output_service_loss_pauses_capture_then_resumes_same_worker(self):
        self.d.speech.alive = True
        self.d.speech.paused = False
        self.d.audio_output.available = False
        self.d.poll_audio_output()
        self.assertFalse(self.d.turn.enabled)
        self.assertTrue(self.d.speech.paused)
        self.d.audio_output.available = True
        self.d.poll_audio_output()
        self.assertTrue(self.d.turn.enabled)
        self.assertFalse(self.d.speech.paused)
        self.assertEqual([], self.d.speech.starts)
        self.assertFalse(self.d.link.closed)

    def test_manual_pause_cancels_initial_audio_wait(self):
        self.d.audio_output.available = False
        self.d.start_voice()
        self.d.stop_voice()
        self.assertTrue(self.d.io_done.wait(1))
        self.d.poll_voice()
        self.d.audio_output.available = True
        self.d.poll_audio_output()
        self.d.poll_voice_recovery()
        self.assertFalse(self.d.waiting_audio_start)
        self.assertEqual([], self.d.speech.starts)

    def test_output_recovery_does_not_clear_nonrecoverable_asr_fault(self):
        self.d.speech.alive = True
        self.d.speech.events.append({**self.error, 'reason': 'asr_error', 'recoverable': False})
        with patch('builtins.print'):
            self.d.poll_voice()
        self.d.audio_output.available = False
        self.d.poll_audio_output()
        self.d.audio_output.available = True
        self.d.poll_audio_output()
        self.assertFalse(self.d.turn.enabled)
        self.assertEqual('error', self.d.speech_state)

    def test_only_app_owned_live_playback_pids_are_reported(self):
        self.d.music.player = SimpleNamespace(process_ids=lambda: [17, 19])
        self.d.audio.process = SimpleNamespace(pid=23, poll=lambda: None)
        self.assertEqual({17, 19, 23}, self.d.audio_process_ids())
        self.d.audio.process = SimpleNamespace(pid=23, poll=lambda: 0)
        self.assertEqual({17, 19}, self.d.audio_process_ids())

    def test_output_flap_during_asr_reload_cannot_leave_new_worker_paused(self):
        self.fail_device()
        self.attempt()
        self.assertEqual('recovering', self.d.voice_recovery.status()['state'])
        self.assertIsNotNone(self.d.recovery_ticket)
        self.d.audio_output.available = False
        self.d.poll_audio_output()
        self.assertFalse(self.d.turn.enabled)
        self.d.audio_output.available = True
        self.d.poll_audio_output()
        self.assertTrue(self.d.turn.enabled)
        self.assertFalse(self.d.speech.paused)
        self.assertEqual('recovering', self.d.voice_recovery.status()['state'])
        self.d.speech.events.append({'type': 'listening'})
        self.d.poll_voice()
        self.assertEqual('idle', self.d.voice_recovery.status()['state'])

    def test_manual_pause_clears_queued_configuration_restart(self):
        self.d.restarting = True
        self.d.next_settings = dict(self.d.settings)
        self.d.io_done.set()
        self.d.stop_voice()
        self.d.poll_voice()
        self.assertIsNone(self.d.next_settings)
        self.assertFalse(self.d.voice_wanted)
        self.assertEqual([], self.d.speech.starts)

    def test_output_loss_releases_stalled_sound_before_service_repair(self):
        self.d.speech.alive = True
        self.d.turn.media_started()
        self.d.audio.active = True
        self.d.audio_output.available = False
        self.d.poll_audio_output()
        self.assertFalse(self.d.audio.active)
        self.assertFalse(self.d.turn.media_active)
        self.assertEqual(1, self.d.stop_calls)
        self.d.poll_audio_output()
        self.assertEqual(1, self.d.stop_calls)
        self.d.audio.active = True  # A pending server reply reached the player.
        self.d.poll_audio_output()
        self.assertFalse(self.d.audio.active)
        self.assertEqual(2, self.d.stop_calls)


class BridgeRecoveryTests(unittest.TestCase):
    def test_restart_drops_previous_exit_but_preserves_new_structured_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'worker.py'
            error = {'type': 'error', 'reason': 'audio_device_unavailable',
                     'recoverable': True, 'message': 'No such device'}
            worker.write_text('import json\nprint(json.dumps(' + repr(error) + '),flush=True)\n')
            process = SpeechProcess(worker=worker)
            process.publish({'type': 'exited', 'old_worker': True})
            process.start(temporary, 'test', python=sys.executable)
            process.thread.join(3)
            events = process.drain()
            self.assertFalse(any(event.get('old_worker') for event in events))
            self.assertIn(error, events)
            self.assertEqual(1, sum(event['type'] == 'exited' for event in events))


if __name__ == '__main__':
    unittest.main()
