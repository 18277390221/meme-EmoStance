"""Standby remains local and owns the full wake-animation handover."""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'meme_software/player/desktop'))
from conversation_turn import ConversationTurn
from meme_huanxing.controller import StandbyController


class Window:
    def __init__(self):
        self.jobs = []
        self.cancelled = []
        self.bindings = []
    def after(self, delay, callback):
        self.jobs.append((delay, callback))
        return len(self.jobs)
    def after_cancel(self, job):
        self.cancelled.append(job)
    def bind(self, *args, **kwargs):
        self.bindings.append((args, kwargs))


class Speech:
    def __init__(self, calls):
        self.calls = calls
        self.paused = False
        self.alive = True
        self.last_activity = 100
    def pause(self):
        self.calls.append(('pause',))
    def resume(self):
        self.calls.append(('resume',))
        self.paused = False
    def running(self):
        return self.alive
    def standby_control(self, enabled, token=None):
        self.calls.append(('control', enabled, token))


class Music:
    def __init__(self, turn, calls):
        self.turn, self.calls = turn, calls
        self.active = False
        self.token = None
    def handle_final(self, text):
        self.calls.append(('normal', text))
        return self.turn.submit_final(text)
    def command(self, body):
        self.calls.append(('music_command', body))


class Screen:
    def __init__(self, parent, **options):
        self.options = options
        self.wakes = 0
        self.closes = 0
        self.thread = threading.get_ident()
    def wake(self):
        self.wakes += 1
    def close(self):
        self.closes += 1
    def complete(self):
        self.options['on_wake_complete']()
    def touch(self):
        self.options['on_wake_request']()
    def error(self, text):
        self.options['on_error'](RuntimeError(text))


class StandbyTests(unittest.TestCase):
    def setUp(self):
        self.calls, self.screens = [], []
        self.now = 100.0
        self.window = Window()
        self.speech = Speech(self.calls)
        self.turn = ConversationTurn(
            lambda text: self.calls.append(('send', text)) or 'request-1',
            self.speech.pause, self.speech.resume,
            speech_start=lambda: self.calls.append(('speech_start',)),
            speech_cancel=lambda: self.calls.append(('speech_cancel',)),
            playback_done=lambda key, failed=False: self.calls.append(('done', key)))
        self.turn.enable()
        self.turn.set_ready(True)
        self.music = Music(self.turn, self.calls)
        self.desktop = SimpleNamespace(turn=self.turn, speech=self.speech, music=self.music,
            audio=SimpleNamespace(is_playing=lambda: False), link=SimpleNamespace(connected=True),
            closed=False, playing=False, pending_event=None, ready_media=None,
            restarting=False, speech_pane=None, manager=None, speech_state='listening', notice='')
        def screen_factory(*args, **kwargs):
            screen = Screen(*args, **kwargs)
            self.screens.append(screen)
            return screen
        self.controller = StandbyController(self.window, self.desktop, '/installed/desktop',
            sink_name='usb-sink', screen_factory=screen_factory, clock=lambda: self.now,
            schedule=False)

    def enter_sleep(self):
        self.controller.handle_final('小me，睡觉去吧')
        self.controller.poll()  # Waiting for in-flight ordinary output.
        self.controller.poll()  # Reserve local standby and pause capture.
        self.assertEqual('pausing', self.controller.state)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('sleeping', self.controller.state)
        return self.controller.token

    def begin_wake(self, token=None):
        self.controller.on_wake(self.controller.token if token is None else token)
        self.controller.poll()
        self.assertEqual('wake_pausing', self.controller.state)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('waking', self.controller.state)

    def test_sleep_command_consumed_without_model_request(self):
        self.controller.speech_started()
        self.enter_sleep()
        self.assertFalse(any(call[0] in ('send', 'normal', 'done') for call in self.calls))
        self.assertIn(('speech_cancel',), self.calls)
        self.assertEqual('standby', self.turn.phase)

    def test_exact_sleep_command_is_local_and_does_not_submit_a_model_turn(self):
        for text in ('小me，睡觉。', '小me，睡觉吧', '小猫，睡觉吧', '小猫，睡觉。',
                     '小猫，睡觉了。', '小me，睡觉吧。'):
            with self.subTest(text=text):
                self.setUp()
                self.controller.handle_final(text)
                self.assertTrue(self.controller.pending_sleep)
                self.assertFalse(any(call[0] in ('send','normal','done') for call in self.calls))

    def test_exact_wake_command_while_awake_is_ordinary_chat(self):
        for text in ('小me，起床', '小me，起床吧', '小猫，起床吧', '小猫，起床。',
                     '小猫，起床了。', '小me，起床吧。'):
            with self.subTest(text=text):
                self.setUp()
                self.assertEqual('request-1', self.controller.handle_final(text))
                self.assertIn(('send',text), self.calls)
                self.assertEqual('awake', self.controller.state)
                self.assertEqual([], self.screens)

    def test_exact_dont_sleep_command_while_awake_is_ordinary_chat(self):
        self.assertEqual('request-1', self.controller.handle_final('小me，别睡了。'))
        self.assertIn(('send','小me，别睡了。'), self.calls)
        self.assertEqual('awake', self.controller.state)
        self.assertEqual([], self.screens)

    def test_ordinary_chat_keeps_existing_router(self):
        result = self.controller.handle_final('今天真开心')
        self.assertEqual('request-1', result)
        self.assertIn(('send', '今天真开心'), self.calls)

    def test_wake_while_awake_is_normal_chat(self):
        self.controller.handle_final('小me，别睡了，起床了')
        self.assertEqual('awake', self.controller.state)
        self.assertIn(('send', '小me，别睡了，起床了'), self.calls)
        self.assertEqual([], self.screens)

    def test_request_callback_does_not_operate_tk_or_pause(self):
        before = len(self.calls)
        worker = threading.Thread(target=self.controller.handle_final,
                                  args=('小me，睡觉去吧',))
        worker.start()
        worker.join()
        self.assertEqual([], self.screens)
        self.assertEqual(before, len(self.calls))
        self.assertTrue(self.controller.status()['requested'])
        self.controller.poll()
        self.controller.poll()
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual(threading.get_ident(), self.screens[0].thread)

    def test_sleep_waits_for_current_reply_and_voice_close(self):
        self.turn.submit_final('先前消息')
        self.controller.request_sleep()
        self.controller.poll()
        self.assertEqual('waiting', self.controller.state)
        self.controller.poll()
        self.assertEqual([], self.screens)
        self.turn.failure()
        self.controller.poll()
        self.assertEqual('pausing', self.controller.state)
        self.controller.poll()
        self.assertEqual([], self.screens)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual(1, len(self.screens))

    def test_sleep_uses_given_asset_directory_and_usb_sink(self):
        self.enter_sleep()
        options = self.screens[0].options
        self.assertEqual(Path('/installed/desktop'), options['directory'])
        self.assertEqual('usb-sink', options['sink_name'])

    def test_wake_only_control_installed_before_mic_resume(self):
        token = self.enter_sleep()
        enable_index = self.calls.index(('control', True, token))
        resume_index = next(i for i in range(enable_index + 1, len(self.calls))
                            if self.calls[i] == ('resume',))
        self.assertLess(enable_index, resume_index)

    def test_normal_speech_during_standby_never_wakes_or_reaches_model(self):
        self.enter_sleep()
        before = len(self.calls)
        self.controller.speech_started()
        self.controller.handle_final('今天的天气怎么样')
        self.controller.speech_cancelled()
        self.controller.poll()
        self.assertEqual('sleeping', self.controller.state)
        self.assertEqual(before, len(self.calls))
        self.assertEqual(0, self.screens[0].wakes)

    def test_standby_endpoint_rearms_only_control_capture(self):
        token = self.enter_sleep()
        before = len(self.calls)
        self.controller.on_utterance(token)
        self.controller.poll()
        self.assertEqual([('resume',)], self.calls[before:])
        self.assertEqual('standby', self.turn.phase)

    def test_old_wake_and_endpoint_tokens_ignored(self):
        self.enter_sleep()
        before = len(self.calls)
        self.controller.on_wake('old-token')
        self.controller.on_utterance('old-token')
        self.controller.poll()
        self.assertEqual(before, len(self.calls))
        self.assertEqual('sleeping', self.controller.state)

    def test_valid_wake_callback_only_records_intent(self):
        token = self.enter_sleep()
        before = len(self.calls)
        self.controller.on_wake(token)
        self.assertEqual(before, len(self.calls))
        self.assertEqual(0, self.screens[0].wakes)
        self.assertEqual('sleeping', self.controller.state)

    def test_wake_animation_waits_for_paused_ack(self):
        self.enter_sleep()
        self.controller.on_wake(self.controller.token)
        self.controller.poll()
        self.controller.poll()
        self.assertEqual('wake_pausing', self.controller.state)
        self.assertEqual(0, self.screens[0].wakes)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual(1, self.screens[0].wakes)

    def test_normal_chat_only_resumes_after_animation_completion(self):
        self.enter_sleep()
        self.begin_wake()
        before = len(self.calls)
        self.turn.tick(now=float('inf'))
        self.controller.handle_final('你好')
        self.controller.poll()
        self.assertEqual('waking', self.controller.state)
        self.assertFalse(any(call[0] in ('send', 'resume') for call in self.calls[before:]))
        self.screens[0].complete()
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertEqual('cooldown', self.turn.phase)
        self.turn.tick(now=float('inf'))
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual(('resume',), self.calls[-1])

    def test_touch_wake_obeys_same_pause_and_animation_handover(self):
        self.enter_sleep()
        self.screens[0].touch()
        self.controller.poll()
        self.assertEqual('wake_pausing', self.controller.state)
        self.assertEqual(0, self.screens[0].wakes)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual(1, self.screens[0].wakes)

    def test_duplicate_wakes_do_not_restart_animation(self):
        token = self.enter_sleep()
        self.begin_wake()
        self.controller.on_wake(token)
        self.controller.poll()
        self.assertEqual(1, self.screens[0].wakes)

    def test_idle_twenty_seconds_enters_wakeable_standby(self):
        self.now += 21
        self.speech.last_activity = self.now
        self.controller.poll()
        self.assertEqual('pausing', self.controller.state)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('sleeping', self.controller.state)

    def test_busy_music_does_not_auto_enter_standby(self):
        self.now += 25
        self.speech.last_activity = self.now
        self.music.active = True
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)

    def test_recording_heartbeat_does_not_postpone_idle(self):
        for _ in range(21):
            self.now += 1
            self.speech.last_activity = self.now
            self.controller.poll()
        self.assertEqual('pausing', self.controller.state)

    def test_real_speech_activity_postpones_idle(self):
        self.now += 19
        self.controller.speech_started()
        self.now += 4
        self.speech.last_activity = self.now
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)

    def test_music_sleep_token_stops_music_and_waits_real_exit(self):
        self.music.active, self.music.token = True, 'music-1'
        self.turn.phase = 'music'
        self.assertTrue(self.controller.on_sleep('music-1'))
        self.controller.poll()
        self.assertIn(('music_command', {'action': 'stop'}), self.calls)
        self.controller.poll()
        self.assertEqual('waiting', self.controller.state)
        self.assertEqual([], self.screens)
        self.music.active = False
        self.turn.phase = 'listening'
        self.controller.poll()
        self.assertEqual('pausing', self.controller.state)

    def test_stale_music_sleep_token_cannot_stop_new_music(self):
        self.music.active, self.music.token = True, 'music-new'
        self.assertFalse(self.controller.on_sleep('music-old'))
        self.controller.poll()
        self.assertFalse(any(call[0] == 'music_command' for call in self.calls))
        self.assertEqual('awake', self.controller.state)

    def test_sleep_prepare_timeout_aborts_without_automatic_resume(self):
        self.controller.request_sleep()
        self.controller.poll()
        self.controller.poll()
        before = len(self.calls)
        self.now += 6
        events = self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertEqual('error', events[-1]['type'])
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))
        self.assertTrue(self.controller.last_error)

    def test_wake_pause_timeout_does_not_bypass_animation(self):
        self.enter_sleep()
        self.controller.on_wake(self.controller.token)
        self.controller.poll()
        self.now += 6
        self.controller.poll()
        self.assertEqual(0, self.screens[0].wakes)
        self.assertEqual('awake', self.controller.state)
        self.assertTrue(self.controller.last_error)

    def test_disconnect_aborts_and_never_reopens_mic(self):
        self.enter_sleep()
        before = len(self.calls)
        self.desktop.link.connected = False
        self.turn.set_ready(False)
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertEqual(1, self.screens[0].closes)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_usb_loss_aborts_and_never_reopens_mic(self):
        self.enter_sleep()
        before = len(self.calls)
        self.speech.alive = False
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_manual_pause_aborts_without_changing_voice_wanted(self):
        self.desktop.voice_wanted = True
        self.enter_sleep()
        self.turn.disable()
        self.controller.abort()
        self.assertTrue(self.desktop.voice_wanted)
        self.assertFalse(self.turn.enabled)
        self.assertEqual('awake', self.controller.state)

    def test_screen_render_error_aborts_instead_of_faking_wake_completion(self):
        self.enter_sleep()
        before = len(self.calls)
        self.screens[0].error('broken gif')
        self.controller.poll()
        self.assertEqual('broken gif', self.controller.last_error)
        self.assertEqual('recovering', self.controller.state)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_healthy_resource_error_returns_to_chat_after_capture_close(self):
        self.enter_sleep()
        self.screens[0].error('broken gif')
        self.controller.poll()
        self.assertEqual('recovering', self.controller.state)
        before = len(self.calls)
        self.controller.poll()
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual(('resume',), self.calls[-1])

    def test_resource_error_with_audio_output_unavailable_never_resumes(self):
        self.desktop.audio_output = SimpleNamespace(ready=lambda: False)
        self.enter_sleep()
        before = len(self.calls)
        self.screens[0].error('USB lost')
        self.controller.poll()
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_external_abort_cancels_pending_internal_resource_recovery(self):
        self.enter_sleep()
        self.screens[0].error('broken gif')
        self.controller.poll()
        self.assertEqual('recovering', self.controller.state)
        self.controller.abort()
        before = len(self.calls)
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_recovery_ack_timeout_requests_manual_restart(self):
        self.enter_sleep()
        self.screens[0].error('broken gif')
        self.controller.poll()
        self.now += 6
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertIn('请暂停后重新开启语音', self.desktop.notice)

    def test_user_voice_pause_during_resource_error_never_resumes(self):
        self.enter_sleep()
        self.desktop.voice_wanted = False
        before = len(self.calls)
        self.screens[0].error('broken gif')
        self.controller.poll()
        self.speech.paused = True
        self.controller.poll()
        self.assertEqual('awake', self.controller.state)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_previous_screen_cannot_complete_or_abort_new_standby(self):
        self.enter_sleep()
        old_screen = self.screens[0]
        self.controller.abort()
        self.turn.enable()
        self.turn.set_ready(True)
        self.enter_sleep()
        self.begin_wake()
        old_screen.complete()
        old_screen.error('old failure')
        self.controller.poll()
        self.assertEqual('waking', self.controller.state)
        self.assertIsNone(self.controller.last_error)

    def test_close_cancels_tk_job_and_is_idempotent(self):
        self.enter_sleep()
        self.controller.job = 99
        before = len(self.calls)
        self.controller.close()
        self.controller.close()
        self.assertEqual([99], self.window.cancelled)
        self.assertEqual([], self.controller.poll())
        self.assertEqual(1, self.screens[0].closes)
        self.assertFalse(any(call[0] == 'resume' for call in self.calls[before:]))

    def test_api_wake_when_awake_is_noop(self):
        self.assertEqual('awake', self.controller.command({'action': 'wake'})['state'])
        self.assertEqual([], self.screens)

    def test_api_rejects_arbitrary_files_or_malformed_commands(self):
        for body in (None, {}, {'action': 'play'}, {'action': 'sleep', 'path': '/tmp/a'}):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.controller.command(body)

    def test_api_sleep_requires_ready_devices(self):
        self.speech.alive = False
        with self.assertRaises(ValueError):
            self.controller.command({'action': 'sleep'})

    def test_health_frame_is_displayed_snapshot_without_tk_access(self):
        self.assertIsNone(self.controller.status()['frame'])
        self.assertFalse(self.controller.status()['animation_waking'])
        self.enter_sleep()
        screen = self.screens[0]
        screen.shown, screen.waking = (14, (800, 480)), False
        self.assertEqual(14, self.controller.status()['frame'])
        screen.shown, screen.waking = (54, (800, 480)), True
        status = []
        reader = threading.Thread(target=lambda: status.append(self.controller.status()))
        reader.start()
        reader.join()
        self.assertEqual(54, status[0]['frame'])
        self.assertTrue(status[0]['animation_waking'])
        screen.shown = None  # No next frame has been presented yet.
        self.assertIsNone(self.controller.status()['frame'])
        self.controller.abort()
        self.assertIsNone(self.controller.status()['frame'])
        self.assertFalse(self.controller.status()['animation_waking'])


if __name__ == '__main__':
    unittest.main()
