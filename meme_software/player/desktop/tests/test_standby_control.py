"""Standby ASR owns wake-only windows and keeps every other utterance local."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

DESKTOP = Path(__file__).resolve().parents[1]
ROOT = DESKTOP.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(Path(__file__).parent))
from capture_turn import MicrophoneGate
from stt_bridge import SpeechProcess
from stt_worker import LocalASRControls, MusicASRControl, MusicControlledDecoder, StandbyASRControl
from test_asr_selection_and_endpoint import Numpy, Recognizer


class StandbyWindowTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.]
        self.events = []
        self.output = lambda kind, **values: self.events.append({'type': kind, **values})
        self.control = StandbyASRControl(self.output, clock=lambda: self.now[0])
        self.control.command(True, 'standby-1')
        self.token = self.control.begin_window()
        self.gate = MicrophoneGate()
        self.epoch = self.gate.epoch
        self.recognizer = Recognizer()

    def sample(self, text, *times, token=None, epoch=None):
        self.recognizer.text = text
        for now in times:
            self.now[0] = now
            self.control.sample(self.recognizer, None, self.gate,
                                self.epoch if epoch is None else epoch,
                                self.token if token is None else token)

    def test_partial_wake_requires_two_spaced_hits_and_pauses_before_signal(self):
        paused = []
        self.control.output = lambda kind, **values: paused.append((kind, self.gate.enabled, values))
        self.sample('小me，别睡了', 0., .1)
        self.assertEqual([], paused)
        self.sample('小me，别睡了', .21)
        self.assertEqual([('standby_wake', False, {'token': 'standby-1'})], paused)
        self.assertTrue(self.control.woken)

    def test_partial_wake_all_supported_asr_profiles_without_endpoint(self):
        voiced = struct.pack('<' + 'h' * 320, *([1000] * 320))
        for profile in ('paraformer', 'lite', 'standard'):
            with self.subTest(profile=profile):
                self.setUp()
                decoder = MusicControlledDecoder(self.recognizer, Numpy, profile=profile,
                    output=lambda *args, **kwargs: self.fail('wake must not need an endpoint'),
                    handler=lambda text: None,
                    music_control=MusicASRControl(self.output), music_token=None,
                    standby_control=self.control, standby_token=self.token,
                    gate=self.gate, epoch=self.epoch)
                self.recognizer.text = '小me，别睡了，起床了'
                decoder.feed(voiced)
                self.assertEqual([], self.events)
                self.now[0] = .21
                decoder.feed(voiced)
                self.assertEqual([{'type': 'standby_wake', 'token': 'standby-1'}], self.events)

    def test_standby_speech_never_starts_a_server_lease_even_after_off(self):
        starts = []
        decoder = MusicControlledDecoder(self.recognizer, Numpy, profile='paraformer',
            output=lambda *args, **kwargs: None, handler=lambda text: None,
            on_speech_start=lambda: starts.append('server lease'),
            music_control=MusicASRControl(self.output), music_token=None,
            standby_control=self.control, standby_token=self.token,
            gate=self.gate, epoch=self.epoch)
        self.control.command(False)
        self.recognizer.text = '迟到的声音'
        decoder.feed(b'\0\0' * 320)
        self.assertEqual([], starts)
        self.assertEqual([], self.events)

    def test_unaddressed_or_quoted_commands_and_ordinary_chat_cannot_wake(self):
        for text in ('起床了', '别睡了', '你好', '我想听音乐', '小me，别起床',
                     '他说小me别睡了', '小me，如果我说别睡了你就起来'):
            with self.subTest(text=text):
                self.setUp()
                self.sample(text, 0., .21, .42)
                self.assertEqual([], self.events)
                self.assertTrue(self.gate.enabled)

    def test_invalid_partial_between_valid_hits_resets_stability(self):
        self.sample('小me，别睡了', 0.)
        self.sample('小me', .21)
        self.sample('小me，别睡了', .42)
        self.assertEqual([], self.events)
        self.sample('小me，别睡了', .63)
        self.assertEqual('standby_wake', self.events[0]['type'])

    def test_partial_hits_more_than_one_second_apart_do_not_trigger(self):
        self.sample('小me，别睡了', 0., 1.1)
        self.assertEqual([], self.events)
        self.sample('小me，别睡了', 1.31)
        self.assertEqual('standby_wake', self.events[0]['type'])

    def test_old_epoch_partial_cannot_wake_resumed_window(self):
        self.gate.command('pause')
        self.gate.command('resume')
        self.sample('小me，别睡了', 0., .21, .42, epoch=self.epoch)
        self.assertEqual([], self.events)
        self.assertTrue(self.gate.enabled)

    def test_old_token_partial_and_final_cannot_wake_a_new_standby(self):
        self.control.command(True, 'standby-2')
        self.control.begin_window()
        self.sample('小me，别睡了', 0., .21, token='standby-1')
        self.control.window_event('standby-1', 'final', text='小me，别睡了')
        self.control.window_event('standby-1', 'utterance_cancelled', message='late')
        self.assertEqual([], self.events)
        self.assertTrue(self.gate.enabled)

    def test_off_suppresses_all_late_standby_finals_and_cancel_events(self):
        self.control.command(False)
        for kind, text in (('final', '迟到的普通话'), ('final', '小me，别睡了'),
                           ('utterance_cancelled', '')):
            self.control.window_event('standby-1', kind, text=text)
        self.assertEqual([], self.events)

    def test_endpoint_valid_wake_once_and_no_text_leaves_worker(self):
        self.control.window_event(self.token, 'final', text='小me，别睡了')
        self.control.window_event(self.token, 'final', text='小me，别睡了')
        self.control.window_event(self.token, 'utterance_cancelled', message='late')
        self.assertEqual([{'type': 'standby_wake', 'token': self.token}], self.events)

    def test_exact_wake_shortcuts_stay_local_in_standby(self):
        for text in ('小me，起床', ' 小 ME ， 起床 。 ', '小咪，起床', '小米，起床。',
                     '小me，起床吧', ' 小 ME ， 起床吧 。 ', '小咪，起床吧', '小米，起床吧。',
                     '小猫，起床吧', '小猫，起床。', '小猫，起床了。', '小me，起床吧。',
                     '小me，起床了。', '小me，别睡了。'):
            with self.subTest(text=text):
                self.setUp()
                self.control.window_event(self.token, 'final', text=text)
                self.control.window_event(self.token, 'final', text=text)
                self.assertEqual([{'type':'standby_wake','token':self.token}], self.events)
                self.assertTrue(self.control.woken)

    def test_exact_sleep_phrase_during_standby_is_only_local_rearm(self):
        for text in ('小me，睡觉。', '小me，睡觉吧', '小猫，睡觉吧', '小猫，睡觉。',
                     '小猫，睡觉了。', '小me，睡觉吧。'):
            with self.subTest(text=text):
                self.setUp()
                self.control.window_event(self.token, 'final', text=text)
                self.assertEqual([{'type':'standby_utterance','token':self.token}], self.events)
                self.assertFalse(self.control.woken)

    def test_partial_then_endpoint_cannot_duplicate_wake(self):
        self.sample('小me，别睡了', 0., .21)
        self.control.window_event(self.token, 'final', text='小me，别睡了')
        self.assertEqual([{'type': 'standby_wake', 'token': self.token}], self.events)

    def test_nonwake_endpoint_is_only_token_rearm_event(self):
        self.control.window_event(self.token, 'final', text='我想听歌')
        self.control.window_event(self.token, 'utterance_cancelled', message='private detail')
        self.assertEqual([{'type': 'standby_utterance', 'token': self.token}] * 2, self.events)

    def test_lifecycle_pause_ack_is_preserved_for_capture_close_barrier(self):
        self.control.window_event(self.token, 'paused', pause_id='17')
        self.assertEqual([{'type': 'paused', 'pause_id': '17'}], self.events)

    def test_normal_final_without_standby_token_is_unchanged(self):
        self.control.window_event(None, 'final', text='小me，别睡了')
        self.assertEqual([{'type': 'final', 'text': '小me，别睡了'}], self.events)

    def test_begin_window_resets_partial_stability(self):
        self.sample('小me，别睡了', 0.)
        self.control.begin_window()
        self.sample('小me，别睡了', .21)
        self.assertEqual([], self.events)


class ModeAndMusicSleepTests(unittest.TestCase):
    def setUp(self):
        self.now, self.events = [0.], []
        self.controls = LocalASRControls(lambda kind, **values: self.events.append({'type': kind, **values}),
                                         clock=lambda: self.now[0])
        self.gate = MicrophoneGate()
        self.recognizer = Recognizer()

    def test_enabling_modes_is_mutually_exclusive(self):
        self.controls.command('music', True, 'music-1')
        self.assertEqual(('music-1', None), self.controls.begin_window())
        self.controls.command('standby', True, 'standby-1')
        self.assertEqual((None, 'standby-1'), self.controls.begin_window())
        self.controls.command('music', True, 'music-2')
        self.assertEqual(('music-2', None), self.controls.begin_window())

    def test_off_only_changes_its_own_mode(self):
        self.controls.command('standby', True, 'standby-1')
        self.controls.command('music', False)
        self.assertEqual((None, 'standby-1'), self.controls.begin_window())
        self.controls.command('music', True, 'music-1')
        self.controls.command('standby', False)
        self.assertEqual(('music-1', None), self.controls.begin_window())

    def test_music_partial_sleep_pauses_before_signal_and_never_sends_text(self):
        self.controls.command('music', True, 'music-1')
        self.controls.begin_window()
        self.recognizer.text = '小me，睡觉去吧'
        for now in (0., .21):
            self.now[0] = now
            self.controls.music.sample(self.recognizer, None, self.gate, 0, 'music-1')
        self.assertEqual([{'type': 'standby_sleep', 'token': 'music-1'}], self.events)
        self.assertFalse(self.gate.enabled)

    def test_music_sleep_endpoint_is_once_and_cancel_does_not_rearm(self):
        self.controls.command('music', True, 'music-1')
        self.controls.music.window_event('music-1', 'final', text='小me，睡觉去吧')
        self.controls.music.window_event('music-1', 'utterance_cancelled')
        self.assertEqual([{'type': 'standby_sleep', 'token': 'music-1'}], self.events)

    def test_music_stop_has_priority_over_sleep(self):
        self.controls.command('music', True, 'music-1')
        with patch('stt_worker.is_sleep_command', return_value=True):
            self.controls.music.window_event('music-1', 'final', text='停止播放')
        self.assertEqual([{'type': 'music_stop', 'token': 'music-1'}], self.events)

    def test_old_music_token_and_disabled_music_cannot_request_sleep(self):
        self.controls.command('music', True, 'music-2')
        self.controls.music.window_event('music-1', 'final', text='小me，睡觉去吧')
        self.controls.command('standby', True, 'standby-1')
        self.controls.music.window_event('music-2', 'final', text='小me，睡觉去吧')
        self.assertEqual([{'type': 'music_utterance', 'token': 'music-1'},
                          {'type': 'music_utterance', 'token': 'music-2'}], self.events)


class StandbyBridgeTests(unittest.TestCase):
    def test_control_commands_reject_whitespace_injection(self):
        process = SpeechProcess()
        commands = []
        process.command = commands.append
        for bad in (None, '', 'standby token', 'standby\nresume', 42):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                process.standby_control(True, bad)
        process.standby_control(True, 'standby-1')
        process.standby_control(False)
        self.assertEqual(['standby_control on standby-1', 'standby_control off'], commands)

    def test_standby_callbacks_never_use_normal_chat_callbacks(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'fake_worker.py'
            frames = [{'type': 'standby_wake', 'token': 'standby-1'},
                      {'type': 'standby_utterance', 'token': 'standby-1'},
                      {'type': 'standby_sleep', 'token': 'music-1'},
                      {'type': 'final', 'text': '普通聊天'}]
            worker.write_text('import json\nframes = ' + repr(frames) +
                              '\nfor frame in frames: print(json.dumps(frame), flush=True)\n')
            final, wake, utterance, sleep, starts = [], [], [], [], []
            process = SpeechProcess(worker=worker, on_final=final.append)
            process.on_standby_wake = wake.append
            process.on_standby_utterance = utterance.append
            process.on_standby_sleep = sleep.append
            process.on_speech_start = lambda: starts.append(True)
            process.start(temporary, 'test-device', python=sys.executable, end_silence=.7)
            process.thread.join(3)
            self.assertFalse(process.running())
            self.assertEqual(['普通聊天'], final)
            self.assertEqual(['standby-1'], wake)
            self.assertEqual(['standby-1'], utterance)
            self.assertEqual(['music-1'], sleep)
            self.assertEqual([], starts)
            self.assertFalse(any('text' in event for event in process.drain()
                                 if event['type'].startswith('standby_')))


if __name__ == '__main__':
    unittest.main()
