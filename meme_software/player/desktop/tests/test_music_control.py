"""Control-only ASR can stop music before a silence endpoint, without lyric IPC."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

DESKTOP = Path(__file__).resolve().parents[1]
ROOT = DESKTOP.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(Path(__file__).parent))
from capture_turn import MicrophoneGate
from stt_bridge import SpeechProcess
from stt_worker import MusicASRControl, MusicControlledDecoder
from test_asr_selection_and_endpoint import Numpy, Recognizer


class ASRControlTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.0]
        self.events = []
        self.control = MusicASRControl(lambda kind, **values: self.events.append({'type': kind, **values}),
                                       clock=lambda: self.now[0])
        self.gate = MicrophoneGate()
        self.epoch = self.gate.epoch

    def test_control_mode_partial_stops_all_profiles_without_silence_or_final(self):
        for profile in ('paraformer', 'lite', 'standard'):
            with self.subTest(profile=profile):
                self.setUp()
                self.control.command(True, 'music-1')
                token = self.control.begin_window()
                recognizer = Recognizer()
                decoder = MusicControlledDecoder(recognizer, Numpy, profile=profile,
                    output=lambda *args, **kwargs: self.fail('music stop must not require final'),
                    handler=lambda text: None, music_control=self.control,
                    gate=self.gate, epoch=self.epoch, music_token=token)
                recognizer.text = '春天在哪里请停止播放'
                voiced = struct.pack('<' + 'h' * 320, *([1000] * 320))
                decoder.feed(voiced)
                self.assertEqual([], self.events)
                self.now[0] = .21
                decoder.feed(voiced)
                self.assertEqual([{'type': 'music_stop', 'token': 'music-1'}], self.events)
                self.assertFalse(self.gate.enabled)
                self.assertNotIn('text', self.events[0])

    def test_disabled_mode_or_old_playback_token_cannot_stop(self):
        recognizer = Recognizer()
        recognizer.text = '停止播放'
        self.control.command(True, 'old')
        self.control.begin_window()
        self.control.command(True, 'new')
        self.control.begin_window()
        for now in (0., .21, .42):
            self.now[0] = now
            self.control.sample(recognizer, None, self.gate, self.epoch, 'old')
        self.control.command(False)
        for now in (.63, .84):
            self.now[0] = now
            self.control.sample(recognizer, None, self.gate, self.epoch, 'new')
        self.assertEqual([], self.events)
        self.assertTrue(self.gate.enabled)

    def test_music_stream_never_emits_ordinary_speech_start_even_after_mode_changes(self):
        starts = []
        self.control.command(True, 'music-1')
        recognizer = Recognizer()
        decoder = MusicControlledDecoder(recognizer, Numpy, profile='paraformer',
            output=lambda *args, **kwargs: None, handler=lambda text: None,
            on_speech_start=lambda: starts.append('server lease'),
            music_control=self.control, gate=self.gate, epoch=self.epoch, music_token='music-1')
        self.control.command(False)
        recognizer.text = '迟到的音乐识别'
        decoder.feed(b'\0\0' * 320)
        self.assertEqual([], starts)

    def test_normal_stream_still_emits_ordinary_speech_start(self):
        starts = []
        recognizer = Recognizer()
        decoder = MusicControlledDecoder(recognizer, Numpy,
            output=lambda *args, **kwargs: None, handler=lambda text: None,
            on_speech_start=lambda: starts.append('server lease'),
            music_control=self.control, gate=self.gate, epoch=self.epoch, music_token=None)
        recognizer.text = '正常聊天'
        decoder.feed(b'\0\0' * 320)
        self.assertEqual(['server lease'], starts)

    def test_final_lyrics_and_late_closed_music_stream_never_become_chat_final(self):
        self.control.command(True, 'music-1')
        self.control.window_event('music-1', 'final', text='这是歌词不是对话')
        self.control.command(False)
        self.control.window_event('music-1', 'final', text='迟到的歌词也不能发送')
        self.assertEqual([{'type': 'music_utterance', 'token': 'music-1'}] * 2, self.events)

    def test_strict_stop_final_routes_locally_once(self):
        self.control.command(True, 'music-1')
        self.control.window_event('music-1', 'final', text='请停止播放')
        self.control.window_event('music-1', 'utterance_cancelled', message='ignored')
        self.assertEqual([{'type': 'music_stop', 'token': 'music-1'}], self.events)

    def test_normal_final_without_control_mode_is_unchanged(self):
        self.control.window_event(None, 'final', text='停止')
        self.assertEqual([{'type': 'final', 'text': '停止'}], self.events)

    def test_negated_stop_partial_never_stops_the_music(self):
        self.control.command(True, 'music-1')
        self.control.begin_window()
        recognizer = Recognizer()
        recognizer.text = '不要停止播放'
        for now in (0., .21, .42):
            self.now[0] = now
            self.control.sample(recognizer, None, self.gate, self.epoch, 'music-1')
        self.assertEqual([], self.events)
        self.assertTrue(self.gate.enabled)


class BridgeControlTests(unittest.TestCase):
    def test_music_events_have_separate_callbacks_and_never_call_on_final(self):
        with tempfile.TemporaryDirectory() as temporary:
            worker = Path(temporary) / 'fake_worker.py'
            frames = [{'type': 'music_stop', 'token': 'music-1'},
                      {'type': 'music_utterance', 'token': 'music-1'},
                      {'type': 'final', 'text': '普通聊天'}]
            worker.write_text('import json\nframes = ' + repr(frames) +
                              '\nfor frame in frames: print(json.dumps(frame), flush=True)\n')
            final, stop, utterance = [], [], []
            process = SpeechProcess(worker=worker, on_final=lambda text: final.append(text))
            process.on_music_stop = stop.append
            process.on_music_utterance = utterance.append
            process.start(temporary, 'test-device', python=sys.executable, end_silence=.7)
            process.thread.join(3)
            self.assertFalse(process.running())
            self.assertEqual(['普通聊天'], final)
            self.assertEqual(['music-1'], stop)
            self.assertEqual(['music-1'], utterance)
            events = process.drain()
            self.assertEqual(1, sum(event['type'] == 'final' for event in events))
            self.assertTrue(all('text' not in event for event in events
                                if event['type'].startswith('music_')))


if __name__ == '__main__':
    unittest.main()
