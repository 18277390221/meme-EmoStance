"""Preserve the existing Pi ASR selection and Paraformer silence endpoint."""
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / 'asr_runtime'))
from asr_selection import selection
from stt_bridge import find_asr_root

spec = importlib.util.spec_from_file_location('pi_asr_speech', DESKTOP / 'asr_runtime/speech.py')
speech = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speech)


class SelectionTests(unittest.TestCase):
    def test_missing_config_preserves_lite_default(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual({'profile': 'lite', 'threads': 2, 'model': None, 'silence_rms': .008},
                             selection(directory))

    def test_paraformer_explicit_model_is_validated_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'chosen-model'
            model.mkdir()
            for name in ('encoder.int8.onnx', 'decoder.int8.onnx', 'tokens.txt'):
                (model / name).touch()
            (root / 'asr-selection.json').write_text(json.dumps(
                {'profile': 'paraformer', 'model': 'chosen-model', 'threads': 3, 'silence_rms': .01}))
            config = selection(root)
            self.assertEqual('paraformer', config['profile'])
            self.assertEqual(model, config['model'])
            self.assertEqual(3, config['threads'])
            self.assertEqual(.01, config['silence_rms'])
            (model / 'decoder.int8.onnx').unlink()
            with self.assertRaisesRegex(ValueError, 'Paraformer'):
                selection(root)

    def test_custom_model_allows_existing_python_without_lite_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.venv/bin').mkdir(parents=True)
            (root / '.venv/bin/python').touch()
            with patch('stt_bridge.selection', return_value={'model': root / 'other-model'}):
                self.assertEqual(root, find_asr_root(root / 'player', explicit=root))

    def test_invalid_profile_and_threads_do_not_silently_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for config in ({'profile': 'unknown'}, {'profile': 'lite', 'threads': True},
                           {'profile': 'paraformer', 'threads': 3}):
                (root / 'asr-selection.json').write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    selection(root)


class Samples(list):
    def astype(self, _dtype):
        return self
    def __truediv__(self, value):
        return Samples(sample / value for sample in self)
    def __mul__(self, other):
        return Samples(first * second for first, second in zip(self, other))


class Numpy:
    float32 = 'float32'
    @staticmethod
    def frombuffer(data, dtype):
        return Samples(struct.unpack('<' + 'h' * (len(data) // 2), data))
    @staticmethod
    def zeros(count, dtype):
        return Samples([0.0] * count)
    mean = staticmethod(lambda samples: sum(samples) / len(samples))
    sqrt = staticmethod(math.sqrt)


class Recognizer:
    def __init__(self):
        self.text = ''
        self.streams = []
    def create_stream(self):
        recognizer = self
        class Stream:
            def __init__(self):
                self.blocks = []
            def accept_waveform(self, rate, samples):
                self.blocks.append((rate, len(samples)))
            def input_finished(self):
                recognizer.text = '完整尾部句子'
        stream = Stream()
        self.streams.append(stream)
        self.text = ''
        return stream
    def is_ready(self, stream):
        return False
    def is_endpoint(self, stream):
        return False
    def get_result(self, stream):
        return self.text


class SpeechStartTests(unittest.TestCase):
    def test_silent_pcm_does_not_claim_speech_started(self):
        recognizer, starts = Recognizer(), []
        decoder = speech.Decoder(recognizer, Numpy, on_speech_start=lambda: starts.append(True))
        decoder.feed(b'\0\0' * 320)
        self.assertEqual([], starts)
        recognizer.text = '你好'
        decoder.feed(b'\0\0' * 320)
        decoder.feed(b'\0\0' * 320)
        self.assertEqual([True], starts)

    def test_paraformer_preserves_pcm_endpoint_and_tail_flush(self):
        recognizer, events = Recognizer(), []
        decoder = speech.Decoder(recognizer, Numpy, profile='paraformer', endpoint_silence=.5,
                                 output=lambda text, **kw: events.append(('final', text)),
                                 handler=lambda text: None,
                                 on_speech_start=lambda: events.append(('start',)))
        voiced = struct.pack('<' + 'h' * 1600, *([1000] * 1600))
        decoder.feed(voiced)
        self.assertEqual([('start',)], events)
        for _index in range(4):
            decoder.feed(b'\0\0' * 1600)
        self.assertEqual([('start',)], events)
        decoder.feed(b'\0\0' * 1600)
        self.assertEqual([('start',), ('final', '完整尾部句子')], events)
        self.assertEqual((16000, 10560), recognizer.streams[0].blocks[-1])
        self.assertEqual('pcm-energy-silence', decoder.stats()['endpoint'])
        self.assertEqual('paraformer', decoder.stats()['profile'])

    def test_point_seven_endpoint_keeps_a_short_pause_and_restarts_silence_after_more_speech(self):
        recognizer, events = Recognizer(), []
        decoder = speech.Decoder(recognizer, Numpy, profile='paraformer', endpoint_silence=.7,
                                 output=lambda text, **kw: events.append(('final', text)),
                                 handler=lambda text: None,
                                 on_speech_start=lambda: events.append(('start',)))
        voiced = struct.pack('<' + 'h' * 1600, *([1000] * 1600))
        silence = b'\0\0' * 1600
        decoder.feed(voiced)
        recognizer.text = '我'
        for _index in range(6):
            decoder.feed(silence)
        self.assertEqual([('start',)], events)  # 0.6 seconds cannot submit "我".
        decoder.feed(voiced)
        recognizer.text = '我还在说'
        for _index in range(6):
            decoder.feed(silence)
        self.assertEqual([('start',)], events)  # Resumed voice restarted the timer.
        decoder.feed(silence)
        self.assertEqual([('start',), ('final', '完整尾部句子')], events)
        self.assertEqual((16000, 10560), recognizer.streams[0].blocks[-1])


if __name__ == '__main__':
    unittest.main()
