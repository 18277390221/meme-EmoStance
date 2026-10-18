"""USB capture failures are recoverable; ASR/configuration faults are not."""
from collections import deque
from contextlib import redirect_stdout
import errno
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / 'asr_runtime'))

from audio import Capture, CaptureError, UserError, capture_failure_reason, require_arecord
from capture_turn import MicrophoneGate, listen_once
from stt_worker import emit_error


class CaptureErrorTests(unittest.TestCase):
    def test_real_alsa_disconnect_diagnostics_allow_recovery(self):
        for detail in (
                'arecord: pcm_read:2221: read error: No such device',
                'arecord: pcm_read:2221: read error: Input/output error',
                "ALSA lib confmisc.c:855:(parse_card) cannot find card 'UACDemoV10'",
                'ALSA lib confmisc.c:165:(snd_config_get_card) Cannot get card index for UACDemoV10',
                'arecord: device_list:274: no soundcards found...',
                'USB audio device disconnected'):
            with self.subTest(detail=detail):
                error = CaptureError('用户可见消息', detail=detail)
                self.assertEqual('audio_device_unavailable', error.reason)
                self.assertTrue(error.recoverable)
                self.assertEqual('用户可见消息', str(error))

    def test_read_errno_is_preserved_and_hardware_errors_allow_recovery(self):
        for code in (errno.ENODEV, errno.ENXIO, errno.EIO):
            with self.subTest(code=code):
                error = CaptureError('设备故障', error=OSError(code, 'capture failed'))
                self.assertTrue(error.recoverable)
                self.assertEqual('audio_device_unavailable', error.reason)

    def test_other_recording_errors_do_not_trigger_reconnect(self):
        for detail, reason in (
                ('arecord: main:831: audio open error: Device or resource busy', 'audio_device_busy'),
                ('arecord: main:831: audio open error: Permission denied', 'audio_permission'),
                ('arecord: set_params:1377: Sample format non available', 'audio_configuration'),
                ('arecord: set_params:1377: Channels count non available', 'audio_configuration'),
                ('arecord: main:831: audio open error: Invalid argument', 'audio_configuration'),
                ('ALSA lib pcm.c:2666:(snd_pcm_open_noupdate) Unknown PCM mistyped', 'audio_configuration'),
                ('overrun!!! (at least 10.315 ms long)', 'audio_overrun'),
                ('arecord: main:831: audio open error: No such file or directory', 'audio_capture_error'),
                ('', 'audio_capture_error')):
            with self.subTest(detail=detail):
                error = CaptureError('不应循环重试', detail=detail)
                self.assertEqual(reason, error.reason)
                self.assertFalse(error.recoverable)

    def test_missing_card_evidence_precedes_followup_unknown_pcm(self):
        missing = "ALSA lib confmisc.c:855:(parse_card) cannot find card 'UACDemoV10'"
        unknown = 'ALSA lib pcm.c:2666:(snd_pcm_open_noupdate) Unknown PCM plughw:CARD=UACDemoV10,DEV=0'
        for detail in (missing + '\n' + unknown, unknown + '\n' + missing):
            with self.subTest(detail=detail):
                error = CaptureError('卡缺失的完整诊断', detail=detail)
                self.assertEqual('audio_device_unavailable', error.reason)
                self.assertTrue(error.recoverable)
        for detail, reason in (
                ('Permission denied', 'audio_permission'),
                ('Device or resource busy', 'audio_device_busy'),
                ('overrun!!!', 'audio_overrun'),
                ('Invalid argument', 'audio_configuration')):
            with self.subTest(reason=reason):
                error = CaptureError('明确的其他故障', detail=detail + '\n' + missing + '\n' + unknown)
                self.assertEqual(reason, error.reason)
                self.assertFalse(error.recoverable)

    def test_missing_executable_is_not_a_missing_microphone(self):
        error = CaptureError('cannot start arecord', error=FileNotFoundError(errno.ENOENT, 'arecord'))
        self.assertFalse(error.recoverable)
        with patch('audio.shutil.which', return_value=None):
            with self.assertRaises(UserError) as raised:
                require_arecord()
        self.assertEqual('audio_tool_missing', raised.exception.reason)
        self.assertFalse(raised.exception.recoverable)

    def test_permission_busy_and_parameter_errno_override_ambiguous_text(self):
        for code, reason in ((errno.EACCES, 'audio_permission'),
                             (errno.EPERM, 'audio_permission'),
                             (errno.EBUSY, 'audio_device_busy'),
                             (errno.EINVAL, 'audio_configuration')):
            with self.subTest(code=code):
                self.assertEqual(reason, capture_failure_reason('', OSError(code, 'capture failed')))

    def test_capture_classifies_actual_child_stderr_and_reaps_it(self):
        for detail, reason, recoverable in (
                ('arecord: pcm_read:2221: read error: No such device', 'audio_device_unavailable', True),
                ('arecord: pcm_read:2221: read error: Input/output error', 'audio_device_unavailable', True),
                ('arecord: main:831: audio open error: Device or resource busy', 'audio_device_busy', False)):
            with self.subTest(detail=detail):
                command = [sys.executable, '-c', 'import sys; sys.stderr.write(' + repr(detail + '\n') + ')']
                capture = Capture(command, strict=True, frame_ms=20, new_session=False)
                with self.assertRaises(CaptureError) as raised:
                    with capture:
                        list(capture.chunks())
                self.assertEqual(reason, raised.exception.reason)
                self.assertEqual(recoverable, raised.exception.recoverable)
                self.assertIsNotNone(capture.proc.poll())
                self.assertFalse(capture.reader.is_alive())
                self.assertFalse(capture.audio_reader.is_alive())

    def test_device_error_stops_before_buffered_pcm_reaches_decoder(self):
        capture = Capture(['fake-arecord'], strict=True)
        capture.blocks = deque([(b'\0\0' * 320, 0, time.monotonic())])
        capture.queued_bytes = 640
        capture.stream_error = '录音设备已断开'
        capture.stream_error_reason = 'audio_device_unavailable'
        with self.assertRaises(CaptureError) as raised:
            next(capture.chunks())
        self.assertTrue(raised.exception.recoverable)
        self.assertEqual(0, capture.delivered_bytes)

    def test_read_thread_carries_device_errno_to_the_event(self):
        capture = Capture(['fake-arecord'], strict=True)
        capture.proc, capture.reader = Mock(), Mock()
        capture.proc.stdout.fileno.return_value = 44
        with patch('audio.os.read', side_effect=OSError(errno.EIO, 'I/O failed')):
            capture._read_audio()
        self.assertTrue(capture.finished)
        self.assertEqual(errno.EIO, capture.read_exception.errno)
        with self.assertRaises(CaptureError) as raised:
            next(capture.chunks())
        self.assertTrue(raised.exception.recoverable)
        self.assertEqual('audio_device_unavailable', raised.exception.reason)

    def test_request_finish_cannot_hide_a_disconnect_diagnostic(self):
        capture = Capture(['fake-arecord'], strict=True)
        capture.normal_finish = True
        capture.finished = True
        capture.reader = Mock()
        capture.errors.append('arecord: pcm_read:2221: read error: No such device')
        with self.assertRaises(CaptureError) as raised:
            list(capture.chunks())
        self.assertTrue(raised.exception.recoverable)
        capture.reader.join.assert_called_once()

    def test_delayed_combined_stderr_after_pcm_eof_preserves_disconnect(self):
        missing = "ALSA lib confmisc.c:855:(parse_card) cannot find card 'UACDemoV10'"
        unknown = 'ALSA lib pcm.c:2666:(snd_pcm_open_noupdate) Unknown PCM plughw:CARD=UACDemoV10,DEV=0'
        for normal_finish in (False, True):
            with self.subTest(normal_finish=normal_finish):
                capture = Capture(['fake-arecord'], strict=True)
                capture.normal_finish = normal_finish
                capture.finished = True
                capture.reader = Mock()
                # PCM EOF was already observed. The final diagnostics become
                # available only when chunks joins the stderr reader.
                capture.reader.join.side_effect = lambda **_: capture.errors.extend((missing, unknown))
                with self.assertRaises(CaptureError) as raised:
                    list(capture.chunks())
                self.assertEqual('audio_device_unavailable', raised.exception.reason)
                self.assertTrue(raised.exception.recoverable)
                self.assertIn(missing, str(raised.exception))
                self.assertIn(unknown, str(raised.exception))
                capture.reader.join.assert_called_once()

    def test_intentional_terminate_is_still_a_successful_pause(self):
        capture = Capture(['fake-arecord'], strict=True)
        capture.normal_finish = True
        capture.finished = True
        capture.reader = Mock()
        capture.errors.append('Aborted by signal Terminated...')
        self.assertEqual([], list(capture.chunks()))

    def test_overrun_backlog_format_and_timeout_are_not_hardware_recovery(self):
        for reason in ('audio_overrun', 'audio_backlog'):
            capture = Capture(['fake-arecord'], strict=True)
            capture.stream_error = '保留原有错误消息'
            capture.stream_error_reason = reason
            with self.assertRaises(CaptureError) as raised:
                next(capture.chunks())
            self.assertEqual(reason, raised.exception.reason)
            self.assertFalse(raised.exception.recoverable)
        capture = Capture(['fake-arecord'], strict=True)
        capture.format_error = True
        with self.assertRaises(CaptureError) as raised:
            next(capture.chunks())
        self.assertEqual('audio_configuration', raised.exception.reason)
        self.assertFalse(raised.exception.recoverable)
        capture.format_error = False
        capture.last_data = time.monotonic() - 16
        with self.assertRaises(CaptureError) as raised:
            next(capture.chunks())
        self.assertEqual('audio_timeout', raised.exception.reason)
        self.assertFalse(raised.exception.recoverable)


class WorkerErrorEventTests(unittest.TestCase):
    def frame(self, error):
        output = io.StringIO()
        with redirect_stdout(output):
            emit_error(error)
        return json.loads(output.getvalue())

    def test_device_error_event_preserves_message_and_adds_recovery_contract(self):
        frame = self.frame(CaptureError('原有 ALSA 错误', detail='read error: No such device'))
        self.assertEqual({'type': 'error', 'message': '原有 ALSA 错误',
                          'reason': 'audio_device_unavailable', 'recoverable': True}, frame)

    def test_model_errors_and_arbitrary_oserror_are_never_retried_as_devices(self):
        for error in (UserError('本地模型缺失：tokens.txt'),
                      OSError(errno.EIO, 'model file Input/output error'),
                      RuntimeError('model no such device'),
                      ValueError('invalid argument'), ImportError('cannot load sherpa_onnx')):
            with self.subTest(error=type(error).__name__):
                frame = self.frame(error)
                self.assertEqual('asr_error', frame['reason'])
                self.assertFalse(frame['recoverable'])
                self.assertEqual(str(error), frame['message'])

    def test_message_is_still_bounded(self):
        self.assertEqual(1200, len(self.frame(UserError('x' * 1400))['message']))

    def test_capture_failure_after_endpoint_never_emits_truncated_final(self):
        gate, events, closed = MicrophoneGate(), [], []

        class FailedCapture:
            def __init__(self, *_args, **_kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                closed.append(True)
            def chunks(self):
                yield b'\0\0' * 320
                raise CaptureError('设备失联', detail='read error: No such device')
            def request_finish(self):
                pass
            def stats(self):
                return {}

        class Decoder:
            utterance_seen = True
            def __init__(self, finalized):
                self.finalized = finalized
            def feed(self, *_args):
                self.finalized('不能上传的残句')
            def stats(self):
                return {}

        with self.assertRaises(CaptureError):
            listen_once(gate, gate.epoch, ['fake-arecord'], Decoder,
                        lambda kind, **values: events.append(kind), FailedCapture)
        self.assertTrue(closed)
        self.assertNotIn('final', events)


if __name__ == '__main__':
    unittest.main()
