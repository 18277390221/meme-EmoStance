"""Keep GIF WAV audio on the selected USB speaker after a reconnect."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import wave

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
from native_core import Audio


def wav_bytes():
    output = BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b'\0\0' * 320)
    return output.getvalue()


class AudioOutputSinkTests(unittest.TestCase):
    sink = 'alsa_output.usb-UACDemoV1.0-00.analog-stereo'

    def test_explicit_sink_uses_pulse_alsa_plugin_and_preserves_wav_bytes(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (None, b'')
        process.poll.return_value = 0
        player = Audio(sink_name=self.sink)
        blob = wav_bytes()
        with patch('native_core.sys.platform', 'linux'), \
                patch('native_core.shutil.which', return_value='/fake/aplay'), \
                patch('native_core.subprocess.Popen', return_value=process) as popen, \
                patch.dict(os.environ, {'PULSE_SINK': 'board-output'}):
            player.play(blob)
            self.assertEqual('board-output', os.environ['PULSE_SINK'])
        arguments, options = popen.call_args
        self.assertEqual((['/fake/aplay', '-D', 'pulse', '-q'],), arguments)
        self.assertEqual(self.sink, options['env']['PULSE_SINK'])
        self.assertNotIn('shell', options)
        deadline = time.monotonic() + 1
        while not process.communicate.called and time.monotonic() < deadline:
            time.sleep(.005)
        process.communicate.assert_called_once_with(input=blob)
        self.assertEqual([], player.pop_errors())

    def test_default_sink_preserves_existing_alsa_command_and_environment(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (None, b'')
        process.poll.return_value = 0
        with patch('native_core.sys.platform', 'linux'), \
                patch('native_core.shutil.which', return_value='/fake/aplay'), \
                patch('native_core.subprocess.Popen', return_value=process) as popen:
            Audio().play(wav_bytes())
        self.assertEqual((['/fake/aplay', '-q'],), popen.call_args.args)
        self.assertNotIn('env', popen.call_args.kwargs)

    def test_invalid_sink_names_are_rejected_before_playback(self):
        for sink in ('', ' ', 'sink\nother', 'sink\0', '$(echo bad)', '/a/path', 1, True):
            with self.subTest(sink=sink), self.assertRaises(ValueError):
                Audio(sink_name=sink)

    def test_real_launch_failure_reaches_the_existing_caller_error_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / 'missing-aplay')
            with patch('native_core.sys.platform', 'linux'), \
                    patch('native_core.shutil.which', return_value=executable):
                with self.assertRaises(FileNotFoundError):
                    Audio(sink_name=self.sink).play(wav_bytes())

    @unittest.skipIf(sys.platform == 'win32', 'Executable test fixture uses a POSIX shebang')
    def test_child_rejecting_selected_sink_exposes_error_without_fallback(self):
        # Run a real child rather than only faking Popen return codes. It models
        # aplay/Pulse rejecting the selected sink while an unrelated output is
        # still available; no hardware or sound server is needed for this test.
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'reject-output'
            executable.write_text('#!' + sys.executable + '\n'
                'import os, sys\n'
                'sys.stdin.buffer.read()\n'
                'sys.stderr.write("No such sink: " + os.environ["PULSE_SINK"])\n'
                'sys.exit(1)\n')
            executable.chmod(0o700)
            player = Audio(sink_name=self.sink)
            self.addCleanup(player.stop)
            with patch('native_core.sys.platform', 'linux'), \
                    patch('native_core.shutil.which', return_value=str(executable)):
                player.play(wav_bytes())
            errors = []
            deadline = time.monotonic() + 2
            while not errors and time.monotonic() < deadline:
                errors = player.pop_errors()
                time.sleep(.005)
            self.assertEqual(1, len(errors))
            self.assertIn('No such sink: ' + self.sink, errors[0])
            self.assertFalse(player.is_playing())

    def test_windows_keeps_winsound_and_does_not_start_a_pulse_child(self):
        winsound = Mock(SND_FILENAME=1, SND_ASYNC=2)
        player = Audio(sink_name=self.sink)
        with patch('native_core.sys.platform', 'win32'), \
                patch.dict(sys.modules, {'winsound': winsound}), \
                patch('native_core.subprocess.Popen') as popen:
            player.play(wav_bytes())
            self.assertTrue(player.is_playing())
            popen.assert_not_called()
            self.assertEqual(3, winsound.PlaySound.call_args.args[1])
            player.stop()


if __name__ == '__main__':
    unittest.main()
