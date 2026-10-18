"""Exercise process ownership and late-token isolation without real audio."""
from io import BytesIO
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from meme_sing.catalog import Track
from meme_sing.playback import MusicPlayer, STDERR_LIMIT

REAL_POPEN = subprocess.Popen


class FakeProcess:
    def __init__(self, *, stderr=b'', ignore_terminate=False):
        self.stderr = BytesIO(stderr)
        self.returncode = None
        self.completed = threading.Event()
        self.terminated = 0
        self.killed = 0
        self.waits = 0
        self.ignore_terminate = ignore_terminate

    def finish(self, code=0):
        self.returncode = code
        self.completed.set()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.waits += 1
        if not self.completed.wait(timeout):
            raise subprocess.TimeoutExpired('ffplay', timeout)
        return self.returncode

    def terminate(self):
        self.terminated += 1
        if not self.ignore_terminate:
            self.finish(-15)

    def kill(self):
        self.killed += 1
        self.finish(-9)


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'music $(stay literal)_轻松_6.mp3'
        self.path.write_bytes(b'ID3localmusic')
        self.track = Track('music-1', '歌曲', self.path.name, self.path, {'轻松': 1.})
        self.processes = []
        self.calls = []
        self.next_process = None
        def popen(arguments, **kwargs):
            self.calls.append((arguments, kwargs))
            process = self.next_process or FakeProcess()
            self.next_process = None
            process.pid = 4000 + len(self.processes)
            self.processes.append(process)
            return process
        self.patcher = patch('meme_sing.playback.subprocess.Popen', side_effect=popen)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.player = MusicPlayer(executable='/fake/ffplay')
        self.addCleanup(self.player.close)

    def event(self, token, kind, timeout=2):
        deadline = time.monotonic() + timeout
        found = []
        while time.monotonic() < deadline:
            for event in self.player.drain():
                found.append(event)
                if event['token'] == token and event['type'] == kind:
                    return event
            time.sleep(.005)
        self.fail('Missing %s for %s, got %r' % (kind, token, found))

    def test_spawn_is_nonblocking_audio_only_and_owns_no_other_process(self):
        started = time.monotonic()
        token = self.player.start(self.track)
        self.assertLess(time.monotonic() - started, .2)
        event = self.event(token, 'started')
        self.assertEqual('music-1', event['track_id'])
        self.assertTrue(self.player.is_playing())
        arguments, kwargs = self.calls[0]
        self.assertEqual(['/fake/ffplay', '-nodisp', '-vn', '-autoexit', '-loglevel', 'error',
                          '-nostats', '-volume', '30', '-i', str(self.path)], arguments)
        self.assertEqual('pulseaudio', kwargs['env']['SDL_AUDIODRIVER'])
        self.assertIs(subprocess.DEVNULL, kwargs['stdin'])
        self.assertIs(subprocess.DEVNULL, kwargs['stdout'])
        self.assertTrue(kwargs['start_new_session'])
        self.assertNotIn('shell', kwargs)
        self.assertEqual(0, self.processes[0].terminated)
        self.processes[0].finish(0)
        self.event(token, 'finished')
        self.assertFalse(self.player.is_playing())
        self.assertGreaterEqual(self.processes[0].waits, 1)

    def test_stop_is_idempotent_and_reaps_its_child(self):
        self.player.stop()
        token = self.player.start(self.track)
        self.event(token, 'started')
        self.player.stop()
        self.player.stop()
        self.event(token, 'stopped')
        self.assertEqual(1, self.processes[0].terminated)
        self.assertEqual(0, self.processes[0].killed)
        self.assertFalse(self.player.is_playing())
        self.assertGreaterEqual(self.processes[0].waits, 1)

    def test_stop_kills_a_child_ignoring_terminate_without_blocking_caller(self):
        self.next_process = FakeProcess(ignore_terminate=True)
        token = self.player.start(self.track)
        self.event(token, 'started')
        with patch('meme_sing.playback.STOP_GRACE', .02):
            started = time.monotonic()
            self.player.stop()
            self.assertLess(time.monotonic() - started, .1)
            self.event(token, 'stopped')
        self.assertEqual(1, self.processes[0].killed)
        self.assertEqual(-9, self.processes[0].returncode)

    def test_old_stopped_token_cannot_clear_new_song(self):
        self.next_process = FakeProcess(ignore_terminate=True)
        first = self.player.start(self.track)
        self.event(first, 'started')
        second = self.player.start(self.track)
        self.event(second, 'started')
        self.assertNotEqual(first, second)
        self.processes[0].finish(-15)
        old = self.event(first, 'stopped')
        self.assertEqual(first, old['token'])
        self.assertTrue(self.player.is_playing())
        self.assertEqual(0, self.processes[1].terminated)
        self.processes[1].finish()
        self.event(second, 'finished')
        self.assertFalse(self.player.is_playing())

    def test_owned_process_ids_include_stopping_child_but_exclude_exited_children(self):
        self.assertEqual([], self.player.process_ids())
        self.next_process = FakeProcess(ignore_terminate=True)
        first = self.player.start(self.track)
        self.event(first, 'started')
        second = self.player.start(self.track)
        self.event(second, 'started')
        old, current = self.processes
        self.assertEqual(1, old.terminated)
        self.assertEqual([old.pid, current.pid], self.player.process_ids())
        # The old token remains owned until reaped, but an already exited
        # child must not be returned even before its monitor removes it.
        with self.player._lock:
            old.finish(-15)
            self.assertEqual([current.pid], self.player.process_ids())
        self.event(first, 'stopped')
        self.assertTrue(self.player.is_playing())
        with self.player._lock:
            current.finish()
            self.assertEqual([], self.player.process_ids())
        self.event(second, 'finished')

    def test_nonzero_exit_reports_bounded_stderr_and_no_finished_event(self):
        self.next_process = FakeProcess(stderr=b'x' * 10000 + b' decoder failed')
        token = self.player.start(self.track)
        self.event(token, 'started')
        self.processes[0].finish(2)
        error = self.event(token, 'error')
        self.assertLessEqual(len(error['message']), STDERR_LIMIT)
        self.assertIn('退出码2', error['message'])
        self.assertIn('decoder failed', error['message'])
        self.assertFalse(self.player.is_playing())

    def test_launch_failure_returns_identified_error_without_child(self):
        with patch('meme_sing.playback.subprocess.Popen', side_effect=OSError('launch failed')):
            token = self.player.start(self.track)
        event = self.event(token, 'error')
        self.assertIn('launch failed', event['message'])
        self.assertEqual('music-1', event['track_id'])
        self.assertFalse(self.player.is_playing())

    def test_missing_ffplay_is_an_error_event(self):
        self.player = MusicPlayer()
        self.addCleanup(self.player.close)
        with patch('meme_sing.playback.shutil.which', return_value=None):
            token = self.player.start(self.track)
        self.assertIn('缺少ffplay', self.event(token, 'error')['message'])
        self.assertEqual([], self.calls)

    def test_invalid_missing_or_symlink_track_cannot_stop_a_valid_song(self):
        token = self.player.start(self.track)
        self.event(token, 'started')
        missing = self.path.parent / 'missing.mp3'
        bad = Track('bad', 'missing', missing.name, missing, {})
        error_token = self.player.start(bad)
        self.event(error_token, 'error')
        self.assertTrue(self.player.is_playing())
        self.assertEqual(0, self.processes[0].terminated)
        link = self.path.parent / 'link.mp3'
        link.symlink_to(self.path)
        error_token = self.player.start(Track('bad', 'link', link.name, link, {}))
        self.event(error_token, 'error')
        self.assertEqual(1, len(self.calls))

    def test_close_stops_all_owned_processes_and_refuses_new_start(self):
        first = self.player.start(self.track)
        self.event(first, 'started')
        self.player.close()
        self.event(first, 'stopped')
        self.assertFalse(self.player.is_playing())
        self.assertEqual(0, self.processes[0].killed)
        token = self.player.start(self.track)
        self.assertIn('已关闭', self.event(token, 'error')['message'])
        self.assertEqual(1, len(self.calls))
        self.player.close()

    def test_invalid_volume_rejected(self):
        for value in (-1, 101, True, 1.5, '30'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                MusicPlayer(volume=value)

    def test_explicit_usb_sink_overrides_ambient_sink_without_changing_parent(self):
        sink = 'alsa_output.usb-UACDemoV1.0-00.analog-stereo'
        self.player = MusicPlayer(executable='/fake/ffplay', sink_name=sink)
        self.addCleanup(self.player.close)
        with patch.dict(os.environ, {'PULSE_SINK': 'board-output'}):
            token = self.player.start(self.track)
            self.event(token, 'started')
            self.assertEqual('board-output', os.environ['PULSE_SINK'])
        arguments, kwargs = self.calls[0]
        self.assertEqual(sink, kwargs['env']['PULSE_SINK'])
        self.assertEqual('pulseaudio', kwargs['env']['SDL_AUDIODRIVER'])
        self.assertNotIn(sink, arguments)
        self.assertNotIn('shell', kwargs)
        self.processes[0].finish()
        self.event(token, 'finished')

    def test_invalid_sink_names_fail_before_a_child_can_start(self):
        for sink in ('', ' ', 'sink\nother', 'sink\0', '$(echo bad)', '/a/path', 1, True):
            with self.subTest(sink=sink), self.assertRaises(ValueError):
                MusicPlayer(executable='/fake/ffplay', sink_name=sink)
        self.assertEqual([], self.calls)

    def test_real_spawn_failure_with_selected_sink_returns_error_event(self):
        self.player = MusicPlayer(executable=self.path.parent / 'missing-ffplay',
                                  sink_name='alsa_output.usb-UACDemoV1.0-00.analog-stereo')
        self.addCleanup(self.player.close)
        with patch('meme_sing.playback.subprocess.Popen', REAL_POPEN):
            token = self.player.start(self.track)
        event = self.event(token, 'error')
        self.assertIn('missing-ffplay', event['message'])
        self.assertFalse(self.player.is_playing())
        self.assertEqual([], self.calls)


if __name__ == '__main__':
    unittest.main()
