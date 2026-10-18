"""Real GIF progression with fake Tk and audio, without a display or soundcard."""
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'meme_software/player/desktop'))
from meme_huanxing.screen import TkSleepScreen


class Widget:
    def __init__(self, *args, **kwargs):
        self.exists = True
        self.bindings = {}
        self.images = []
    def pack(self, **kwargs):
        pass
    def place(self, **kwargs):
        pass
    def lift(self):
        pass
    def bind(self, name, callback):
        self.bindings[name] = callback
    def focus_set(self):
        pass
    def configure(self, **kwargs):
        self.images.append((threading.get_ident(), kwargs))
    def winfo_width(self):
        return 32
    def winfo_height(self):
        return 24
    def winfo_exists(self):
        return self.exists
    def destroy(self):
        self.exists = False


class Parent:
    def __init__(self):
        self.jobs = []
        self.cancelled = []
    def after(self, delay, callback):
        self.jobs.append((delay, callback))
        return len(self.jobs)
    def after_cancel(self, job):
        self.cancelled.append(job)


class Audio:
    def __init__(self, muted=False, sink_name=None):
        self.muted, self.sink_name = muted, sink_name
        self.playing = False
        self.plays = []
        self.stops = 0
        self.errors = []
    def play(self, wav):
        self.plays.append(wav)
        self.playing = True
    def stop(self):
        self.stops += 1
        self.playing = False
    def is_playing(self):
        return self.playing
    def pop_errors(self):
        errors, self.errors = self.errors, []
        return errors


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.parent, self.completions, self.requests, self.errors = Parent(), [], [], []
        self.now = 100.0
        self.stack.enter_context(patch.dict(sys.modules, {
            'tkinter': SimpleNamespace(Frame=Widget, Label=Widget),
            'native_core': SimpleNamespace(Audio=Audio)}))
        self.stack.enter_context(patch('PIL.ImageTk.PhotoImage',
                                       side_effect=lambda image, **options: image))
        self.stack.enter_context(patch('meme_huanxing.screen.time.monotonic',
                                       side_effect=lambda: self.now))
        self.make_assets()
        self.screen = self.create_screen()
        self.addCleanup(self.screen.close)

    def make_assets(self, count=55):
        frames = [Image.new('RGB', (8, 8), (number * 4 % 255, number, 1))
                  for number in range(count)]
        frames[0].save(self.directory / 'sleep.gif', save_all=True,
                       append_images=frames[1:], duration=100, loop=0, optimize=False)
        (self.directory / 'sleep.wav').write_bytes(b'fake-wave')

    def create_screen(self, **options):
        return TkSleepScreen(self.parent, directory=self.directory,
            on_wake_request=lambda: self.requests.append('request'),
            on_wake_complete=lambda: self.completions.append('complete'),
            on_error=lambda error: self.errors.append(str(error)),
            sink_name='usb-output', **options)

    def await_frame(self, number=None):
        deadline = time.perf_counter() + 2
        while time.perf_counter() < deadline:
            self.screen.poll()
            if self.screen.closed:
                return False
            if self.screen.shown is not None and (
                    number is None or self.screen.shown[0] == number):
                return True
            # This is decoder scheduling, not the presentation clock.
            threading.Event().wait(.002)
        self.fail('decoder frame was not delivered')

    def next_frame(self):
        self.now = self.screen.deadline + .001
        self.screen.poll()
        if not self.screen.closed:
            self.await_frame()

    def test_sleep_loops_only_first_fifteen_frames(self):
        self.await_frame(0)
        observed = [0]
        for _ in range(18):
            self.next_frame()
            observed.append(self.screen.shown[0])
        self.assertEqual(list(range(15)) + [0, 1, 2, 3], observed)
        self.assertEqual([], self.completions)

    def test_touch_only_requests_wake_without_starting_animation(self):
        self.await_frame(0)
        self.assertEqual('break', self.screen.request_wake())
        self.assertEqual(['request'], self.requests)
        self.assertFalse(self.screen.waking)
        self.assertTrue(self.screen.audio.is_playing())

    def test_wake_plays_frames_fifteen_to_fiftyfour_once(self):
        self.await_frame(0)
        self.screen.wake()
        self.await_frame(15)
        observed = [15]
        for _ in range(39):
            self.next_frame()
            observed.append(self.screen.shown[0])
        self.assertEqual(list(range(15, 55)), observed)
        self.assertEqual([], self.completions)
        self.next_frame()
        self.assertTrue(self.screen.closed)
        self.assertEqual(['complete'], self.completions)

    def test_last_wake_frame_own_duration_precedes_completion(self):
        self.await_frame(0)
        self.screen.wake()
        self.await_frame(15)
        while self.screen.number < 54:
            self.next_frame()
        self.now = self.screen.deadline - .01
        self.screen.poll()
        self.assertFalse(self.screen.closed)
        self.assertEqual([], self.completions)
        self.now = self.screen.deadline + .001
        self.screen.poll()
        self.assertEqual(['complete'], self.completions)

    def test_wake_stops_loop_sound_and_never_replays_it(self):
        self.await_frame(0)
        self.assertEqual([b'fake-wave'], self.screen.audio.plays)
        self.screen.wake()
        self.assertFalse(self.screen.audio.playing)
        self.await_frame(15)
        self.next_frame()
        self.assertEqual([b'fake-wave'], self.screen.audio.plays)

    def test_sleep_wav_repeats_after_owned_process_finishes(self):
        self.await_frame(0)
        self.screen.audio.playing = False
        self.screen.poll()
        self.assertEqual([b'fake-wave', b'fake-wave'], self.screen.audio.plays)

    def test_audio_is_pinned_to_requested_usb_sink(self):
        self.assertEqual('usb-output', self.screen.audio.sink_name)

    def test_muted_screen_keeps_animation_without_starting_sound(self):
        self.screen.close()
        self.screen = self.create_screen(muted=True)
        self.addCleanup(self.screen.close)
        self.await_frame(0)
        self.assertEqual([], self.screen.audio.plays)
        self.screen.wake()
        self.await_frame(15)
        self.assertTrue(self.screen.waking)

    def test_duplicate_wake_does_not_restart_first_wake_frame(self):
        self.await_frame(0)
        self.screen.wake()
        self.await_frame(15)
        self.next_frame()
        self.screen.wake()
        self.assertEqual(16, self.screen.number)

    def test_missing_assets_fail_before_overlay_or_audio(self):
        self.screen.close()
        (self.directory / 'sleep.wav').unlink()
        with self.assertRaisesRegex(ValueError, 'sleep.wav'):
            self.create_screen()

    def test_corrupt_gif_errors_instead_of_pretending_wake_succeeded(self):
        self.screen.close()
        (self.directory / 'sleep.gif').write_bytes(b'corrupt')
        self.screen = self.create_screen()
        self.addCleanup(self.screen.close)
        self.await_frame()
        self.assertTrue(self.screen.closed)
        self.assertEqual(1, len(self.errors))
        self.assertEqual([], self.completions)

    def test_short_gif_errors_instead_of_skipping_required_wake_segment(self):
        self.screen.close()
        self.make_assets(count=15)
        self.screen = self.create_screen()
        self.addCleanup(self.screen.close)
        self.await_frame()
        self.assertTrue(self.screen.closed)
        self.assertIn('不足16帧', self.errors[0])
        self.assertEqual([], self.completions)

    def test_audio_failure_aborts_without_wake_completion(self):
        self.await_frame(0)
        self.screen.audio.errors.append('USB output disconnected')
        self.screen.poll()
        self.assertEqual(['USB output disconnected'], self.errors)
        self.assertTrue(self.screen.closed)
        self.assertEqual([], self.completions)

    def test_destroy_only_owns_the_actual_overlay(self):
        self.screen.destroyed(SimpleNamespace(widget=Widget()))
        self.assertFalse(self.screen.closed)
        self.screen.destroyed(SimpleNamespace(widget=self.screen.overlay))
        self.assertTrue(self.screen.closed)
        self.assertEqual([], self.completions)
        self.assertEqual(['待机画面意外关闭，已取消待机。'], self.errors)

    def test_unexpected_destroy_during_wake_reports_error_instead_of_hanging(self):
        self.await_frame(0)
        self.screen.wake()
        self.await_frame(15)
        self.screen.destroyed(SimpleNamespace(widget=self.screen.overlay))
        self.assertEqual(['待机画面意外关闭，已取消待机。'], self.errors)
        self.assertEqual([], self.completions)

    def test_owned_close_destroy_event_does_not_report_a_spurious_error(self):
        self.await_frame(0)
        self.screen.close()
        self.screen.destroyed(SimpleNamespace(widget=self.screen.overlay))
        self.assertEqual([], self.errors)
        self.assertEqual([], self.completions)

    def test_close_stops_worker_audio_and_cancels_scheduled_poll(self):
        self.await_frame(0)
        job = self.screen.job
        self.screen.close()
        self.screen.close()
        self.screen.worker.join(timeout=2)
        self.assertFalse(self.screen.worker.is_alive())
        self.assertIn(job, self.parent.cancelled)
        self.assertFalse(self.screen.audio.playing)
        self.assertEqual([], self.completions)

    def test_tk_image_updates_only_run_on_poll_thread(self):
        self.await_frame(0)
        self.next_frame()
        self.assertEqual({threading.get_ident()},
                         {thread for thread, _ in self.screen.image.images})


if __name__ == '__main__':
    unittest.main()
