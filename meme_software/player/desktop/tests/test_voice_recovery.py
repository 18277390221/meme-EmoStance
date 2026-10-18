"""Retry timing and ALSA reconnect checks without devices, subprocesses or sleep."""
from pathlib import Path
import sys
import tempfile
import unittest

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
from voice_recovery import VoiceRecovery, configured_capture_available


ERROR = {'type': 'error', 'reason': 'audio_device_unavailable', 'recoverable': True}


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.recovery = VoiceRecovery(clock=self.clock)

    def next_ticket(self, delay=1):
        self.clock.advance(delay)
        ticket = self.recovery.poll(device_available=None)
        self.assertIsNotNone(ticket)
        self.assertTrue(self.recovery.attempt_started(ticket))
        return ticket

    def test_unknown_alias_can_try_real_worker_after_backoff(self):
        self.assertTrue(self.recovery.fail(ERROR))
        self.clock.advance(.99)
        self.assertIsNone(self.recovery.poll())
        self.clock.advance(.01)
        ticket = self.recovery.poll()
        self.assertEqual(0, self.recovery.status()['attempts'])
        self.assertTrue(self.recovery.attempt_started(ticket))
        self.assertEqual('recovering', self.recovery.status()['state'])
        self.assertEqual(1, self.recovery.status()['attempts'])

    def test_retry_budget_is_1_2_4_8_8_and_exhaustion_stays_visible(self):
        self.recovery.fail(ERROR)
        for count, delay in enumerate((1, 2, 4, 8, 8), 1):
            self.next_ticket(delay)
            self.assertEqual(count, self.recovery.status()['attempts'])
            self.assertEqual(count < 5, self.recovery.fail(ERROR))
        self.assertEqual('exhausted', self.recovery.status()['state'])
        self.assertEqual('audio_device_unavailable', self.recovery.status()['last_error'])
        self.clock.advance(10000)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.assertFalse(self.recovery.fail(ERROR))

    def test_long_unplug_does_not_consume_any_worker_attempts(self):
        self.recovery.fail(ERROR)
        expected_delays = (1, 2, 4, 8, 8, 8)
        self.clock.advance(1)
        for delay in expected_delays:
            self.assertIsNone(self.recovery.poll(device_available=False))
            self.assertEqual(delay, self.recovery.status()['retry_in_seconds'])
            self.assertEqual(0, self.recovery.status()['attempts'])
            self.clock.advance(delay)
        self.clock.advance(3600)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.assertEqual('device_stabilizing', self.recovery.status()['state'])
        self.clock.advance(1)
        ticket = self.recovery.poll(device_available=True)
        self.assertTrue(self.recovery.attempt_started(ticket))
        self.assertEqual(1, self.recovery.status()['attempts'])

    def test_device_must_be_stable_and_unplug_resets_stability(self):
        self.recovery.fail(ERROR)
        self.clock.advance(1)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(1)
        self.assertIsNone(self.recovery.poll(device_available=False))
        self.clock.advance(1)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(.99)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(.01)
        self.assertIsNotNone(self.recovery.poll(device_available=True))

    def test_brief_disconnect_during_stabilization_is_not_missed(self):
        self.recovery.fail(ERROR)
        self.clock.advance(1)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(.5)
        self.assertIsNone(self.recovery.poll(device_available=False))
        self.clock.advance(.5)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(.5)
        self.assertIsNone(self.recovery.poll(device_available=True))
        self.clock.advance(1)
        self.assertIsNotNone(self.recovery.poll(device_available=True))

    def test_old_worker_music_and_restart_block_without_consuming_budget(self):
        for blocker in ('worker_running', 'music_active', 'restarting'):
            self.recovery.cancel()
            self.recovery.fail(ERROR)
            self.clock.advance(10)
            self.assertIsNone(self.recovery.poll(**{blocker: True}))
            self.assertEqual(0, self.recovery.status()['attempts'])
            self.assertIsNotNone(self.recovery.poll())

    def test_ticket_is_returned_once_and_can_only_start_once(self):
        self.recovery.fail(ERROR)
        self.clock.advance(1)
        ticket = self.recovery.poll()
        self.assertIsNone(self.recovery.poll())
        self.assertTrue(self.recovery.attempt_started(ticket))
        self.assertFalse(self.recovery.attempt_started(ticket))
        self.assertIsNone(self.recovery.poll())

    def test_manual_pause_cancels_returned_ticket_and_old_success(self):
        self.recovery.fail(ERROR)
        self.clock.advance(1)
        ticket = self.recovery.poll()
        self.recovery.cancel()
        self.assertFalse(self.recovery.attempt_started(ticket))
        self.assertFalse(self.recovery.recovered(ticket))
        self.assertIsNone(self.recovery.poll())

    def test_old_generation_cannot_start_or_clear_a_new_attempt(self):
        self.recovery.fail(ERROR)
        old = self.next_ticket()
        self.recovery.cancel()
        self.recovery.fail(ERROR)
        new = self.next_ticket()
        self.assertNotEqual(old, new)
        self.assertFalse(self.recovery.attempt_started(old))
        self.assertFalse(self.recovery.recovered(old))
        self.assertEqual(1, self.recovery.status()['attempts'])
        self.assertTrue(self.recovery.recovered(new))

    def test_duplicate_failure_does_not_postpone_retry_or_reset_budget(self):
        self.recovery.fail(ERROR)
        self.next_ticket()
        self.recovery.fail(ERROR)
        self.clock.advance(1)
        self.recovery.fail(ERROR)
        self.assertEqual(1, self.recovery.status()['attempts'])
        self.assertEqual(1, self.recovery.status()['retry_in_seconds'])
        self.clock.advance(1)
        self.assertIsNotNone(self.recovery.poll())

    def test_only_listening_acknowledgement_resets_attempt_budget(self):
        self.recovery.fail(ERROR)
        self.next_ticket()
        # A running/loading worker gets no success credit merely by existing.
        self.clock.advance(100)
        self.assertIsNone(self.recovery.poll(worker_running=True))
        self.assertEqual(1, self.recovery.status()['attempts'])
        self.assertTrue(self.recovery.recovered())
        self.assertEqual('idle', self.recovery.status()['state'])
        self.recovery.fail(ERROR)
        self.assertEqual(0, self.recovery.status()['attempts'])
        self.assertEqual(1, self.recovery.status()['retry_in_seconds'])

    def test_reason_and_boolean_are_both_required(self):
        for event in ({'recoverable': True, 'reason': 'runtime_error'},
                      {'recoverable': 1, 'reason': 'audio_device_unavailable'},
                      {'recoverable': True}, {'reason': 'audio_device_unavailable'}, None):
            with self.subTest(event=event):
                self.recovery.fail(ERROR)
                self.assertFalse(self.recovery.fail(event))
                self.clock.advance(10)
                self.assertIsNone(self.recovery.poll())
                self.assertEqual('error', self.recovery.status()['state'])

    def test_nonrecoverable_error_cancels_an_inflight_ticket(self):
        self.recovery.fail(ERROR)
        ticket = self.next_ticket()
        self.assertFalse(self.recovery.fail({'reason': 'permission_denied', 'recoverable': False}))
        self.assertFalse(self.recovery.recovered(ticket))
        self.assertEqual('error', self.recovery.status()['state'])

    def test_late_device_error_cannot_override_terminal_nonrecoverable_error(self):
        self.recovery.fail({'reason': 'model_invalid', 'recoverable': False})
        self.assertFalse(self.recovery.fail(ERROR))
        self.clock.advance(100)
        self.assertIsNone(self.recovery.poll())
        self.assertEqual('error', self.recovery.status()['state'])
        self.assertEqual('nonrecoverable_error', self.recovery.status()['last_error'])
        self.recovery.cancel()  # Explicitly enabling voice begins a fresh episode.
        self.assertTrue(self.recovery.fail(ERROR))
        self.assertEqual('backoff', self.recovery.status()['state'])

    def test_status_has_only_bounded_diagnostics(self):
        self.recovery.fail(dict(ERROR, message='private audio device path and recognized words'))
        self.assertEqual({'state', 'attempts', 'max_attempts', 'last_error', 'retry_in_seconds'},
                         set(self.recovery.status()))
        self.assertNotIn('private', str(self.recovery.status()))
        self.clock.advance(100)
        self.assertEqual(0, self.recovery.status()['retry_in_seconds'])

    def test_invalid_configuration_rejected(self):
        for kwargs in ({'delays': ()}, {'delays': (-1,)}, {'stable_seconds': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VoiceRecovery(**kwargs)


class CaptureAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / 'proc'
        self.dev = self.root / 'dev'
        (self.proc / 'asound').mkdir(parents=True)
        (self.dev / 'snd').mkdir(parents=True)
        self.cards = self.proc / 'asound' / 'cards'
        self.cards.write_text(' 0 [Headphones     ]: bcm2835_headpho - Headphones\n'
                              ' 2 [UACDemoV10     ]: USB-Audio - UACDemoV1.0\n')

    def check(self, device):
        return configured_capture_available(device, self.proc, self.dev)

    def capture(self, card, pcm=0):
        path = self.dev / 'snd' / f'pcmC{card}D{pcm}c'
        path.touch()
        return path

    def test_named_card_re_resolves_changed_usb_card_number(self):
        node = self.capture(2)
        self.assertTrue(self.check('plughw:CARD=UACDemoV10,DEV=0'))
        node.unlink()
        self.cards.write_text(' 3 [UACDemoV10 ]: USB-Audio - Reconnected\n')
        self.capture(3)
        self.assertTrue(self.check('plughw:CARD=UACDemoV10,DEV=0'))

    def test_numeric_named_and_plug_prefix_formats(self):
        self.capture(2, 1)
        for name in ('hw:2,1', 'plughw:2,1', 'plug:hw:2,1',
                     'plug:plughw:CARD=UACDemoV10,DEV=1', 'hw:UACDemoV10,1',
                     'hw:CARD=2,DEV=1'):
            with self.subTest(name=name):
                self.assertTrue(self.check(name))
        self.assertFalse(self.check('hw:2,0'))

    def test_absent_named_card_is_false_and_does_not_choose_another_capture(self):
        self.capture(0)
        self.assertFalse(self.check('plughw:CARD=Missing,DEV=0'))
        self.assertFalse(self.check('plughw:CARD=UACDemoV10,DEV=0'))

    def test_default_only_accepts_capture_node_not_playback_or_control(self):
        (self.dev / 'snd' / 'pcmC2D0p').touch()
        (self.dev / 'snd' / 'controlC2').touch()
        self.assertFalse(self.check('default'))
        self.capture(2)
        for alias in ('default', 'sysdefault', 'pulse', 'pipewire'):
            with self.subTest(alias=alias):
                self.assertTrue(self.check(alias))

    def test_unknown_or_malformed_alias_delegates_to_actual_worker(self):
        for device in ('my_custom_capture', '', None, 'hw:', 'hw:2,-1',
                       'hw:CARD=UACDemoV10,DEV=no', 'hw:2,0,1',
                       'hw:CARD=UACDemoV10,FOO=0'):
            with self.subTest(device=device):
                self.assertIsNone(self.check(device))

    def test_unreadable_card_map_is_unknown_but_numeric_device_can_be_checked(self):
        self.cards.unlink()
        self.capture(2)
        self.assertIsNone(self.check('plughw:CARD=UACDemoV10,DEV=0'))
        self.assertTrue(self.check('hw:2,0'))


if __name__ == '__main__':
    unittest.main()
