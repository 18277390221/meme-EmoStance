"""USB output recovery without audio hardware, services, or real subprocesses."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import threading
import unittest

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
from audio_output import (DEFAULT_USB_SINK, UsbAudioOutput, _Cancelled, _CommandError,
                          configured_playback_available)


def sink(index=9, *, name=DEFAULT_USB_SINK, volume=32768, mute=False):
    return {'index': index, 'name': name, 'mute': mute,
            'volume': {'front-left': {'value': volume}, 'front-right': {'value': volume}}}


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class AudioService:
    def __init__(self):
        self.sinks = [sink()]
        self.default = 'alsa_output.platform-board.analog-stereo'
        self.streams = []
        self.calls = []
        self.failure = None
        self.hook = None

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if self.hook:
            self.hook(args)
        if self.failure:
            raise self.failure
        if args == ['pactl', '-f', 'json', 'list', 'sinks']:
            output = json.dumps(self.sinks)
        elif args == ['pactl', '-f', 'json', 'list', 'sink-inputs']:
            output = json.dumps(self.streams)
        elif args == ['pactl', 'get-default-sink']:
            output = self.default + '\n'
        elif args[:2] == ['pactl', 'set-default-sink']:
            self.default = args[2]
            output = ''
        else:
            output = ''
        return subprocess.CompletedProcess(args, 0, output, '')

    def commands(self, prefix):
        return [args for args, _ in self.calls if args[:len(prefix)] == prefix]


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.service = AudioService()
        self.physical = True
        self.router = UsbAudioOutput(command_runner=self.service, clock=self.clock,
                                     playback_available=lambda: self.physical)
        self.addCleanup(self.router.close)

    def test_routes_exact_stable_name_among_multiple_sinks(self):
        self.service.sinks.insert(0, sink(1, name='alsa_output.usb-other.analog-stereo'))
        self.router._check_once()
        self.assertEqual(DEFAULT_USB_SINK, self.service.default)
        self.assertTrue(self.router.ready())
        self.assertEqual('ready', self.router.status()['state'])
        self.assertEqual('', self.router.status()['error'])

    def test_sink_number_changes_after_reconnection(self):
        self.router._check_once()
        self.service.sinks = []
        self.physical = False
        self.router._check_once()
        self.assertFalse(self.router.ready())
        self.service.sinks = [sink(47)]
        self.service.default = 'board'
        self.physical = True
        self.router._check_once()
        self.assertTrue(self.router.ready())
        self.assertEqual(DEFAULT_USB_SINK, self.service.default)
        self.assertEqual([], self.service.commands(['systemctl']))

    def test_matching_sink_is_not_ready_when_physical_device_is_absent(self):
        self.physical = False
        self.router._check_once()
        self.assertFalse(self.router.ready())
        self.assertEqual('waiting_device', self.router.status()['state'])
        self.assertEqual([], self.service.commands(['pactl', 'set-default-sink']))

    def test_no_wireplumber_restart_without_physical_usb_playback(self):
        self.service.sinks = []
        self.physical = False
        for now in (0, 3, 100, 3600):
            self.clock.now = now
            self.router._check_once()
        self.assertEqual([], self.service.commands(['systemctl']))

    def test_waits_for_enumeration_then_repairs_only_once_per_episode(self):
        self.service.sinks = []
        for now in (0, 1, 1.99):
            self.clock.now = now
            self.router._check_once()
        self.assertEqual([], self.service.commands(['systemctl']))
        for now in (2, 3, 8, 100):
            self.clock.now = now
            self.router._check_once()
        self.assertEqual([['systemctl', '--user', 'restart', 'wireplumber']],
                         self.service.commands(['systemctl']))
        self.router.request_recovery()
        self.router._check_once()
        self.assertEqual(1, len(self.service.commands(['systemctl'])))

    def test_new_physical_detachment_grants_one_new_repair(self):
        self.service.sinks = []
        self.router._check_once()
        self.clock.now = 2
        self.router._check_once()
        self.physical = False
        self.router._check_once()
        self.physical = True
        self.clock.now = 10
        self.router._check_once()
        self.clock.now = 12
        self.router._check_once()
        self.assertEqual(2, len(self.service.commands(['systemctl'])))

    def test_busy_audio_waits_without_spending_repair_budget(self):
        self.service.sinks = []
        self.router._can_repair = lambda: False
        self.router._check_once()
        self.clock.now = 20
        self.router._check_once()
        self.assertEqual('waiting_service', self.router.status()['state'])
        self.assertEqual([], self.service.commands(['systemctl']))
        self.router._can_repair = lambda: True
        self.router._check_once()
        self.assertEqual(1, len(self.service.commands(['systemctl'])))

    def test_present_target_is_ready_without_waiting_for_repair_gate(self):
        self.router._can_repair = lambda: self.fail('Gate is only relevant to a service restart')
        self.router._check_once()
        self.assertTrue(self.router.ready())

    def test_gate_exception_does_not_spend_repair_budget(self):
        self.service.sinks = []
        self.router._check_once()
        self.clock.now = 2

        def broken_gate():
            raise RuntimeError('private error')

        self.router._can_repair = broken_gate
        with self.assertRaisesRegex(_CommandError, '^audio_repair_gate_failed$'):
            self.router._check_once()
        self.router._can_repair = lambda: True
        self.router._check_once()
        self.assertEqual(1, len(self.service.commands(['systemctl'])))

    def test_sink_success_then_disappearance_grants_one_new_repair(self):
        self.service.sinks = []
        self.router._check_once()
        self.clock.now = 2
        self.router._check_once()
        self.service.sinks = [sink()]
        self.router._check_once()
        self.service.sinks = []
        self.clock.now = 3
        self.router._check_once()
        self.clock.now = 5
        self.router._check_once()
        self.assertEqual(2, len(self.service.commands(['systemctl'])))

    def test_mute_is_cleared_without_changing_nonzero_volume(self):
        self.service.sinks = [sink(volume=12000, mute=True)]
        self.router._check_once()
        self.assertEqual([], self.service.commands(['pactl', 'set-sink-volume']))
        self.assertEqual([['pactl', 'set-sink-mute', DEFAULT_USB_SINK, '0']],
                         self.service.commands(['pactl', 'set-sink-mute']))

    def test_zero_volume_restores_previous_nonzero_level(self):
        self.service.sinks = [sink(volume=23000)]
        self.router._check_once()
        self.service.sinks = [sink(volume=0)]
        self.router.request_recovery()
        self.router._check_once()
        self.assertEqual([['pactl', 'set-sink-volume', DEFAULT_USB_SINK, '23000']],
                         self.service.commands(['pactl', 'set-sink-volume']))

    def test_first_zero_volume_uses_small_default(self):
        self.service.sinks = [sink(volume=0)]
        self.router._check_once()
        self.assertEqual([['pactl', 'set-sink-volume', DEFAULT_USB_SINK, '19661']],
                         self.service.commands(['pactl', 'set-sink-volume']))

    def test_ready_monitor_preserves_user_mute_and_zero_volume(self):
        self.router._check_once()
        self.service.sinks = [sink(volume=0, mute=True)]
        for _ in range(3):
            self.router._check_once()
        self.assertTrue(self.router.ready())
        self.assertEqual([], self.service.commands(['pactl', 'set-sink-mute']))
        self.assertEqual([], self.service.commands(['pactl', 'set-sink-volume']))

    def test_actual_reconnection_unmutes_and_restores_last_nonzero_volume(self):
        self.service.sinks = [sink(volume=15000)]
        self.router._check_once()
        self.service.sinks = [sink(volume=26000)]
        self.router._check_once()
        self.service.sinks = []
        self.physical = False
        self.router._check_once()
        self.service.sinks = [sink(index=55, volume=0, mute=True)]
        self.physical = True
        self.router._check_once()
        self.assertEqual([['pactl', 'set-sink-volume', DEFAULT_USB_SINK, '26000']],
                         self.service.commands(['pactl', 'set-sink-volume']))
        self.assertEqual([['pactl', 'set-sink-mute', DEFAULT_USB_SINK, '0']],
                         self.service.commands(['pactl', 'set-sink-mute']))

    def test_switching_default_back_restores_audible_volume(self):
        self.router._check_once()
        self.service.default = 'board'
        self.service.sinks = [sink(volume=0, mute=True)]
        self.router._check_once()
        self.assertTrue(self.router.ready())
        self.assertEqual(1, len(self.service.commands(['pactl', 'set-sink-volume'])))
        self.assertEqual(1, len(self.service.commands(['pactl', 'set-sink-mute'])))

    def test_initial_restore_is_not_reapplied_each_poll(self):
        self.service.sinks = [sink(volume=0, mute=True)]
        for _ in range(3):
            self.router._check_once()
        self.assertEqual(1, len(self.service.commands(['pactl', 'set-sink-volume'])))
        self.assertEqual(1, len(self.service.commands(['pactl', 'set-sink-mute'])))

    def test_preserves_stereo_balance_if_any_channel_has_volume(self):
        item = sink(volume=0)
        item['volume']['front-right']['value'] = 6000
        self.service.sinks = [item]
        self.router._check_once()
        self.assertEqual([], self.service.commands(['pactl', 'set-sink-volume']))

    def test_moves_only_owned_audio_to_current_sink(self):
        self.router._owned_pids = lambda: (123, 124)
        self.service.streams = [
            {'index': 4, 'sink': 1, 'properties': {'application.process.id': '123'}},
            {'index': 5, 'sink': 1, 'properties': {'application.process.id': '999'}},
            {'index': 6, 'sink': 9, 'properties': {'application.process.id': '124'}},
            {'index': 7, 'sink': 1, 'properties': {}},
        ]
        self.router._check_once()
        self.assertEqual([['pactl', 'move-sink-input', '4', DEFAULT_USB_SINK]],
                         self.service.commands(['pactl', 'move-sink-input']))

    def test_stream_ending_during_move_does_not_invalidate_output_route(self):
        self.router._owned_pids = lambda: (123,)
        self.service.streams = [
            {'index': 4, 'sink': 1, 'properties': {'application.process.id': '123'}}]
        original = self.service.__call__

        def runner(args, **kwargs):
            result = original(args, **kwargs)
            if args[:2] == ['pactl', 'move-sink-input']:
                return subprocess.CompletedProcess(args, 1, '', 'No such entity')
            return result

        self.router._runner = runner
        self.router._check_once()
        self.assertTrue(self.router.ready())

    def test_skips_default_mutation_when_already_selected(self):
        self.service.default = DEFAULT_USB_SINK
        self.router._check_once()
        self.assertTrue(self.router.ready())
        self.assertEqual([], self.service.commands(['pactl', 'set-default-sink']))

    def test_failed_default_change_does_not_claim_ready(self):
        original = self.service.__call__

        def runner(args, **kwargs):
            if args[:2] == ['pactl', 'set-default-sink']:
                return subprocess.CompletedProcess(args, 1, '', 'private error')
            return original(args, **kwargs)

        self.router._runner = runner
        with self.assertRaisesRegex(_CommandError, 'command_failed'):
            self.router._check_once()
        self.assertFalse(self.router.ready())

    def test_every_command_has_timeout_and_no_shell(self):
        self.router._check_once()
        for args, kwargs in self.service.calls:
            self.assertIsInstance(args, list)
            self.assertEqual(2., kwargs['timeout'])
            self.assertFalse(kwargs.get('shell', False))
            self.assertTrue(kwargs['capture_output'])

    def test_close_cancels_after_inflight_read_before_mutation(self):
        self.service.hook = lambda args: self.router.close()
        with self.assertRaises(_Cancelled):
            self.router._check_once()
        self.assertEqual(1, len(self.service.calls))
        self.assertEqual('closed', self.router.status()['state'])
        self.assertFalse(self.router.ready())
        self.assertFalse(self.router.request_recovery())
        self.assertFalse(self.router.start())

    def test_new_generation_cancels_old_read_before_mutation(self):
        self.service.hook = lambda args: self.router.request_recovery()
        with self.assertRaises(_Cancelled):
            self.router._check_once()
        self.assertEqual(1, len(self.service.calls))
        self.assertEqual('checking', self.router.status()['state'])

    def test_timeout_is_small_sanitized_error(self):
        self.service.failure = subprocess.TimeoutExpired('pactl', 2, output='private')
        with self.assertRaisesRegex(_CommandError, '^command_timeout$'):
            self.router._check_once()
        self.assertFalse(self.router.ready())

    def test_failed_service_restart_spends_episode_budget(self):
        self.service.sinks = []
        original = self.service.__call__

        def runner(args, **kwargs):
            result = original(args, **kwargs)
            if args[0] == 'systemctl':
                raise subprocess.TimeoutExpired(args, 2)
            return result

        self.router._runner = runner
        self.router._check_once()
        self.clock.now = 2
        with self.assertRaises(_CommandError):
            self.router._check_once()
        self.clock.now = 30
        self.router._check_once()
        self.assertEqual(1, len(self.service.commands(['systemctl'])))

    def test_request_wakes_single_background_watcher(self):
        called = threading.Event()
        owner_threads = []
        self.router._owned_pids = lambda: owner_threads.append(threading.current_thread().name) or ()
        self.service.hook = lambda args: called.set() if args[:2] == ['pactl', 'get-default-sink'] else None
        self.assertTrue(self.router.start())
        self.assertFalse(self.router.start())
        self.assertTrue(called.wait(1))
        self.router.close()
        self.router._thread.join(timeout=1)
        self.assertFalse(self.router._thread.is_alive())
        self.assertTrue(all(name == 'usb-audio-output' for name in owner_threads))

    def test_watch_timeout_revokes_ready_and_does_not_restart_service(self):
        observed = threading.Event()
        snapshot = {}
        original_publish = self.router._publish
        original_run = self.service.__call__

        def publish(generation, state, error='', ready=False):
            original_publish(generation, state, error, ready)
            if state == 'ready':
                self.router.request_recovery()
            elif state == 'error':
                snapshot.update(self.router.status())
                self.router.close()
                observed.set()

        def run(args, **kwargs):
            if self.router._generation > 0:
                raise subprocess.TimeoutExpired(args, 2, output='private data')
            return original_run(args, **kwargs)

        self.router._runner = run
        self.router._publish = publish
        self.router.start()
        self.assertTrue(observed.wait(1))
        self.router._thread.join(timeout=1)
        self.assertFalse(snapshot['ready'])
        self.assertEqual('command_timeout', snapshot['error'])
        self.assertEqual([], self.service.commands(['systemctl']))

    def test_malformed_response_is_rejected_without_restarting_service(self):
        self.router._runner = lambda args, **kwargs: subprocess.CompletedProcess(args, 0, '{}', '')
        with self.assertRaisesRegex(_CommandError, '^invalid_audio_service_response$'):
            self.router._check_once()
        self.assertFalse(self.router.ready())

    def test_sink_name_is_validated(self):
        for name in ('', 'sink\nother', 'sink name', 'x; bad', None):
            with self.assertRaises(ValueError):
                UsbAudioOutput(name)


class PlaybackNodeTests(unittest.TestCase):
    def test_stable_card_identity_survives_numeric_alsa_card_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'proc/asound').mkdir(parents=True)
            (root / 'dev/snd').mkdir(parents=True)
            cards = root / 'proc/asound/cards'
            cards.write_text(' 2 [UACDemoV10    ]: USB-Audio - UACDemoV1.0\n')
            node = root / 'dev/snd/pcmC2D0p'
            node.touch()
            self.assertTrue(configured_playback_available(proc_root=root / 'proc', dev_root=root / 'dev'))
            node.unlink()
            cards.write_text(' 5 [UACDemoV10    ]: USB-Audio - UACDemoV1.0\n')
            (root / 'dev/snd/pcmC5D0p').touch()
            self.assertTrue(configured_playback_available(proc_root=root / 'proc', dev_root=root / 'dev'))

    def test_capture_node_or_other_usb_speaker_is_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'proc/asound').mkdir(parents=True)
            (root / 'dev/snd').mkdir(parents=True)
            (root / 'proc/asound/cards').write_text(' 2 [UACDemoV10 ]: USB-Audio - audio\n')
            (root / 'dev/snd/pcmC2D0c').touch()
            (root / 'dev/snd/pcmC3D0p').touch()
            self.assertFalse(configured_playback_available(proc_root=root / 'proc', dev_root=root / 'dev'))

    def test_missing_proc_files_wait_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(configured_playback_available(proc_root=Path(directory), dev_root=Path(directory)))


if __name__ == '__main__':
    unittest.main()
