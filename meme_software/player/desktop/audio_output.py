"""Restore the named USB speaker after hotplug without blocking the UI.

Only this process's audio streams are moved.  ALSA card numbers and PulseAudio
sink numbers are deliberately not identities: both can change after reconnect.
"""
from pathlib import Path
import json
import re
import subprocess
import threading
import time


DEFAULT_USB_SINK = (
    'alsa_output.usb-Jieli_Technology_UACDemoV1.0_4150354735353004-00.analog-stereo'
)
DEFAULT_USB_CARD = 'UACDemoV10'


def configured_playback_available(card=DEFAULT_USB_CARD, *,
                                  proc_root=Path('/proc'), dev_root=Path('/dev')):
    """Resolve a stable ALSA card name to its current physical playback node."""
    if not isinstance(card, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', card):
        return False
    try:
        cards = (Path(proc_root) / 'asound' / 'cards').read_text(encoding='utf-8')
        for line in cards.splitlines():
            parsed = re.match(r'^\s*(\d+)\s+\[([^\]]+)\]\s*:', line)
            if parsed and parsed.group(2).strip() == card:
                sound = Path(dev_root) / 'snd'
                return any(re.fullmatch(r'pcmC' + parsed.group(1) + r'D\d+p', path.name)
                           and path.exists() for path in sound.glob('pcmC*D*p'))
        return False
    except (OSError, UnicodeError):
        return False


class _Cancelled(Exception):
    pass


class _CommandError(Exception):
    pass


class UsbAudioOutput:
    """One background watcher; public methods only take a short memory lock.

    ``get_owned_pids`` runs on the watcher and must not access Tk.  It may return
    the application and its audio children; other applications are never moved.
    A present ALSA device whose sink is still absent gets one WirePlumber restart
    per disappearance episode, after a grace period for normal enumeration.
    """

    def __init__(self, sink_name=DEFAULT_USB_SINK, get_owned_pids=None, *, can_repair=None,
                 playback_card=DEFAULT_USB_CARD, playback_available=None,
                 command_runner=subprocess.run, clock=time.monotonic,
                 poll_seconds=1., command_timeout=2., enumeration_grace=2.):
        if (not isinstance(sink_name, str) or len(sink_name) > 256
                or not re.fullmatch(r'[A-Za-z0-9_.:-]+', sink_name)):
            raise ValueError('Invalid audio sink name')
        if poll_seconds <= 0 or command_timeout <= 0 or enumeration_grace < 0:
            raise ValueError('Invalid audio recovery timing')
        self.sink_name = sink_name
        self._owned_pids = get_owned_pids or (lambda: ())
        self._can_repair = can_repair or (lambda: True)
        self._playback_available = playback_available or (
            lambda: configured_playback_available(playback_card))
        self._runner = command_runner
        self._clock = clock
        self._poll_seconds = float(poll_seconds)
        self._timeout = float(command_timeout)
        self._enumeration_grace = float(enumeration_grace)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._generation = 0
        self._restore_requested = True
        self._thread = None
        self._state = 'inactive'
        self._error = ''
        self._is_ready = False
        # Only the watcher changes episode bookkeeping.
        self._sink_present = None
        self._physical_present = None
        self._missing_since = None
        self._repair_used = False
        self._last_volume = None
        self._failures = 0

    def start(self):
        with self._lock:
            if self._closed or self._thread is not None:
                return False
            self._state = 'checking'
            self._thread = threading.Thread(target=self._watch, name='usb-audio-output', daemon=True)
            self._thread.start()
        return True

    def request_recovery(self):
        """Wake the watcher without granting another service restart budget."""
        with self._lock:
            if self._closed:
                return False
            self._generation += 1
            self._restore_requested = True
            self._state = 'checking'
            self._is_ready = False
            self._error = ''
        self._wake.set()
        return True

    def ready(self):
        with self._lock:
            return self._is_ready and not self._closed

    def status(self):
        with self._lock:
            return {'state': self._state, 'sink_name': self.sink_name,
                    'error': self._error, 'ready': self._is_ready and not self._closed}

    def close(self):
        """Cancel immediately; an already running command has its own timeout."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            self._is_ready = False
            self._state = 'closed'
            self._error = ''
        self._wake.set()

    def _valid(self, generation):
        with self._lock:
            return not self._closed and generation == self._generation

    def _publish(self, generation, state, error='', ready=False):
        with self._lock:
            if self._closed or generation != self._generation:
                raise _Cancelled()
            self._state = state
            self._error = error
            self._is_ready = ready

    def _run(self, generation, command):
        if not self._valid(generation):
            raise _Cancelled()
        try:
            result = self._runner(command, capture_output=True, text=True,
                                  timeout=self._timeout, check=False)
        except subprocess.TimeoutExpired:
            raise _CommandError('command_timeout') from None
        except OSError:
            raise _CommandError('command_unavailable') from None
        if not self._valid(generation):
            raise _Cancelled()
        if result.returncode:
            raise _CommandError('command_failed')
        output = result.stdout or ''
        if len(output) > 1024 * 1024:
            raise _CommandError('invalid_audio_service_response')
        return output

    def _json(self, generation, kind):
        output = self._run(generation, ['pactl', '-f', 'json', 'list', kind])
        try:
            data = json.loads(output)
        except (TypeError, ValueError):
            raise _CommandError('invalid_audio_service_response') from None
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise _CommandError('invalid_audio_service_response')
        return data

    def _remember_or_restore_volume(self, generation, sink, *, restore):
        volume = sink.get('volume')
        values = []
        if isinstance(volume, dict):
            for channel in volume.values():
                value = channel.get('value') if isinstance(channel, dict) else None
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    values.append(value)
        if values and max(values) > 0:
            self._last_volume = min(65536, max(values))
        elif values and restore:
            restored = self._last_volume or 19661  # 30% only when every channel is zero.
            self._run(generation, ['pactl', 'set-sink-volume', self.sink_name, str(restored)])
        if restore and sink.get('mute') is True:
            self._run(generation, ['pactl', 'set-sink-mute', self.sink_name, '0'])

    def _move_owned_inputs(self, generation, sink):
        try:
            owned = {str(pid) for pid in self._owned_pids()
                     if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0}
        except Exception:
            raise _CommandError('audio_owner_probe_failed') from None
        if not owned:
            return
        for stream in self._json(generation, 'sink-inputs'):
            properties = stream.get('properties')
            pid = properties.get('application.process.id') if isinstance(properties, dict) else None
            if str(pid) not in owned:
                continue
            if str(stream.get('sink')) in {str(sink.get('index')), self.sink_name}:
                continue
            index = stream.get('index')
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                continue
            try:
                self._run(generation, ['pactl', 'move-sink-input', str(index), self.sink_name])
            except _CommandError as error:
                if str(error) != 'command_failed':
                    raise
                # Short meme clips can finish between enumeration and this move.
                # The next probe retries a surviving stream; the output route is
                # already correct, so this alone must not suspend capture.

    def _check_once(self, generation=None):
        """One bounded probe, also used by deterministic device-free tests."""
        if generation is None:
            with self._lock:
                generation = self._generation
        if not self._valid(generation):
            raise _Cancelled()
        sinks = self._json(generation, 'sinks')
        sink = next((item for item in sinks if item.get('name') == self.sink_name), None)
        physical = self._playback_available() is True
        if not self._valid(generation):
            raise _Cancelled()
        now = self._clock()
        if sink is not None and physical:
            reappeared = self._sink_present is not True or self._physical_present is not True
            self._sink_present = True
            self._physical_present = True
            self._missing_since = None
            self._repair_used = False
            default = self._run(generation, ['pactl', 'get-default-sink']).strip()
            with self._lock:
                if self._closed or generation != self._generation:
                    raise _Cancelled()
                restore = self._restore_requested or reappeared or default != self.sink_name
            if default != self.sink_name:
                self._run(generation, ['pactl', 'set-default-sink', self.sink_name])
            self._remember_or_restore_volume(generation, sink, restore=restore)
            self._move_owned_inputs(generation, sink)
            with self._lock:
                if self._closed or generation != self._generation:
                    raise _Cancelled()
                self._restore_requested = False
            self._publish(generation, 'ready', ready=True)
            self._failures = 0
            return self._poll_seconds
        if self._sink_present is True or self._physical_present != physical:
            self._repair_used = False
            self._missing_since = now if physical else None
        self._sink_present = False
        self._physical_present = physical
        if not physical:
            self._missing_since = None
            self._publish(generation, 'waiting_device', 'usb_playback_unavailable')
        else:
            if self._missing_since is None:
                self._missing_since = now
            if not self._repair_used and now - self._missing_since >= self._enumeration_grace:
                try:
                    permitted = self._can_repair() is True
                except Exception:
                    raise _CommandError('audio_repair_gate_failed') from None
                if not permitted:
                    self._publish(generation, 'waiting_service', 'audio_devices_busy')
                    self._failures = 0
                    return self._poll_seconds
                self._repair_used = True  # Timeouts and errors also spend this episode's attempt.
                self._publish(generation, 'repairing', 'usb_sink_not_enumerated')
                self._run(generation, ['systemctl', '--user', 'restart', 'wireplumber'])
            else:
                self._publish(generation, 'waiting_sink', 'usb_sink_not_enumerated')
        self._failures = 0
        return self._poll_seconds

    def _watch(self):
        while True:
            with self._lock:
                if self._closed:
                    return
                generation = self._generation
                self._wake.clear()
            delay = self._poll_seconds
            try:
                delay = self._check_once(generation)
            except _Cancelled:
                continue
            except _CommandError as error:
                self._failures += 1
                delay = min(8., self._poll_seconds * (2 ** min(self._failures - 1, 3)))
                try:
                    self._publish(generation, 'error', str(error))
                except _Cancelled:
                    continue
            except Exception:
                # Keep one watcher alive; never respawn daemon processes on a loop.
                try:
                    self._publish(generation, 'error', 'audio_output_probe_failed')
                except _Cancelled:
                    continue
                delay = max(1., min(8., self._poll_seconds * 4))
            self._wake.wait(delay)
