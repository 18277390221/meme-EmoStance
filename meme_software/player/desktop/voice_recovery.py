"""Nonblocking retry policy for an unplugged capture device.

The UI owns worker lifetimes. This module only issues revocable restart tickets;
creating a process is not success until its ``listening`` event is accepted.
"""
from dataclasses import dataclass
from pathlib import Path
import re
import time


RECOVERABLE_REASONS = frozenset({'audio_device_unavailable'})


def configured_capture_available(device, proc_root=Path('/proc'), dev_root=Path('/dev')):
    """Return True/False for known ALSA devices, None for an unknown alias.

    Resolve the configured card every time: USB reconnection can change ALSA's
    numeric card index. Reading these small files never opens the sound device.
    """
    if not isinstance(device, str):
        return None
    device = device.strip()
    while device.startswith('plug:'):
        device = device[5:]
    sound = Path(dev_root) / 'snd'
    if device in {'default', 'sysdefault', 'pulse', 'pipewire'}:
        try:
            return any(re.fullmatch(r'pcmC\d+D\d+c', path.name)
                       and path.exists() for path in sound.glob('pcmC*D*c'))
        except OSError:
            return None
    match = re.fullmatch(r'(?:plughw|hw):(.*)', device)
    if not match:
        return None
    fields = match.group(1).split(',')
    card = None
    pcm = 0
    if fields[0].startswith('CARD='):
        card = fields[0][5:]
        if len(fields) > 2 or (len(fields) == 2 and not fields[1].startswith('DEV=')):
            return None
        if len(fields) == 2:
            try:
                pcm = int(fields[1][4:])
            except ValueError:
                return None
    else:
        if len(fields) > 2:
            return None
        card = fields[0]
        if len(fields) == 2:
            try:
                pcm = int(fields[1])
            except ValueError:
                return None
    if pcm < 0 or not card or not re.fullmatch(r'[A-Za-z0-9_-]+', card):
        return None
    if card.isdecimal():
        index = int(card)
    else:
        try:
            cards = (Path(proc_root) / 'asound' / 'cards').read_text(encoding='utf-8')
        except (OSError, UnicodeError):
            return None
        index = None
        for line in cards.splitlines():
            parsed = re.match(r'^\s*(\d+)\s+\[([^\]]+)\]\s*:', line)
            if parsed and parsed.group(2).strip() == card:
                index = int(parsed.group(1))
                break
        if index is None:
            return False
    try:
        return (sound / f'pcmC{index}D{pcm}c').exists()
    except OSError:
        return None


@dataclass(frozen=True)
class RecoveryTicket:
    generation: int
    serial: int


class VoiceRecovery:
    """Bound actual worker retries without timing out an absent USB device."""

    def __init__(self, clock=time.monotonic, delays=(1, 2, 4, 8, 8), stable_seconds=1):
        if not delays or any(delay < 0 for delay in delays) or stable_seconds < 0:
            raise ValueError('恢复间隔需要为非负数，且至少允许一次重试。')
        self.clock = clock
        self.delays = tuple(float(delay) for delay in delays)
        self.stable_seconds = float(stable_seconds)
        self.generation = 0
        self.serial = 0
        self.state = 'idle'
        self.attempts = 0
        self.last_error = ''
        self.next_check = None
        self.available_since = None
        self.probe_failures = 0
        self.ticket = None

    def fail(self, event):
        """Schedule only a trusted hardware error; other failures cancel retries."""
        if (not isinstance(event, dict) or event.get('recoverable') is not True
                or event.get('reason') not in RECOVERABLE_REASONS):
            self.cancel()
            self.state = 'error'
            self.last_error = 'nonrecoverable_error'
            return False
        if self.state in {'error', 'exhausted'}:
            return False  # Only a manual cancel/enable can clear a terminal error.
        self.last_error = event['reason']
        if self.state in {'backoff', 'waiting_device', 'device_stabilizing', 'scheduled'}:
            return True  # Duplicate errors from one dying worker do not reset the budget.
        if self.state not in {'recovering', 'starting'}:
            self.generation += 1
            self.attempts = 0
        self.ticket = None
        self.available_since = None
        self.probe_failures = 0
        if self.attempts >= len(self.delays):
            self.state = 'exhausted'
            self.next_check = None
            return False
        self.state = 'backoff'
        self.next_check = self.clock() + self.delays[self.attempts]
        return True

    def poll(self, *, worker_running=False, music_active=False, restarting=False,
             device_available=None):
        """Return one ticket after old audio owners exit and the device stabilizes.

        False availability waits indefinitely without spending a worker attempt.
        None is an unknown ALSA alias and delegates the check to the real worker.
        """
        if self.state not in {'backoff', 'waiting_device', 'device_stabilizing'}:
            return None
        if worker_running or music_active or restarting:
            return None
        now = self.clock()
        # Polling may observe a brief disconnect during the stabilization window.
        # Do not ignore it just because the next scheduled probe is still ahead.
        if device_available is False and self.available_since is not None:
            self.available_since = None
            self.state = 'waiting_device'
            self.next_check = max(self.next_check or now, now + 1.)
        if self.next_check is not None and now < self.next_check:
            return None
        if device_available is False:
            self.available_since = None
            self.state = 'waiting_device'
            delay = (1, 2, 4, 8)[min(self.probe_failures, 3)]
            self.probe_failures += 1
            self.next_check = now + delay
            return None
        if device_available is True:
            if self.available_since is None:
                self.available_since = now
            remaining = self.stable_seconds - (now - self.available_since)
            if remaining > 0:
                self.state = 'device_stabilizing'
                self.next_check = now + remaining
                return None
        else:
            self.available_since = None
        self.serial += 1
        self.ticket = RecoveryTicket(self.generation, self.serial)
        self.state = 'scheduled'
        self.next_check = None
        return self.ticket

    def attempt_started(self, ticket):
        """Atomically accept the ticket immediately before creating a worker."""
        if self.state != 'scheduled' or ticket is None or ticket != self.ticket:
            return False
        self.attempts += 1
        self.state = 'recovering'
        return True

    def recovered(self, ticket=None):
        """Call only for the current worker's real ``listening`` acknowledgement."""
        if self.state != 'recovering' or (ticket is not None and ticket != self.ticket):
            return False
        self.cancel()
        return True

    def cancel(self):
        """Invalidate every pending restart, including a ticket already handed out."""
        self.generation += 1
        self.state = 'idle'
        self.attempts = 0
        self.last_error = ''
        self.next_check = None
        self.available_since = None
        self.probe_failures = 0
        self.ticket = None

    def status(self):
        """Small diagnostic fields only; no audio, transcripts or device paths."""
        remaining = None if self.next_check is None else max(0., self.next_check - self.clock())
        return {'state': self.state, 'attempts': self.attempts,
                'max_attempts': len(self.delays), 'last_error': self.last_error,
                'retry_in_seconds': None if remaining is None else round(remaining, 2)}
