#!/usr/bin/env python3
"""Owned ASR subprocess: local model/PCM -> JSON lines, never audio files/network."""
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import threading
import time
import wave

# Installed Pi package is next to this script; source checkouts share one copy.
if not (Path(__file__).resolve().parent / 'meme_sing').is_dir():
    repository = Path(__file__).resolve().parents[3]
    if (repository / 'meme_sing').is_dir():
        sys.path.insert(0, str(repository))
from meme_sing.core import StopCommandDetector, is_stop_request

if not (Path(__file__).resolve().parent / 'meme_huanxing').is_dir():
    repository = Path(__file__).resolve().parents[3]
    if (repository / 'meme_huanxing').is_dir() and str(repository) not in sys.path:
        sys.path.insert(0, str(repository))
from meme_huanxing.commands import is_sleep_command, is_wake_command

sys.path.insert(0, str(Path(__file__).resolve().parent / 'asr_runtime'))
from audio import Capture, UserError, capture_command, require_arecord
from speech import Decoder, MODEL_NAMES, create_recognizer
from capture_turn import MicrophoneGate, listen_once
from asr_selection import selection


EMIT_LOCK = threading.Lock()


class MusicASRControl:
    """Stop-only partial recognition. No lyrics leave the ASR subprocess."""
    def __init__(self, output, clock=time.monotonic, lock=None):
        self.output, self.clock = output, clock
        self.lock = lock or threading.RLock()
        self.enabled = threading.Event()
        self.token = None
        self.stopped = False
        self.last_sample = float('-inf')
        self.detector = StopCommandDetector(interval=.2, stable_hits=2)
        self.sleep_hits, self.last_sleep_hit = 0, None

    def command(self, enabled, token=None):
        with self.lock:
            self.token = token if enabled else None
            self.stopped = False
            self.last_sample = float('-inf')
            self.detector.reset()
            self.sleep_hits, self.last_sleep_hit = 0, None
            (self.enabled.set if enabled else self.enabled.clear)()

    def begin_window(self):
        with self.lock:
            self.detector.reset()
            self.last_sample = float('-inf')
            self.sleep_hits, self.last_sleep_hit = 0, None
            return self.token if self.enabled.is_set() else None

    def sample(self, recognizer, stream, gate, epoch, token):
        with self.lock:
            now = self.clock()
            if (not self.enabled.is_set() or token != self.token or self.stopped or
                    not gate.current(epoch) or now - self.last_sample < .2):
                return
            self.last_sample = now
            text = recognizer.get_result(stream).strip()
            if self.detector.feed(text, now=now):
                self.stopped = True
                gate.command('pause')
                self.output('music_stop', token=token)
            elif is_sleep_command(text):
                if self.last_sleep_hit is not None and now - self.last_sleep_hit > 1.:
                    self.sleep_hits = 0
                self.last_sleep_hit = now
                self.sleep_hits += 1
                if self.sleep_hits >= 2:
                    self.stopped = True
                    gate.command('pause')
                    self.output('standby_sleep', token=token)
            else:
                self.sleep_hits, self.last_sleep_hit = 0, None

    def window_event(self, token, kind, **values):
        if token is not None and kind in ('final', 'utterance_cancelled'):
            with self.lock:
                # Even a late event from a closed music stream stays local.
                if token == self.token and self.stopped:
                    return
                stopped = kind == 'final' and is_stop_request(values.get('text'))
                sleeping = (kind == 'final' and not stopped and self.enabled.is_set() and
                            token == self.token and is_sleep_command(values.get('text')))
                if stopped and token == self.token:
                    self.stopped = True
                elif sleeping:
                    self.stopped = True
            self.output('music_stop' if stopped else 'standby_sleep' if sleeping else 'music_utterance',
                        token=token)
        else:
            self.output(kind, **values)


class StandbyASRControl:
    """Wake-only recognition: standby speech stays inside the owned worker."""
    def __init__(self, output, clock=time.monotonic, lock=None):
        self.output, self.clock = output, clock
        self.lock = lock or threading.RLock()
        self.enabled = threading.Event()
        self.token = None
        self.woken = False
        self._reset_partial()

    def _reset_partial(self):
        self.last_sample = float('-inf')
        self.last_hit = None
        self.hits = 0

    def command(self, enabled, token=None):
        with self.lock:
            self.token = token if enabled else None
            self.woken = False
            self._reset_partial()
            (self.enabled.set if enabled else self.enabled.clear)()

    def begin_window(self):
        with self.lock:
            self._reset_partial()
            return self.token if self.enabled.is_set() else None

    def sample(self, recognizer, stream, gate, epoch, token):
        with self.lock:
            now = self.clock()
            if (not self.enabled.is_set() or token != self.token or self.woken or
                    not gate.current(epoch) or now - self.last_sample < .2):
                return
            self.last_sample = now
            text = recognizer.get_result(stream).strip()
            if not is_wake_command(text):
                self.hits, self.last_hit = 0, None
                return
            if self.last_hit is not None and now - self.last_hit > 1.:
                self.hits = 0
            self.last_hit = now
            self.hits += 1
            if self.hits >= 2:
                self.woken = True
                # The UI waits for listen_once's later paused acknowledgement,
                # after Capture has released ALSA, before any wake animation.
                gate.command('pause')
                self.output('standby_wake', token=token)

    def window_event(self, token, kind, **values):
        if token is not None and kind in ('final', 'utterance_cancelled'):
            with self.lock:
                # A closed standby window never becomes normal conversation,
                # and an old standby token cannot wake a later standby session.
                if not self.enabled.is_set() or token != self.token or self.woken:
                    return
                wake = kind == 'final' and is_wake_command(values.get('text'))
                if wake:
                    self.woken = True
                self.output('standby_wake' if wake else 'standby_utterance', token=token)
        else:
            self.output(kind, **values)


class LocalASRControls:
    """Atomically select one control-only mode for the next capture window."""
    def __init__(self, output, clock=time.monotonic):
        self.lock = threading.RLock()
        self.music = MusicASRControl(output, clock, self.lock)
        self.standby = StandbyASRControl(output, clock, self.lock)

    def command(self, mode, enabled, token=None):
        with self.lock:
            selected, other = ((self.music, self.standby) if mode == 'music' else
                               (self.standby, self.music))
            if enabled:
                other.command(False)
            selected.command(enabled, token)

    def begin_window(self):
        with self.lock:
            return self.music.begin_window(), self.standby.begin_window()


class MusicControlledDecoder(Decoder):
    def __init__(self, *args, music_control, gate, epoch, music_token,
                 standby_control=None, standby_token=None, **kwargs):
        if music_token is not None or standby_token is not None:
            # A closed music stream may finish decoding after normal chat has
            # resumed. It must never acquire a server speech_start lease.
            kwargs['on_speech_start'] = None
        super().__init__(*args, **kwargs)
        self.music_control = music_control
        self.music_gate, self.music_epoch, self.music_token = gate, epoch, music_token
        self.standby_control, self.standby_token = standby_control, standby_token

    def feed(self, data, rate=16000):
        super().feed(data, rate)
        if self.standby_token is not None:
            self.standby_control.sample(self.recognizer, self.stream, self.music_gate,
                                        self.music_epoch, self.standby_token)
        elif self.music_token is not None:
            self.music_control.sample(self.recognizer, self.stream, self.music_gate,
                                      self.music_epoch, self.music_token)


def emit(kind, **values):
    with EMIT_LOCK:
        print(json.dumps({'type': kind, **values}, ensure_ascii=False), flush=True)


def emit_error(error):
    # OSError from loading a model/WAV is not evidence of a missing microphone.
    # Capture attaches a recoverable flag only to explicit device diagnostics.
    emit('error', message=str(error)[:1200],
         reason=getattr(error, 'reason', 'asr_error'),
         recoverable=getattr(error, 'recoverable', False) is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asr-root', required=True, type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--wav', type=Path, help='公开样本测试，不录音')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--end-silence', type=float, default=.7,
                        help='停顿多少秒提交整句，0.5到5，默认0.7')
    args = parser.parse_args()
    if not math.isfinite(args.end_silence) or not .5 <= args.end_silence <= 5:
        parser.error('--end-silence需要在0.5到5秒之间')
    gate = MicrophoneGate(not args.paused)
    local_controls = LocalASRControls(emit)
    music_control, standby_control = local_controls.music, local_controls.standby
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: gate.command('stop'))

    def control():
        try:
            for line in sys.stdin:
                parts = line.strip().split()
                name = parts[0] if parts else ''
                if name in ('music_control', 'standby_control') and len(parts) >= 2:
                    mode = 'music' if name == 'music_control' else 'standby'
                    if parts[1] == 'off':
                        local_controls.command(mode, False)
                    elif parts[1] == 'on' and len(parts) == 3:
                        local_controls.command(mode, True, parts[2])
                    continue
                if name in ('pause', 'resume', 'stop'):
                    pause_id = parts[1] if len(parts) == 2 and name == 'pause' else None
                    if gate.command(name, pause_id):
                        emit('paused', pause_id=pause_id)
                if name == 'stop':
                    return
        finally:
            gate.command('stop')
    try:
        emit('loading')
        if not args.wav:
            threading.Thread(target=control, daemon=True).start()
        import numpy as np
        import sherpa_onnx
        config = selection()
        profile = config['profile']
        model = config['model'] or args.asr_root / 'models' / MODEL_NAMES[profile]
        # Load once; each listen window creates only a fresh recognizer stream.
        recognizer = create_recognizer(sherpa_onnx, model, profile, config['threads'],
                                       endpoint_silence=args.end_silence)
        def make_decoder(output, epoch=None, music_token=None, standby_token=None):
            return MusicControlledDecoder(recognizer, np, output=output, handler=lambda text: None, profile=profile,
                           endpoint_silence=args.end_silence, silence_rms=config['silence_rms'],
                           music_control=music_control, gate=gate, epoch=epoch, music_token=music_token,
                           standby_control=standby_control, standby_token=standby_token,
                           on_speech_start=lambda: emit('utterance_started'))
        if args.wav:
            decoder = make_decoder(lambda text, **_: emit('final', text=text))
            with wave.open(str(args.wav), 'rb') as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (1, 2, 'NONE'):
                    raise UserError('样本必须是PCM16单声道WAV')
                for block in iter(lambda: stream.readframes(max(1, stream.getframerate() // 50)), b''):
                    decoder.feed(block, stream.getframerate())
            decoder.finish()
            emit('stats', data=decoder.stats())
            return 0
        command = capture_command(require_arecord(), args.device, 16000, period_ms=20)
        if not gate.enabled:
            emit('paused', pause_id=gate.pause_id)
        while True:
            epoch = gate.wait()
            if epoch is None:
                break
            music_token, standby_token = local_controls.begin_window()
            def window_event(kind, **values):
                if standby_token is not None:
                    standby_control.window_event(standby_token, kind, **values)
                else:
                    music_control.window_event(music_token, kind, **values)
            listen_once(gate, epoch, command,
                        lambda output: make_decoder(output, epoch, music_token, standby_token),
                        window_event,
                        capture_factory=Capture)
        return 0
    except (UserError, OSError, RuntimeError, ValueError, ImportError, wave.Error) as error:
        emit_error(error)
        return 1
    finally:
        gate.command('stop')
        emit('stopped')


if __name__ == '__main__':
    raise SystemExit(main())
