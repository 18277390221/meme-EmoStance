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

sys.path.insert(0, str(Path(__file__).resolve().parent / 'asr_runtime'))
from audio import Capture, UserError, capture_command, require_arecord
from speech import Decoder, MODEL_NAMES, create_recognizer
from capture_turn import MicrophoneGate, listen_once


EMIT_LOCK = threading.Lock()


def emit(kind, **values):
    with EMIT_LOCK:
        print(json.dumps({'type': kind, **values}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asr-root', required=True, type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--wav', type=Path, help='公开样本测试，不录音')
    parser.add_argument('--paused', action='store_true')
    parser.add_argument('--end-silence', type=float, default=1.2)
    args = parser.parse_args()
    if not math.isfinite(args.end_silence) or not .5 <= args.end_silence <= 5:
        parser.error('--end-silence需要在0.5到5秒之间')
    gate = MicrophoneGate(not args.paused)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: gate.command('stop'))

    def control():
        try:
            for line in sys.stdin:
                parts = line.strip().split()
                name = parts[0] if parts else ''
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
        model = args.asr_root / 'models' / MODEL_NAMES['lite']
        # Load once; each listen window creates only a fresh recognizer stream.
        recognizer = create_recognizer(sherpa_onnx, model, 'lite', 2,
                                       endpoint_silence=args.end_silence)
        def make_decoder(output):
            return Decoder(recognizer, np, output=output, handler=lambda text: None)
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
            listen_once(gate, epoch, command, make_decoder, emit, capture_factory=Capture)
        return 0
    except (UserError, OSError, RuntimeError, ValueError, ImportError, wave.Error) as error:
        emit('error', message=str(error)[:1200])
        return 1
    finally:
        gate.command('stop')
        emit('stopped')


if __name__ == '__main__':
    raise SystemExit(main())
