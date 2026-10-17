#!/usr/bin/env python3
"""Owned ASR subprocess: local model/PCM -> JSON lines, never audio files/network."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent / 'asr_runtime'))
from audio import Capture, UserError, capture_command, require_arecord
from speech import Decoder, MODEL_NAMES, create_recognizer


def emit(kind, **values):
    print(json.dumps({'type': kind, **values}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asr-root', required=True, type=Path)
    parser.add_argument('--device', required=True)
    parser.add_argument('--wav', type=Path, help='公开样本测试，不录音')
    args = parser.parse_args()
    stop, done = threading.Event(), threading.Event()
    decoder, capture = None, None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def control():
        # Parent exit/closed pipe also requests microphone release.
        while True:
            try:
                data = os.read(sys.stdin.fileno(), 64)
            except OSError:
                break
            if not data or b'stop' in data:
                break
        stop.set()
    try:
        emit('loading')
        import numpy as np
        import sherpa_onnx
        model = args.asr_root / 'models' / MODEL_NAMES['lite']
        decoder = Decoder(create_recognizer(sherpa_onnx, model, 'lite', 2), np,
                          output=lambda text, **_: emit('final', text=text),
                          handler=lambda text: None)
        if not args.wav:
            threading.Thread(target=control, daemon=True).start()
        if stop.is_set():
            return 0
        if args.wav:
            with wave.open(str(args.wav), 'rb') as stream:
                if (stream.getnchannels(), stream.getsampwidth(), stream.getcomptype()) != (1, 2, 'NONE'):
                    raise UserError('样本必须是PCM16单声道WAV')
                emit('listening')
                for block in iter(lambda: stream.readframes(max(1, stream.getframerate() // 50)), b''):
                    if stop.is_set():
                        break
                    decoder.feed(block, stream.getframerate())
            decoder.finish()
            return 0
        command = capture_command(require_arecord(), args.device, 16000, period_ms=20)
        # arecord stays in our worker's process group, so emergency cleanup cannot orphan it.
        with Capture(command, strict=True, frame_ms=20, new_session=False) as capture:
            def monitor():
                while not done.wait(.05):
                    if stop.is_set():
                        try:
                            capture.request_finish()
                        except OSError:
                            pass
                        return
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            last_activity = last_stats = 0.0
            heard = False
            try:
                for block in capture.chunks():
                    if not heard:
                        heard = True
                        emit('listening')
                    decoder.feed(block, 16000)
                    now = time.monotonic()
                    if now - last_activity >= .2:
                        emit('activity')
                        last_activity = now
                    if now - last_stats >= 10:
                        emit('stats', data=dict(decoder.stats(), capture=capture.stats()))
                        last_stats = now
                decoder.finish()
            finally:
                done.set()
                watcher.join(timeout=2.2)
        return 0
    except (UserError, OSError, RuntimeError, ValueError, ImportError, wave.Error) as error:
        emit('error', message=str(error)[:1200])
        return 1
    finally:
        done.set()
        if decoder:
            emit('stats', data=dict(decoder.stats(), capture=capture.stats() if capture else {}))
        emit('stopped')


if __name__ == '__main__':
    raise SystemExit(main())
