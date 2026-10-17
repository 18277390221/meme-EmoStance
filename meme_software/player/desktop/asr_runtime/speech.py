#!/usr/bin/env python3
"""离线中文流式识别：Sherpa-ONNX + 本地Zipformer INT8模型。"""
import argparse
import json
import signal
import threading
from pathlib import Path
import sys
import time
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import Capture, UserError, capture_command, handle_command, list_devices, require_arecord
from text_review import review_documents

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = {
    "standard": "sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30",
    "lite": "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01",
}


class PartialDisplay:
    """TTY上原位显示可修订临时文字；文件/管道只收到最终句，不刷临时行。"""
    def __init__(self, stream=None, interval=0.15):
        self.stream = stream or sys.stderr
        self.interval, self.last_time, self.text = interval, 0.0, ''
        self.enabled = self.stream.isatty()

    def due(self):
        return self.enabled and time.monotonic() - self.last_time >= self.interval

    def update(self, text):
        self.last_time = time.monotonic()
        if self.enabled and text and text != self.text:
            self.stream.write('\r\x1b[2K[识别中，可修订] ' + text)
            self.stream.flush()
            self.text = text

    def clear(self):
        if self.enabled and self.text:
            self.stream.write('\r\x1b[2K')
            self.stream.flush()
        self.text = ''


def model_files(folder, profile):
    folder = Path(folder).expanduser().resolve()
    names = ["tokens.txt"] + (["encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"]
                              if profile == "standard" else ["model.int8.onnx"])
    for name in names:
        if not (folder / name).is_file():
            raise UserError("本地模型缺失：%s。请保留完整模型目录，或用--model指定对应profile的目录。" % (folder / name))
    return {name: str(folder / name) for name in names}


def create_recognizer(module, folder, profile, threads):
    files = model_files(folder, profile)
    options = dict(tokens=files["tokens.txt"], num_threads=threads, sample_rate=16000,
                   feature_dim=80, provider="cpu", enable_endpoint_detection=True,
                   rule1_min_trailing_silence=2.4, rule2_min_trailing_silence=0.8,
                   rule3_min_utterance_length=20, decoding_method="greedy_search")
    if profile == "standard":
        return module.OnlineRecognizer.from_transducer(
            encoder=files["encoder.int8.onnx"], decoder=files["decoder.onnx"],
            joiner=files["joiner.int8.onnx"], **options)
    return module.OnlineRecognizer.from_zipformer2_ctc(model=files["model.int8.onnx"], **options)


class Decoder:
    def __init__(self, recognizer, numpy, output=print, handler=handle_command, reviewer=None, partial=None):
        self.recognizer, self.np = recognizer, numpy
        self.output, self.handler = output, handler
        self.reviewer = reviewer
        self.partial = partial
        self.stream = recognizer.create_stream()
        self.inference_seconds = 0.0
        self.audio_seconds = 0.0
        self.gaps = 0
        self.input_rate = None

    def emit(self):
        if self.partial:
            self.partial.clear()
        text = self.recognizer.get_result(self.stream).strip()
        if text:
            self.output(text, flush=True)
            if self.reviewer:
                review = self.reviewer(text)
                if review.changes:
                    print('[文档场景建议，待确认] ' + review.suggestion, file=sys.stderr, flush=True)
            self.handler(text)

    def decode_ready(self):
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def feed(self, data, rate=16000):
        if len(data) % 2:
            raise UserError("PCM长度不是16位样本的整数倍，已停止。")
        if self.input_rate is not None and self.input_rate != rate:
            raise UserError("同一录音流的采样率发生变化，已停止以避免错配。")
        self.input_rate = rate
        started = time.monotonic()
        samples = self.np.frombuffer(data, dtype="<i2").astype(self.np.float32) / 32768.0
        self.stream.accept_waveform(rate, samples)
        self.decode_ready()
        if self.recognizer.is_endpoint(self.stream):
            self.emit()
            self.recognizer.reset(self.stream)
        elif self.partial and self.partial.due():
            self.partial.update(self.recognizer.get_result(self.stream).strip())
        self.inference_seconds += time.monotonic() - started
        self.audio_seconds += len(data) / (rate * 2.0)

    def discontinuity(self):
        if self.partial:
            self.partial.clear()
        # 创建新流清除编码器上下文，不能把断流两侧拼成一句或指令。
        self.stream = self.recognizer.create_stream()
        self.gaps += 1
        print("警告：音频有缺失，本句已丢弃，请重说。可加--stats查看耗时，或用--profile lite测试轻量模式。",
              file=sys.stderr, flush=True)

    def finish(self):
        rate = self.input_rate or 16000
        self.stream.accept_waveform(rate, self.np.zeros(rate // 2, dtype=self.np.float32))
        self.stream.input_finished()
        self.decode_ready()
        self.emit()

    def stats(self):
        return {"engine": "sherpa-onnx", "audio_seconds": round(self.audio_seconds, 2),
                "inference_seconds": round(self.inference_seconds, 2),
                "rtf": round(self.inference_seconds / self.audio_seconds, 3) if self.audio_seconds else None,
                "gap_events": self.gaps}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", default="default", help="ALSA录音设备名")
    parser.add_argument("--profile", choices=MODEL_NAMES, default="lite", help="默认lite轻量中文模型；standard中文主模型")
    parser.add_argument("--model", type=Path, help="显式指定本地模型目录，不自动下载")
    parser.add_argument("--threads", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--sample-rate", type=int, choices=(16000, 32000, 44100, 48000), default=16000)
    parser.add_argument("--stats", action="store_true", help="每10秒及退出时显示真实处理RTF/丢帧统计")
    parser.add_argument('--realtime', action='store_true', help='20ms持续采集、TTY临时文字、排空退出；溢出明确停止')
    parser.add_argument("--text-domain", choices=("none", "documents"), default="none",
                        help="documents仅额外显示规则纠错建议，保留原始文字，不自动执行建议；默认none")
    parser.add_argument("--wav", type=Path, help="仅测试：识别已有PCM16单声道WAV；默认只使用麦克风")
    args = parser.parse_args(argv)
    decoder = None
    capture = None
    partial = PartialDisplay() if args.realtime and not args.wav else None
    try:
        if args.list_devices:
            list_devices(require_arecord())
            return 0
        arecord = None if args.wav else require_arecord()
        folder = args.model or ROOT / "models" / MODEL_NAMES[args.profile]
        model_files(folder, args.profile)
        try:
            import numpy as np
            import sherpa_onnx
        except (ImportError, OSError) as exc:
            raise UserError("离线依赖/本地动态库无法加载，请先安装匹配系统的升级包：%s" % exc)
        print("加载本地中文模型（%s），不联网……" % args.profile, file=sys.stderr, flush=True)
        decoder = Decoder(create_recognizer(sherpa_onnx, folder, args.profile, args.threads), np,
                          reviewer=review_documents if args.text_domain == 'documents' else None,
                          partial=partial)
        if args.wav:
            with wave.open(str(args.wav), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getcomptype() != "NONE":
                    raise UserError("测试WAV必须是PCM16单声道；不自动改写原文件。")
                for block in iter(lambda: stream.readframes(1600), b""):
                    decoder.feed(block, stream.getframerate())
            decoder.finish()
        else:
            print("正在离线识别；说话后停顿约1秒看整句，Ctrl+C退出。设备：%s；PCM16单声道%dHz；线程%d。" %
                  (args.device, args.sample_rate, args.threads), file=sys.stderr, flush=True)
            last_stats = time.monotonic()
            command = capture_command(arecord, args.device, args.sample_rate, period_ms=20 if args.realtime else 100)
            capture_instance = Capture(command, strict=True, frame_ms=20) if args.realtime else Capture(command)
            with capture_instance as capture:
                try:
                    if args.realtime:
                        consume_realtime(capture, decoder, args.sample_rate, args.stats, partial)
                    else:
                        for block in capture.chunks():
                            if capture.gap_before_chunk:
                                decoder.discontinuity()
                            decoder.feed(block, args.sample_rate)
                            if args.stats and time.monotonic() - last_stats >= 10:
                                print("[性能] " + json.dumps(decoder.stats(), ensure_ascii=False), file=sys.stderr, flush=True)
                                last_stats = time.monotonic()
                except KeyboardInterrupt:
                    decoder.finish()
            print("已退出并释放麦克风。", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 0
    except (UserError, OSError, RuntimeError, ValueError, wave.Error) as exc:
        print("识别停止：%s。录音子进程已清理；未保存或上传音频。" % exc, file=sys.stderr)
        return 1
    finally:
        if partial:
            partial.clear()
        if args.stats and decoder:
            statistics = decoder.stats()
            if args.realtime and capture:
                statistics['capture'] = capture.stats()
            print("[性能] " + json.dumps(statistics, ensure_ascii=False), file=sys.stderr, flush=True)


def consume_realtime(capture, decoder, rate, show_stats=False, partial=None, stop_event=None):
    """SIGINT只请求结束；当前块及队列排空后补尾，避免立刻中断丢尾块。"""
    stop_event = stop_event or threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    finishing, last_stats = False, time.monotonic()
    try:
        for block in capture.chunks():
            if stop_event.is_set() and not finishing:
                capture.request_finish()
                finishing = True
            decoder.feed(block, rate)
            if show_stats and time.monotonic() - last_stats >= 10:
                if partial:
                    partial.clear()
                print('[性能] ' + json.dumps(dict(decoder.stats(), capture=capture.stats()), ensure_ascii=False),
                      file=sys.stderr, flush=True)
                last_stats = time.monotonic()
        decoder.finish()
    finally:
        signal.signal(signal.SIGINT, previous)


if __name__ == "__main__":
    sys.exit(main())
