#!/usr/bin/env python3
"""中文离线语音识别。录音只经过内存管道，绝不落盘或上传。"""
from collections import deque
import errno
import os
import shutil
import subprocess
import sys
import threading
import time

class UserError(Exception):
    def __init__(self, message, *, reason='asr_error', recoverable=False):
        super().__init__(message)
        self.reason = reason
        self.recoverable = recoverable


def capture_failure_reason(detail, error=None):
    """Classify raw capture diagnostics, never model/dependency error text.

    Missing PCM names and generic EOF/timeouts are ambiguous configuration or
    driver faults. Only explicit missing-card/device or device I/O evidence
    allows the UI to wait for the microphone and restart its owned worker.
    """
    low = detail.lower()
    code = getattr(error, 'errno', None)
    if code in (errno.EACCES, errno.EPERM) or any(
            word in low for word in ('permission denied', 'operation not permitted')):
        return 'audio_permission'
    if code == errno.EBUSY or 'device or resource busy' in low or 'device busy' in low:
        return 'audio_device_busy'
    if any(word in low for word in ('overrun', 'xrun')):
        return 'audio_overrun'
    if code == errno.EINVAL or any(word in low for word in (
            'invalid argument', 'sample format', 'rate is not accurate',
            'channels count', 'sample rate', 'unable to install hw params')):
        return 'audio_configuration'
    if code in (errno.ENODEV, errno.ENXIO, errno.EIO) or any(word in low for word in (
            'no such device', 'cannot find card', 'no soundcards found',
            'cannot get card index for',
            'device disconnected', 'device has been disconnected', 'input/output error')):
        return 'audio_device_unavailable'
    # ALSA often appends Unknown PCM after failing to resolve a missing USB
    # card. That generic follow-up must not hide the explicit device failure.
    if 'unknown pcm' in low:
        return 'audio_configuration'
    return 'audio_capture_error'


class CaptureError(UserError):
    def __init__(self, message, *, detail='', error=None, reason=None):
        reason = reason or capture_failure_reason(detail, error)
        super().__init__(message, reason=reason,
                         recoverable=reason == 'audio_device_unavailable')


def handle_command(text):
    """后续可在这里接应用。当前只打印模拟，不访问 GPIO 或服务器。"""
    command = "".join(text.split()).strip("，。！？,.!?")
    if command in ("开灯", "关灯"):
        print("[模拟操作] " + command + "（未连接硬件）", flush=True)


def audio_hint(message):
    low = message.lower()
    if "overrun" in low or "xrun" in low:
        return "录音缓冲区溢出：音频未被及时读取，可能与CPU负载或识别耗时有关；不是麦克风权限错误。"
    if "permission denied" in low or "operation not permitted" in low:
        return "麦克风权限不足：检查 /dev/snd 的访问权限和桌面会话授权，必要时请管理员配置音频组后重新登录。"
    if "busy" in low:
        return "麦克风被其他程序占用，请关闭录音程序或改选设备。"
    if any(x in low for x in ("rate", "channels", "sample format", "invalid argument")):
        return "设备不支持当前音频参数。优先选择 plughw:CARD=声卡名,DEV=0，让 ALSA 转换为单声道、16 位、指定采样率。"
    if any(x in low for x in ("no such", "unknown pcm", "no soundcards", "cannot find card", "no such device")):
        return "找不到录音设备，请插好 USB 麦克风，运行 --list-devices，再用 --device 选择录音设备；HDMI 输出不是麦克风。"
    return "录音失败：检查设备连接、权限、占用情况及 --device 名称。"


def require_arecord():
    if sys.platform != "linux":
        raise UserError("此录音程序面向树莓派 Linux。当前不是 Linux；可运行 --help 或环境检测脚本。")
    path = shutil.which("arecord")
    if not path:
        raise UserError("缺少 ALSA 录音工具 arecord。请使用匹配系统的 system-debs 离线安装，不要在树莓派联网下载。",
                        reason='audio_tool_missing')
    return path


def list_devices(arecord):
    for option, title in (("-l", "硬件录音设备"), ("-L", "可选择的 ALSA 设备名（--device 使用这里的名称）")):
        print("\n" + title + "：", flush=True)
        try:
            result = subprocess.run([arecord, option], capture_output=True, text=True, errors="replace", timeout=10)
        except subprocess.TimeoutExpired:
            raise UserError("列举录音设备超时，请检查声卡驱动。")
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(audio_hint(result.stderr) + "\nALSA: " + result.stderr.strip(), file=sys.stderr)
        if result.returncode:
            raise UserError("无法列出设备，退出码 %s。" % result.returncode)


def capture_command(arecord, device, rate, period_ms=100):
    # plughw/default 支持 ALSA 插件转换；不允许静默协商成另一采样率。
    # 不用 --fatal-errors：允许ALSA恢复短暂XRUN；丢帧时显式重置本句。
    if device.startswith("hw:"):
        device = "plughw:" + device[3:]
    elif not device.startswith(("plug:", "plughw:")):
        device = "plug:" + device
    return [arecord, "--device=" + device, "--format=S16_LE", "--channels=1",
            "--rate=" + str(rate), "--file-type=raw", "--quiet",
            "--buffer-time=1000000", "--period-time=" + str(period_ms * 1000), "-"]


class Capture:
    def __init__(self, command, strict=False, frame_ms=100, new_session=True):
        self.command = command
        self.new_session = new_session
        self.errors = deque(maxlen=30)
        self.proc = None
        self.reader = None
        self.audio_reader = None
        self.format_error = False
        self.strict = strict
        self.stream_error = None
        self.stream_error_reason = None
        self.normal_finish = False
        self.input_bytes = 0
        self.delivered_bytes = 0
        self.discarded_bytes = 0
        self.peak_queued_bytes = 0
        self.max_wait_seconds = 0.0
        self.overrun_events = 0
        self.overflow_events = 0
        self.condition = threading.Condition()
        self.stopping = threading.Event()
        self.blocks = deque()
        self.queued_bytes = 0
        rate = next((int(x.split("=", 1)[1]) for x in command if x.startswith("--rate=")), 16000)
        self.rate = rate
        self.max_bytes = rate * 2 * 3  # 最多3秒PCM，识别跟不上时不无限积压。
        self.block_bytes = max(2, rate * 2 * frame_ms // 1000 // 2 * 2)
        self.epoch = 0
        self.gap_before_chunk = False
        self.finished = False
        self.read_error = None
        self.read_exception = None
        self.last_data = time.monotonic()

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.DEVNULL,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                         start_new_session=self.new_session, env=dict(os.environ, LC_ALL="C"))
            self.reader = threading.Thread(target=self._read_errors, daemon=True)
            self.reader.start()
            self.audio_reader = threading.Thread(target=self._read_audio, daemon=True)
            self.audio_reader.start()
            return self
        except BaseException:
            # __enter__未成功时Python不会自动调用__exit__，启动中取消也要释放设备。
            self.__exit__()
            raise

    def _read_errors(self):
        for line in iter(self.proc.stderr.readline, b""):
            text = line.decode("utf-8", errors="replace").strip()
            self.errors.append(text)
            if capture_failure_reason(text) == 'audio_device_unavailable':
                with self.condition:
                    # Do not decode buffered partial speech after losing the
                    # source, even if an endpoint is hidden in those samples.
                    self.stream_error = audio_hint(text) + '\n录音流已停止。ALSA: ' + text
                    self.stream_error_reason = 'audio_device_unavailable'
                    self.condition.notify_all()
            if "rate is not accurate" in text.lower():
                with self.condition:
                    self.format_error = True
                    self.condition.notify_all()
            if "overrun" in text.lower():
                with self.condition:
                    self.overrun_events += 1
                    if self.strict:
                        if self.stream_error_reason != 'audio_device_unavailable':
                            self.stream_error = 'ALSA发生音频溢出，无法保证完整收音；已停止继续识别。请降低负载后重试。'
                            self.stream_error_reason = 'audio_overrun'
                    else:
                        self._discard_locked()
                    self.condition.notify_all()

    def _discard_locked(self):
        self.discarded_bytes += self.queued_bytes
        self.blocks.clear()
        self.queued_bytes = 0
        self.epoch += 1

    def _read_audio(self):
        # 采集线程始终取走PCM，不等待模型推理或终端打印。
        pending = b""
        try:
            while not self.stopping.is_set():
                block = os.read(self.proc.stdout.fileno(), self.block_bytes)
                if not block:
                    break
                pending += block
                end = len(pending) - len(pending) % 2
                with self.condition:
                    self.last_data = time.monotonic()
                    if end:
                        data, pending = pending[:end], pending[end:]
                        self.input_bytes += len(data)
                        if self.queued_bytes + len(data) > self.max_bytes:
                            self.overflow_events += 1
                            if self.strict:
                                if self.stream_error_reason != 'audio_device_unavailable':
                                    self.stream_error = '识别积压超过3秒，已停止；没有清空旧音频后假装继续。请降低负载或线程数后重试。'
                                    self.stream_error_reason = 'audio_backlog'
                                # 不清空尚待处理的PCM；拒收本次新增数据并明确记录其字节数。
                                self.discarded_bytes += len(data)
                                self.condition.notify_all()
                                break
                            self._discard_locked()
                        self.blocks.append((data, self.epoch, time.monotonic()))
                        self.queued_bytes += len(data)
                        self.peak_queued_bytes = max(self.peak_queued_bytes, self.queued_bytes)
                    self.condition.notify_all()
        except Exception as exc:
            self.read_error = str(exc)
            self.read_exception = exc
            if capture_failure_reason(self.read_error, exc) == 'audio_device_unavailable':
                with self.condition:
                    self.stream_error = '录音流已停止。读取错误：' + self.read_error
                    self.stream_error_reason = 'audio_device_unavailable'
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def chunks(self):
        previous_epoch = 0
        while True:
            with self.condition:
                if self.stream_error and (not self.blocks or
                                          self.stream_error_reason == 'audio_device_unavailable'):
                    raise CaptureError(self.stream_error, reason=self.stream_error_reason or 'audio_capture_error')
                if self.format_error:
                    raise CaptureError("ALSA 实际采样率与请求值不一致，已停止，避免用错误采样率识别。请使用 plughw 设备。",
                                       reason='audio_configuration')
                if not self.blocks:
                    if self.finished:
                        break
                    if time.monotonic() - self.last_data > 15:
                        raise CaptureError("15 秒未收到音频数据；静音也应产生 PCM 数据。请检查麦克风和驱动。",
                                           reason='audio_timeout')
                    self.condition.wait(timeout=0.2)
                    continue
                data, epoch, queued_at = self.blocks.popleft()
                self.queued_bytes -= len(data)
                self.delivered_bytes += len(data)
                self.max_wait_seconds = max(self.max_wait_seconds, time.monotonic() - queued_at)
            self.gap_before_chunk = epoch != previous_epoch
            previous_epoch = epoch
            yield data
        self.reader.join(timeout=1)
        detail = "\n".join(self.errors)
        if self.read_error:
            detail += "\n读取错误：" + self.read_error
        # A pause racing with a USB disconnect must not turn truncated buffered
        # audio into a successful final merely because terminate() was requested.
        if (self.normal_finish and not self.stream_error and not self.read_error and
                capture_failure_reason(detail) != 'audio_device_unavailable'):
            return
        raise CaptureError(audio_hint(detail) + "\n录音流已停止。ALSA: " + (detail or "无错误详情"),
                           detail=detail, error=self.read_exception)

    def request_finish(self):
        """停止录音子进程，读取线程继续排空管道；chunks再排空已经收到的PCM。"""
        self.normal_finish = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)

    def stats(self):
        with self.condition:
            return {'input_samples': self.input_bytes // 2,
                    'delivered_samples': self.delivered_bytes // 2,
                    'queued_samples': self.queued_bytes // 2,
                    'discarded_samples': self.discarded_bytes // 2,
                    'peak_queue_ms': round(self.peak_queued_bytes / (self.rate * 2) * 1000, 1),
                    'max_queue_wait_ms': round(self.max_wait_seconds * 1000, 1),
                    'overrun_events': self.overrun_events, 'overflow_events': self.overflow_events}

    def __exit__(self, *_):
        self.stopping.set()
        if self.proc:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
            if self.reader and self.reader.is_alive():
                self.reader.join(timeout=1)
            if self.audio_reader and self.audio_reader.is_alive():
                self.audio_reader.join(timeout=2)
            self.proc.stdout.close()
            self.proc.stderr.close()

