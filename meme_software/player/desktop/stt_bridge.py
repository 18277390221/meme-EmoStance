"""ASR subprocess lifecycle; UI polling never runs inference or blocking network I/O."""
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def find_asr_root(player_root, explicit=None):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        root = Path(player_root).resolve()
        candidates = [root.parent / 'stt']
        for parent in (root.parent, root.parent.parent):
            candidates += sorted(parent.glob('offline-asr-upgrade-install-*/offline-asr-upgrade'))
    matches = []
    for path in candidates:
        path = path.resolve()
        if path in matches:
            continue
        if ((path / '.venv/bin/python').is_file() and
                (path / 'models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/model.int8.onnx').is_file()):
            matches.append(path)
    if not matches:
        raise ValueError('未找到已离线安装的STT。请在语音设置中选择已有ASR目录，或先离线安装配套stt。')
    if len(matches) > 1:
        raise ValueError('找到多份STT安装，请在语音设置中明确选择一份，避免占用错误的设备。')
    return matches[0]


class SpeechProcess:
    def __init__(self, worker=None, on_final=None):
        self.worker = Path(worker or Path(__file__).with_name('stt_worker.py'))
        self.on_final = on_final
        self.process = None
        self.thread = None
        self.lock = threading.Lock()
        self.events = deque()
        self.last_activity = 0.0
        self.overflow = False
        self.accepting = True

    def publish(self, event):
        with self.lock:
            if event['type'] == 'activity':
                self.last_activity = time.monotonic()
                return
            if len(self.events) >= 256:
                # Fail visibly rather than silently discarding final sentences.
                if not self.overflow:
                    self.overflow = True
                    self.events.append({'type': 'error', 'message': '界面事件积压，已停止收音；请重新开启。'})
                    self.request_stop()
                return
            self.events.append(event)

    def drain(self):
        with self.lock:
            events = list(self.events)
            self.events.clear()
        return events

    def running(self):
        return bool(self.thread and self.thread.is_alive())

    def start(self, asr_root, device, python=None, wav=None):
        if self.running():
            raise ValueError('已有本软件的收音实例，请先停止。')
        python = Path(python) if python else Path(asr_root) / '.venv/bin/python'
        if not python.is_file():
            raise ValueError('该STT目录没有已安装的Python环境，不能复制Windows虚拟环境代替。')
        command = [str(python), '-B', '-u', str(self.worker), '--asr-root', str(asr_root), '--device', device]
        if wav:
            command += ['--wav', str(wav)]
        self.accepting = True
        self.overflow = False
        self.last_activity = 0.0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                        env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'),
                                        start_new_session=sys.platform != 'win32')
        process = self.process
        def stderr():
            # Model diagnostics can be verbose; retain no recognized text or recordings.
            try:
                for _ in process.stderr:
                    pass
            except (OSError, ValueError):
                pass
        threading.Thread(target=stderr, daemon=True).start()

        def read():
            failed = False
            try:
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or event.get('type') not in {
                            'loading', 'listening', 'activity', 'final', 'error', 'stats', 'stopped'}:
                            raise ValueError('unknown event')
                        if event['type'] == 'error':
                            failed = True
                        if event['type'] == 'final' and self.on_final:
                            event['request_id'] = self.on_final(event.get('text'))
                        self.publish(event)
                    except (ValueError, TypeError):
                        failed = True
                        self.publish({'type': 'error', 'message': '识别进程返回格式异常，已停止收音。'})
                        self.request_stop()
                code = process.wait()
                if code and not failed:
                    self.publish({'type': 'error', 'message': '识别进程异常退出，请检查设备或本地依赖。'})
            finally:
                for pipe in (process.stdout, process.stderr, process.stdin):
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
                self.publish({'type': 'exited'})
        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def request_stop(self):
        process = self.process
        if process and process.poll() is None:
            try:
                process.stdin.write('stop\n')
                process.stdin.flush()
            except (OSError, ValueError):
                pass

    def close(self, timeout=8):
        self.request_stop()
        if self.thread:
            self.thread.join(timeout)
        if self.running():
            # Emergency cleanup of only our own worker/arecord process group.
            if sys.platform == 'win32':
                self.process.kill()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.thread.join(2)
            self.publish({'type': 'error', 'message': '收音退出超时，已强制释放；末尾音频未获完整处理。'})
