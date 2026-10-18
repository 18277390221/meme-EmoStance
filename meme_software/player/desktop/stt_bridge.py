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
from asr_selection import selection


def find_asr_root(player_root, explicit=None):
    config = selection()
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
                (config['model'] is not None or
                 (path / 'models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01/model.int8.onnx').is_file())):
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
        self.on_speech_start = None
        self.on_speech_cancel = None
        self.on_music_stop = None
        self.on_music_utterance = None
        self.on_standby_wake = None
        self.on_standby_utterance = None
        self.on_standby_sleep = None
        self.process = None
        self.thread = None
        self.lock = threading.Lock()
        self.events = deque()
        self.last_activity = 0.0
        self.overflow = False
        self.accepting = True
        self.command_lock = threading.Lock()
        self.paused = True
        self.pause_id = None
        self.command_serial = 0

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

    def start(self, asr_root, device, python=None, wav=None, paused=False, end_silence=.7):
        if self.running():
            raise ValueError('已有本软件的收音实例，请先停止。')
        python = Path(python) if python else Path(asr_root) / '.venv/bin/python'
        if not python.is_file():
            raise ValueError('该STT目录没有已安装的Python环境，不能复制Windows虚拟环境代替。')
        command = [str(python), '-B', '-u', str(self.worker), '--asr-root', str(asr_root), '--device', device]
        if wav:
            command += ['--wav', str(wav)]
        command += ['--end-silence', str(end_silence)]
        if paused:
            command.append('--paused')
        self.paused = paused
        self.pause_id = 'initial' if paused else None
        self.accepting = True
        self.overflow = False
        self.last_activity = 0.0
        with self.lock:
            # The prior reader has exited before a new worker can start. Its
            # queued stopped/exited events must not disable the fresh worker.
            self.events.clear()
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
                            'loading', 'listening', 'paused', 'activity', 'utterance_started',
                            'utterance_cancelled', 'final', 'music_stop', 'music_utterance',
                            'standby_wake', 'standby_utterance', 'standby_sleep',
                            'error', 'stats', 'stopped'}:
                            raise ValueError('unknown event')
                        if event['type'] == 'error':
                            failed = True
                        if event['type'] == 'stopped' or (event['type'] == 'paused' and
                                self.pause_id is not None and event.get('pause_id') == self.pause_id):
                            self.paused = True
                            self.last_activity = 0.0
                        elif event['type'] == 'listening':
                            self.paused = False
                        if event['type'] == 'utterance_started' and self.on_speech_start:
                            self.on_speech_start()
                        elif event['type'] == 'utterance_cancelled' and self.on_speech_cancel:
                            self.on_speech_cancel()
                        if event['type'] == 'final' and self.on_final:
                            event['request_id'] = self.on_final(event.get('text'))
                        elif event['type'] == 'music_stop' and self.on_music_stop:
                            self.on_music_stop(event.get('token'))
                        elif event['type'] == 'music_utterance' and self.on_music_utterance:
                            self.on_music_utterance(event.get('token'))
                        elif event['type'] == 'standby_wake' and self.on_standby_wake:
                            self.on_standby_wake(event.get('token'))
                        elif event['type'] == 'standby_utterance' and self.on_standby_utterance:
                            self.on_standby_utterance(event.get('token'))
                        elif event['type'] == 'standby_sleep' and self.on_standby_sleep:
                            self.on_standby_sleep(event.get('token'))
                        self.publish(event)
                    except (ValueError, TypeError):
                        failed = True
                        self.publish({'type': 'error', 'message': '识别进程返回格式异常，已停止收音。'})
                        self.request_stop()
                code = process.wait()
                if code and not failed:
                    self.publish({'type': 'error', 'message': '识别进程异常退出，请检查设备或本地依赖。'})
            finally:
                self.paused = True
                for pipe in (process.stdout, process.stderr, process.stdin):
                    try:
                        pipe.close()
                    except (OSError, ValueError):
                        pass
                self.publish({'type': 'exited'})
        self.thread = threading.Thread(target=read, daemon=True)
        self.thread.start()

    def request_stop(self):
        self.command('stop')

    def music_control(self, enabled, token=None):
        if enabled:
            if not isinstance(token, str) or not token or any(char.isspace() for char in token):
                raise ValueError('音乐控制需要有效的播放编号。')
            self.command('music_control on ' + token)
        else:
            self.command('music_control off')

    def standby_control(self, enabled, token=None):
        if enabled:
            if not isinstance(token, str) or not token or any(char.isspace() for char in token):
                raise ValueError('待机控制需要有效的会话编号。')
            self.command('standby_control on ' + token)
        else:
            self.command('standby_control off')

    def command(self, name):
        process = self.process
        if process and process.poll() is None:
            try:
                with self.command_lock:
                    process.stdin.write(name + '\n')
                    process.stdin.flush()
            except (OSError, ValueError):
                pass

    def pause(self):
        # paused becomes true only after the worker confirms Capture has closed.
        if self.process and self.process.poll() is None:
            self.command_serial += 1
            self.pause_id = str(self.command_serial)
            self.paused = False
            self.command('pause ' + self.pause_id)
        else:
            self.paused = True

    def resume(self):
        if self.process and self.process.poll() is None:
            self.pause_id = None
            self.paused = False
            self.command('resume')

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
