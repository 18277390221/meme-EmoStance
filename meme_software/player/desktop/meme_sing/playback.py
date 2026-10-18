"""Own ffplay music subprocesses without blocking the GUI or taking the mic."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

from .catalog import Track

STDERR_LIMIT = 2048
STOP_GRACE = .5


@dataclass(eq=False)
class _Playback:
    token: str
    track_id: str
    process: subprocess.Popen
    stopping_at: float | None = None
    stderr: bytes = b''


class MusicPlayer:
    def __init__(self, volume=30, executable=None, sink_name=None):
        if type(volume) is not int or not 0 <= volume <= 100:
            raise ValueError('音乐音量需要是0到100的整数。')
        if sink_name is not None and (
                not isinstance(sink_name, str) or
                re.fullmatch(r'[A-Za-z0-9_.:-]+', sink_name) is None):
            raise ValueError('音频输出设备名称无效。')
        self.volume = volume
        self.executable = executable
        self.sink_name = sink_name
        self._lock = threading.RLock()
        self._events = deque(maxlen=128)
        self._active: _Playback | None = None
        self._records: dict[str, _Playback] = {}
        self._workers: set[threading.Thread] = set()
        self._closed = False

    def _event(self, kind, token, track_id, message=None):
        event = {'type': kind, 'token': token, 'track_id': track_id}
        if message is not None:
            event['message'] = message
        self._events.append(event)

    def _stop_locked(self, record):
        if record is None or record.stopping_at is not None:
            return
        record.stopping_at = time.monotonic()
        if record.process.poll() is None:
            try:
                record.process.terminate()
            except ProcessLookupError:
                pass
            except OSError:
                # The owning monitor still applies its bounded kill deadline.
                pass

    def start(self, track: Track) -> str:
        token = uuid.uuid4().hex
        track_id = getattr(track, 'id', '')
        with self._lock:
            try:
                if self._closed:
                    raise ValueError('音乐播放器已关闭。')
                if not isinstance(track_id, str) or not track_id:
                    raise ValueError('歌曲缺少有效编号。')
                path = Path(track.path)
                if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError('本地音乐文件不存在、为空或不允许使用符号链接。')
                executable = str(self.executable) if self.executable is not None else shutil.which('ffplay')
                if not executable:
                    raise ValueError('缺少ffplay，无法播放本地音乐。')
                self._stop_locked(self._active)
                environment = os.environ.copy()
                # SDL's backend is named "pulseaudio"; "pulse" is only the
                # ALSA plugin name and is not an SDL audio driver.
                environment['SDL_AUDIODRIVER'] = 'pulseaudio'
                if self.sink_name is not None:
                    # An explicit missing Pulse sink fails visibly instead of
                    # selecting a different speaker after USB hotplug.
                    environment['PULSE_SINK'] = self.sink_name
                process = subprocess.Popen(
                    [executable, '-nodisp', '-vn', '-autoexit', '-loglevel', 'error',
                     '-nostats', '-volume', str(self.volume), '-i', str(path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, env=environment, start_new_session=True,
                    bufsize=0)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                prefix = '音乐播放失败：'
                self._event('error', token, track_id, prefix + str(error)[:STDERR_LIMIT - len(prefix)])
                return token
            record = _Playback(token, track_id, process)
            self._active = record
            self._records[token] = record
            self._event('started', token, track_id)
            worker = threading.Thread(target=self._monitor, args=(record,),
                                      name='meme-music-' + token[:8], daemon=True)
            self._workers.add(worker)
            worker.start()
            return token

    def _read_stderr(self, record):
        stream = record.process.stderr
        if stream is None:
            return
        try:
            while True:
                data = stream.read(4096)
                if not data:
                    break
                if isinstance(data, str):
                    data = data.encode('utf-8', 'replace')
                with self._lock:
                    record.stderr = (record.stderr + data)[-STDERR_LIMIT:]
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def _monitor(self, record):
        reader = threading.Thread(target=self._read_stderr, args=(record,),
                                  name='meme-music-stderr', daemon=True)
        reader.start()
        error = None
        return_code = None
        killed = False
        try:
            while True:
                try:
                    return_code = record.process.wait(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    with self._lock:
                        deadline = record.stopping_at
                    if deadline is not None and time.monotonic() - deadline >= STOP_GRACE and not killed:
                        try:
                            record.process.kill()
                        except ProcessLookupError:
                            pass
                        killed = True
        except (OSError, ValueError) as exc:
            error = str(exc)
            try:
                if record.process.poll() is None:
                    record.process.kill()
                record.process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        finally:
            reader.join(.2)
            with self._lock:
                if record.stopping_at is not None:
                    kind, message = 'stopped', None
                elif error is not None or return_code != 0:
                    detail = record.stderr.decode('utf-8', 'replace').strip()
                    kind = 'error'
                    prefix = '音乐播放失败（退出码' + str(return_code) + '）：'
                    message = prefix + (error or detail or '播放器未正常退出')[-(STDERR_LIMIT - len(prefix)):]
                else:
                    kind, message = 'finished', None
                self._event(kind, record.token, record.track_id, message)
                if self._active is record:
                    self._active = None
                self._records.pop(record.token, None)
                self._workers.discard(threading.current_thread())

    def stop(self) -> None:
        with self._lock:
            self._stop_locked(self._active)

    def is_playing(self) -> bool:
        with self._lock:
            return self._active is not None and self._active.process.poll() is None

    def process_ids(self) -> list[int]:
        """Return live owned children, including ones still stopping."""
        with self._lock:
            return [record.process.pid for record in self._records.values()
                    if record.process.poll() is None]

    def drain(self) -> list[dict]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for record in tuple(self._records.values()):
                self._stop_locked(record)
            workers = tuple(self._workers)
        deadline = time.monotonic() + 1.2
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
