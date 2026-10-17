"""Native rendering/audio helpers; no browser and no network clients."""
from contextlib import closing
from io import BytesIO
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import wave

from PIL import Image


def check_library(root):
    path = Path(root) / 'data/assets.sqlite3'
    if not path.is_file():
        raise ValueError('缺少 data/assets.sqlite3，请复制完整播放器，不能只复制桌面入口。')
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        count = db.execute('SELECT count(*) FROM assets').fetchone()[0]
    if not count:
        raise ValueError('素材库为空；请恢复原有 data，程序不会自动重新导入已删除素材。')
    return count


def read_asset(player, key):
    with closing(player.connect()) as db:
        row = db.execute('SELECT * FROM assets WHERE id=?', (key,)).fetchone()
    if row is None:
        raise ValueError('此素材不在本地数据库中，请先导入素材。')
    return dict(row)


def validate_wav(blob):
    with wave.open(BytesIO(blob), 'rb') as audio:
        if audio.getcomptype() != 'NONE':
            raise ValueError('声音需要未压缩的 WAV 文件。')
        return audio.getnframes() / audio.getframerate()


class Clip:
    """Decode one GIF frame at a time, preserving disposal/compositing via Pillow."""
    def __init__(self, blob, duration):
        self.source = Image.open(BytesIO(blob))
        if self.source.format not in ('GIF', 'PNG', 'JPEG', 'WEBP'):
            self.source.close()
            raise ValueError('桌面版支持 GIF/PNG/JPEG/WebP；视频请先转换为 GIF。')
        self.duration = float(duration)
        self.current = None
        self.index = -1

    def next(self):
        self.index += 1
        try:
            self.source.seek(self.index)
        except EOFError:
            self.index = 0
            self.source.seek(0)
        self.current = self.source.convert('RGBA')
        return max(20, min(int(self.source.info.get('duration', 100) or 100), 60000))

    def fit(self, width, height):
        image = self.current.copy()
        image.thumbnail((max(1, width), max(1, height)), Image.Resampling.BILINEAR)
        # thumbnail preserves original size for tiny images; upscale if appropriate.
        scale = min(width / image.width, height / image.height)
        if scale > 1:
            image = image.resize((max(1, round(image.width * scale)),
                                  max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
        return image

    def close(self):
        self.source.close()
        self.current = None


class Audio:
    def __init__(self, muted=False):
        self.muted = muted
        self.process = None
        self.errors = []
        self.lock = threading.Lock()
        self.scratch = None

    def play(self, blob):
        self.stop()
        if self.muted or not blob:
            return
        validate_wav(blob)
        if sys.platform == 'win32':
            import winsound
            self.scratch = tempfile.TemporaryDirectory(prefix='meme-cat-audio-')
            path = Path(self.scratch.name) / 'asset.wav'
            path.write_bytes(blob)
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        executable = shutil.which('aplay')
        if not executable:
            raise ValueError('缺少 aplay（alsa-utils）；当前机器不能播放配套声音。')
        # Send the existing WAV bytes; never record audio or override its sample rate.
        process = subprocess.Popen([executable, '-q'], stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.process = process

        def feed():
            try:
                _, error = process.communicate(input=blob)
                if process.returncode and self.process is process:
                    with self.lock:
                        self.errors.append('声音播放失败：' + error.decode('utf-8', 'replace')[:300])
            except (OSError, ValueError) as error:
                if self.process is process:
                    with self.lock:
                        self.errors.append('声音播放失败：' + str(error))
        threading.Thread(target=feed, daemon=True).start()

    def pop_errors(self):
        with self.lock:
            errors, self.errors = self.errors, []
        return errors

    def stop(self):
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=.4)
            except subprocess.TimeoutExpired:
                process.kill()
            except ProcessLookupError:
                pass
        if sys.platform == 'win32' and not self.muted:
            import winsound
            winsound.PlaySound(None, 0)
        if self.scratch is not None:
            self.scratch.cleanup()
            self.scratch = None
