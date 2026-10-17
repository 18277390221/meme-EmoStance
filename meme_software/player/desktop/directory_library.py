"""Read-only, exact-ID mapping for the team's sucai_meme/meme_gif media."""
import base64
from contextlib import closing
from io import BytesIO
from pathlib import Path
import re
import threading
import wave

from PIL import Image
from launch import Player, PlayerError

# Previously removed silent media stay excluded; never silently restore them.
EXCLUDED = frozenset({'meme-0056', 'meme-0153', 'meme-0175', 'meme-0176', 'meme-0179'})
FILE_ID = re.compile(r'^(meme-\d{4})(?:-.+)?\.gif$', re.IGNORECASE)


def gif_duration(blob):
    """Scan GIF block timings without decoding pixels before the renderer does."""
    stream = BytesIO(blob)
    def take(size):
        result = stream.read(size)
        if len(result) != size:
            raise PlayerError('GIF内容不完整，无法确定完整播放时长。')
        return result
    def subblocks():
        while True:
            size = take(1)[0]
            if not size:
                return
            take(size)
    if take(6) not in (b'GIF87a', b'GIF89a'):
        raise PlayerError('GIF文件头无效。')
    screen = take(7)
    if screen[4] & 128:
        take(3 * (2 ** ((screen[4] & 7) + 1)))
    seconds, delay, frames = 0.0, .1, 0
    while True:
        marker = take(1)[0]
        if marker == 0x3B:
            if not frames:
                raise PlayerError('GIF没有可播放画面。')
            return seconds
        if marker == 0x21:
            label = take(1)[0]
            if label == 0xF9:
                size = take(1)[0]
                data = take(size)
                if size != 4 or take(1) != b'\0':
                    raise PlayerError('GIF帧时长控制块无效。')
                delay = max(.02, min(int.from_bytes(data[1:3], 'little') / 100 or .1, 60))
            else:
                subblocks()
        elif marker == 0x2C:
            descriptor = take(9)
            if descriptor[8] & 128:
                take(3 * (2 ** ((descriptor[8] & 7) + 1)))
            take(1)  # LZW minimum code size.
            subblocks()
            seconds += delay
            delay, frames = .1, frames + 1
        else:
            raise PlayerError('GIF包含未知数据块。')


def find_directory(root, explicit=None):
    root = Path(root).expanduser().resolve()
    if explicit is not None:
        directory = Path(explicit).expanduser().resolve()
        if not directory.is_dir():
            raise PlayerError('指定的表情目录不存在：' + str(directory))
        return directory
    # No drive scans, downloads, or fallback to an unrelated SQLite meme ID.
    for parent in (root, *tuple(root.parents)[:2]):
        for directory in (parent / 'sucai_meme/meme_gif',
                          parent / 'meme-EmoStance/sucai_meme/meme_gif'):
            if directory.is_dir():
                return directory.resolve()
    raise PlayerError('找不到 sucai_meme/meme_gif；请放在播放器或仓库目录下，'
                      '或用 --meme-dir 指定本地完整路径。不会使用旧库中的同名编号代替。')


class DirectoryAssets:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.lock = threading.RLock()
        self.index = {}
        self.refresh()

    def safe_file(self, path):
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory):
            raise PlayerError('素材文件不能是符号链接或位于指定目录外：' + path.name)
        if not path.is_file():
            raise PlayerError('素材文件不存在：' + path.name)
        return path

    def refresh(self):
        with self.lock:
            index = {}
            try:
                paths = sorted(self.directory.iterdir())
            except OSError as error:
                raise PlayerError('无法读取表情目录：' + str(self.directory)) from error
            for path in paths:
                match = FILE_ID.fullmatch(path.name)
                if not match:
                    continue
                key = match[1].lower()
                if key in EXCLUDED:
                    continue
                self.safe_file(path)
                if key in index:
                    raise PlayerError('表情编号重复，无法确定服务器对应文件：' + key)
                audio = path.with_suffix('.wav')
                if audio.exists() or audio.is_symlink():
                    self.safe_file(audio)
                else:
                    audio = None
                index[key] = (path, audio)
            if not index:
                raise PlayerError('表情目录没有可用的 meme-四位编号-名称.gif 文件：' + str(self.directory))
            self.index = index

    def assets(self):
        self.refresh()
        with self.lock:
            return [{'id':key, 'label':visual.stem[len(key):].lstrip('-') or key,
                     'visual_type':'gif', 'duration_seconds':None, 'scene_index':None,
                     'visual_name':visual.name, 'visual_mime':'image/gif',
                     'audio_name':audio.name if audio else None,
                     'audio_mime':'audio/wav' if audio else None,
                     'updated_at':visual.stat().st_mtime_ns}
                    for key,(visual,audio) in sorted(self.index.items())]

    def read(self, key):
        self.refresh()
        with self.lock:
            if key not in self.index:
                raise PlayerError('编号不在 sucai_meme/meme_gif 中或已剔除：' + str(key))
            visual, audio = self.index[key]
            visual_blob = self.safe_file(visual).read_bytes()
            audio_blob = self.safe_file(audio).read_bytes() if audio else None
        # This runs on the rendering worker, never on the Tk input loop.
        audio_seconds = 0.0
        if audio_blob:
            try:
                with wave.open(BytesIO(audio_blob),'rb') as wav:
                    if wav.getcomptype() != 'NONE':
                        raise PlayerError('配套声音必须是未压缩WAV：' + audio.name)
                    audio_seconds = wav.getnframes()/wav.getframerate()
            except wave.Error as error:
                raise PlayerError('配套WAV内容无效：' + audio.name) from error
        gif_seconds = gif_duration(visual_blob)
        with Image.open(BytesIO(visual_blob)) as image:
            if image.format != 'GIF':
                raise PlayerError('扩展名为GIF但内容格式不正确：' + visual.name)
        duration = max(gif_seconds, audio_seconds, .5)
        if duration > 120:
            raise PlayerError('素材时长超过120秒：' + visual.name)
        return {'id':key, 'label':visual.stem[len(key):].lstrip('-') or key,
                'visual_type':'gif','duration_seconds':duration,'scene_index':None,
                'visual_name':visual.name,'visual_mime':'image/gif','visual_blob':visual_blob,
                'audio_name':audio.name if audio else None,'audio_mime':'audio/wav' if audio else None,
                'audio_blob':audio_blob,'updated_at':visual.stat().st_mtime_ns,
                'source_path':str(visual)}


class DirectoryPlayer(Player):
    """Meme IDs exclusively from the team folder; user imports stay in SQLite."""
    def __init__(self, root, directory):
        self.directory_assets = DirectoryAssets(directory)
        super().__init__(root)

    def assets(self):
        return self.directory_assets.assets() + [row for row in super().assets() if row['id'].startswith('user-')]

    def library(self):
        result = super().library()
        allowed = {row['id'] for row in self.assets()}
        result['items'] = [item for item in result['items'] if item['id'] in allowed]
        return result

    def read_native_asset(self, key):
        if key.startswith('meme-'):
            return self.directory_assets.read(key)
        if not key.startswith('user-'):
            raise PlayerError('未知素材编号：' + str(key))
        with closing(self.connect()) as db:
            row = db.execute('SELECT * FROM assets WHERE id=?',(key,)).fetchone()
        if row is None:
            raise PlayerError('用户素材不存在：' + key)
        return dict(row)

    def media(self, key, channel):
        if channel not in ('visual','audio'):
            return None
        if key.startswith('meme-'):
            try:
                row = self.directory_assets.read(key)
            except PlayerError:
                return None
            return row[channel+'_mime'],row[channel+'_blob']
        return super().media(key,channel) if key.startswith('user-') else None

    def save_rows(self, rows):
        rows = list(rows)
        if any(not row[0].startswith('user-') for row in rows):
            raise PlayerError('meme编号由 sucai_meme/meme_gif 只读映射维护；新增素材请用user-编号。')
        return super().save_rows(rows)

    def export_pack(self):
        items=[]
        for entry in self.assets():
            row=self.read_native_asset(entry['id'])
            item={k:row[k] for k in ('id','label','visual_type','duration_seconds','scene_index')}
            for channel in ('visual','audio'):
                blob=row[channel+'_blob']
                item[channel]=None if blob is None else {
                    'name':row[channel+'_name'],'type':row[channel+'_mime'],
                    'data':base64.b64encode(blob).decode('ascii')}
            items.append(item)
        return {'format':'meme-player-library','version':1,'items':items}
