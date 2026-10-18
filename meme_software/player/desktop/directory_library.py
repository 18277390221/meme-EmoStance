"""Exact meme IDs, conversation-stable background GIFs, and original WAVs."""
import base64
from contextlib import closing
from io import BytesIO
from pathlib import Path
import os
import re
import random
import threading
import wave

from PIL import Image
from launch import Player, PlayerError

# Previously removed silent media stay excluded; never silently restore them.
EXCLUDED = frozenset({'meme-0056', 'meme-0153', 'meme-0175', 'meme-0176', 'meme-0179'})
FILE_ID = re.compile(r'^(meme-\d{4})(?:-.+)?\.gif$', re.IGNORECASE)
BACKGROUND_FILE_ID = re.compile(
    r'^(meme-\d{4})(?:-.+)?-([1-9]|10)\.gif$', re.IGNORECASE)


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
    """Original IDs/audio with a separate, mandatory background GIF directory.

    Only background numbers covering every allowed original meme can be chosen.
    A selected background is never silently replaced when a file disappears.
    """
    def __init__(self, directory, background_directory=None, choice=None):
        self.directory = Path(directory).expanduser().resolve()
        background = (Path(background_directory).expanduser() if background_directory is not None
                      else self.directory.parent / 'meme_gif_background')
        if background.is_symlink():
            raise PlayerError('背景目录不能是符号链接：' + str(background))
        self.background_directory = background.resolve()
        self.lock = threading.RLock()
        self.index = {}
        self.backgrounds = {}
        self.available_background_ids = ()
        self._background_id = None
        self._conversation_id = None
        self._background_revision = 0
        self._background_selections = {}
        self._index_signature = None
        self._choice = choice or random.choice
        self.refresh()
        self._background_id = self._choose_background()

    def safe_file(self, path, directory=None):
        directory = self.directory if directory is None else directory
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise PlayerError('素材文件不能是符号链接或位于指定目录外：' + path.name)
        if not path.is_file():
            raise PlayerError('素材文件不存在：' + path.name)
        return path

    @property
    def background_id(self):
        with self.lock:
            return self._background_id

    @property
    def conversation_id(self):
        with self.lock:
            return self._conversation_id

    @property
    def background_revision(self):
        with self.lock:
            return self._background_revision

    def _choose_background(self, *, exclude_current=False):
        candidates = self.available_background_ids
        if exclude_current and len(candidates) > 1:
            candidates = tuple(value for value in candidates if value != self._background_id)
        selected = self._choice(candidates)
        if type(selected) is not int or selected not in candidates:
            raise PlayerError('随机选择的背景编号不可用。')
        return selected

    def begin_conversation(self, conversation_id, background_revision=0):
        if not isinstance(conversation_id, str) or not conversation_id or len(conversation_id) > 256:
            raise PlayerError('新对话需要有效的conversation_id。')
        if type(background_revision) is not int or background_revision < 0:
            raise PlayerError('background_revision需要是非负整数。')
        with self.lock:
            self.refresh()
            pair = conversation_id, background_revision
            if pair in self._background_selections:
                # Old playback snapshots never move the active revision back.
                return self._validate_background(self._background_selections[pair])
            if conversation_id == self._conversation_id and background_revision < self._background_revision:
                raise PlayerError('对话背景版本已过期，不能倒退。')
            changed_revision = conversation_id == self._conversation_id and background_revision != self._background_revision
            if conversation_id != self._conversation_id or changed_revision:
                self._background_id = self._choose_background(exclude_current=changed_revision)
                self._conversation_id = conversation_id
                self._background_revision = background_revision
                self._background_selections[pair] = self._background_id
                if len(self._background_selections) > 1024:
                    del self._background_selections[next(iter(self._background_selections))]
            self._validate_background(self._background_id)
            return self._background_id

    def _validate_background(self, background_id):
        if type(background_id) is not int or not 1 <= background_id <= 10:
            raise PlayerError('背景编号必须是1到10的整数。')
        if background_id not in self.available_background_ids:
            raise PlayerError('背景编号缺少部分表情，无法保持本轮背景一致：' + str(background_id))
        return background_id

    def _directory_signature(self):
        values = []
        for directory in (self.directory, self.background_directory):
            if directory.is_symlink():
                raise PlayerError('素材目录不能是符号链接：' + str(directory))
            try:
                state = directory.stat()
                names = frozenset(os.listdir(directory))
            except OSError as error:
                raise PlayerError('无法读取素材目录：' + str(directory)
                                  + '；不会回退播放原始GIF。') from error
            values.append((state.st_dev, state.st_ino, state.st_mtime_ns, state.st_ctime_ns, names))
        return tuple(values)

    def refresh(self):
        with self.lock:
            # Listing names also detects rapid changes on filesystems whose
            # directory timestamps have coarse resolution. Avoid resolving and
            # checking every variant unless this lightweight snapshot changes.
            signature = self._directory_signature()
            if self._index_signature == signature and self.index:
                return
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
            try:
                background_paths = sorted(self.background_directory.iterdir())
            except OSError as error:
                raise PlayerError('无法读取带背景表情目录：' + str(self.background_directory)
                                  + '；不会回退播放原始GIF。') from error
            backgrounds = {}
            for path in background_paths:
                match = BACKGROUND_FILE_ID.fullmatch(path.name)
                if not match:
                    continue
                key, background_id = match[1].lower(), int(match[2])
                if key not in index:
                    continue
                self.safe_file(path, self.background_directory)
                pair = (key, background_id)
                if pair in backgrounds:
                    raise PlayerError('同一表情和背景编号存在重复文件：' + key + '-' + str(background_id))
                backgrounds[pair] = path
            for (key, background_id), path in backgrounds.items():
                expected = index[key][0].stem + '-' + str(background_id) + '.gif'
                if path.name != expected:
                    raise PlayerError('带背景GIF名称与原表情不对应：' + path.name + '；应为' + expected)
            complete = tuple(background_id for background_id in range(1, 11)
                             if all((key, background_id) in backgrounds for key in index))
            if not complete:
                raise PlayerError('带背景目录中没有覆盖全部表情的背景编号；不会回退播放原始GIF。')
            self.index, self.backgrounds = index, backgrounds
            self.available_background_ids = complete
            self._index_signature = signature

    def _entry(self, key, background_id):
        if key not in self.index:
            raise PlayerError('编号不在 sucai_meme/meme_gif 中或已剔除：' + str(key))
        background_id = self._validate_background(
            self._background_id if background_id is None else background_id)
        original, audio = self.index[key]
        visual = self.backgrounds[(key, background_id)]
        return original, visual, audio, background_id

    def assets(self, background_id=None):
        with self.lock:
            self.refresh()
            rows = []
            for key in sorted(self.index):
                original, visual, audio, selected = self._entry(key, background_id)
                rows.append({'id':key, 'label':original.stem[len(key):].lstrip('-') or key,
                             'visual_type':'gif', 'duration_seconds':None, 'scene_index':None,
                             'visual_name':visual.name, 'visual_mime':'image/gif',
                             'audio_name':audio.name if audio else None,
                             'audio_mime':'audio/wav' if audio else None,
                             'updated_at':visual.stat().st_mtime_ns,
                             'source_path':str(visual), 'background_id':selected})
            return rows

    def read(self, key, background_id=None):
        with self.lock:
            self.refresh()
            original, visual, audio, selected = self._entry(key, background_id)
            visual_blob = self.safe_file(visual, self.background_directory).read_bytes()
            audio_blob = self.safe_file(audio).read_bytes() if audio else None
            updated_at = visual.stat().st_mtime_ns
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
        return {'id':key, 'label':original.stem[len(key):].lstrip('-') or key,
                'visual_type':'gif','duration_seconds':duration,'scene_index':None,
                'visual_name':visual.name,'visual_mime':'image/gif','visual_blob':visual_blob,
                'audio_name':audio.name if audio else None,'audio_mime':'audio/wav' if audio else None,
                'audio_blob':audio_blob,'updated_at':updated_at,
                'source_path':str(visual), 'background_id':selected}


class DirectoryPlayer(Player):
    """Meme IDs exclusively from the team folder; user imports stay in SQLite."""
    def __init__(self, root, directory, background_directory=None, choice=None):
        self.directory_assets = DirectoryAssets(directory, background_directory, choice)
        super().__init__(root)

    def assets(self):
        return self.directory_assets.assets() + [row for row in super().assets() if row['id'].startswith('user-')]

    def library(self):
        result = super().library()
        allowed = {row['id'] for row in self.assets()}
        result['items'] = [item for item in result['items'] if item['id'] in allowed]
        return result

    @property
    def background_id(self):
        return self.directory_assets.background_id

    @property
    def conversation_id(self):
        return self.directory_assets.conversation_id

    @property
    def background_revision(self):
        return self.directory_assets.background_revision

    def begin_conversation(self, conversation_id, background_revision=0):
        with self.lock:
            return self.directory_assets.begin_conversation(conversation_id, background_revision)

    def respond(self, body):
        if not isinstance(body, dict):
            return super().respond(body)
        with self.lock:
            request_id = body.get('request_id')
            if isinstance(request_id, str) and request_id in self.requests:
                # A cached response retains its own conversation/background,
                # even when a newer conversation is now selected.
                return super().respond(body)
            if 'background_id' in body and body['background_id'] is None:
                raise PlayerError('background_id需要是1到10的整数')
            if 'conversation_id' in body and 'background_id' not in body:
                conversation_id = body['conversation_id']
                if not isinstance(conversation_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', conversation_id):
                    raise PlayerError('conversation_id格式不正确')
                # Validate before selecting a conversation, so an invalid API
                # request cannot change the active background.
                key, text = body.get('meme_id'), body.get('reply_text')
                if not isinstance(key, str) or key not in {row['id'] for row in self.library()['items']}:
                    raise PlayerError('meme_id 不在素材库中')
                if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                    raise PlayerError('reply_text 需要是1到2000字的文本')
                if 'request_id' in body and (not isinstance(request_id, str) or
                        not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request_id)):
                    raise PlayerError('request_id格式不正确')
                revision = body.get('background_revision', 0)
                body = dict(body, background_id=self.begin_conversation(conversation_id, revision),
                            background_revision=revision)
            # An explicit background is an immutable playback snapshot. Replays
            # of older events must not change the current conversation.
            return super().respond(body)

    def read_native_asset(self, key, background_id=None):
        if key.startswith('meme-'):
            return self.directory_assets.read(key, background_id)
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
