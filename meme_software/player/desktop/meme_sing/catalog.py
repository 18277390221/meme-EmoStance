"""Read-only local music catalogue, retaining filename emotion/style labels."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import re

SUPPORTED_SUFFIXES = frozenset({'.mp3', '.wav', '.ogg', '.flac'})
# One delivered filename omits '_' between its title and Chinese mood labels.
# Anchoring to the end handles that file without parsing an artist/title as tags.
TAG_TAIL = re.compile(r'([\u4e00-\u9fff]+\d*(?:[，,、][\u4e00-\u9fff]+\d*)*)$')
TAG = re.compile(r'([\u4e00-\u9fff]+)(\d*)$')


@dataclass(frozen=True)
class Track:
    id: str
    title: str
    filename: str
    path: Path
    tags: dict[str, float]


def filename_metadata(filename: str) -> tuple[str, dict[str, float]]:
    """Strip the nonunique numeric suffix and normalize optional tag weights."""
    stem = Path(filename).stem
    body = re.sub(r'_\d+$', '', stem)
    match = TAG_TAIL.search(body)
    if match is None:
        return body or stem, {}
    tags: dict[str, float] = {}
    for item in re.split(r'[，,、]', match[1]):
        label = TAG.fullmatch(item)
        if label is None:
            raise ValueError('音乐文件名中的情绪标签格式无效：' + filename)
        weight = float(label[2]) if label[2] else 1.0
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('音乐文件名中的情绪权重无效：' + filename)
        tags[label[1]] = tags.get(label[1], 0.0) + weight
    total = sum(tags.values())
    if total <= 0:
        raise ValueError('音乐文件名中的情绪权重不能全部为零：' + filename)
    title = body[:match.start()].rstrip('_').strip()
    # A plain Chinese title (no separator, weights or tag list) is a title,
    # rather than metadata. Delivered tagged files all have one of those clues.
    if not title or (not body[:match.start()].endswith('_') and not re.search(r'\d|[，,、]', match[1])):
        return body or stem, {}
    return title, {key: weight / total for key, weight in tags.items()}


class MusicCatalog:
    def __init__(self, directory):
        original = Path(directory).expanduser()
        if original.is_symlink():
            raise ValueError('音乐目录不能是符号链接。')
        try:
            self.directory = original.resolve(strict=True)
            if not self.directory.is_dir():
                raise ValueError('音乐路径不是目录。')
            paths = sorted(self.directory.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise ValueError('无法读取本地音乐目录：' + str(original)) from error
        tracks = []
        index = {}
        for path in paths:
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            self._validate_file(path)
            title, tags = filename_metadata(path.name)
            identity = 'music-' + hashlib.sha256(os.fsencode(path.name)).hexdigest()[:24]
            if identity in index:
                raise ValueError('音乐编号重复，无法确定对应文件。')
            track = Track(identity, title, path.name, path, tags)
            tracks.append(track)
            index[identity] = track
        if not tracks:
            raise ValueError('本地音乐目录没有可用的MP3/WAV/OGG/FLAC文件。')
        self.tracks = tuple(tracks)
        self._index = index

    def _validate_file(self, path: Path) -> None:
        try:
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.directory):
                raise ValueError('音乐文件不能是符号链接或位于音乐目录外：' + path.name)
            if not path.is_file() or path.stat().st_size <= 0:
                raise ValueError('音乐文件不是可用的非空普通文件：' + path.name)
        except OSError as error:
            raise ValueError('本地音乐文件不可读：' + path.name) from error

    def resolve(self, track_id: str) -> Track:
        if not isinstance(track_id, str) or track_id not in self._index:
            raise ValueError('歌曲编号不在本地音乐目录中。')
        track = self._index[track_id]
        self._validate_file(track.path)
        return track
