"""Local music intents and transparent mood rules; no model or network calls.

Only successful chat pairs are observed. The server's conversation_id owns
reset boundaries; this module never guesses a new dialogue from local time.
Old evidence can expire for selection without changing that identity.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
import random
import re
import time
import unicodedata
from typing import Any


CONTEXT_TTL_SECONDS = 120.0
FAMILIES = {
    "欢乐": ("高兴", "开心", "快乐", "愉悦", "欢乐", "欣喜", "喜悦", "欢快", "搞笑", "滑稽", "呆萌", "庆祝", "兴奋", "幸福", "期待"),
    "悲伤": ("悲伤", "伤感", "忧伤", "忧郁", "难过", "伤心", "低落", "难受", "沮丧", "痛苦", "失落", "绝望", "孤独", "委屈", "寂寞", "心情不好"),
    "轻松": ("轻松", "放松", "休闲", "平静", "安静", "舒缓", "温柔", "治愈", "舒适", "安心", "宁静"),
    "怀念": ("怀念", "怀旧", "思念", "回忆", "想念"),
    "浪漫": ("浪漫", "爱情", "恋爱", "甜蜜", "心动", "表白"),
    "活力": ("活力", "热情", "激昂", "动感", "激情", "热血", "元气", "振奋", "鼓劲"),
    "紧张": ("紧张", "蓄势待发", "焦虑", "着急", "压力", "烦躁", "愤怒", "生气"),
    "自信": ("自傲", "酷", "坚定", "自信", "耍帅", "强大", "帅气"),
    "沉思": ("沉思", "思考", "认真", "专注", "迷茫", "沉重", "反思"),
    "中性": ("中性", "普通", "日常", "万能表情"),
}
_FAMILY = {word: family for family, words in FAMILIES.items() for word in words}
_MOOD_WORD = re.compile("|".join(map(re.escape, sorted(_FAMILY, key=len, reverse=True))))
_NEGATION = re.compile(r"(?:不是|并非|并不|不再|不|没|没有|别|不要|不想|不会|毫不|不怎么|并不是)(?:太|很|那么|怎么|觉得|感到|感觉|说|听|放|播放|听点|听首|听些|听一首)?$")
_QUOTED = re.compile(r"[\"'“”‘’「」『』《》]")
_CONDITION = re.compile(r"如果|要是|假如|除非|一旦|直到|(?:等|等到).*(?:再|就)|什么时候")
_MENTION = re.compile(r"(?:说|提到|提及|引用|读出|念出|叫做|叫|词|意思|听到|歌词|歌名).*(?:停止|停|暂停|别放了|不听了)")
_STOP_NEGATION = re.compile(r"(?:不要|不想|不能|不用|不必|别|不准|没有|没说|不是)(?:再|继续|去|让|给我|把|音乐|歌曲|歌|它|说|现在|马上|随便)*(?:停止|停|暂停)|继续(?:播放|放|听)")
_STOP_PREFIX = r"(?:(?:请你|请|麻烦|帮我|给我|我想要|我想|我要|小猫|meme猫|meme|现在|马上|快|赶紧|先|能不能|可以|能))*"
_MUSIC_OBJECT = r"(?:(?:这|那)?一?(?:首|段)?(?:音乐|歌曲|歌))"
_STOP_COMMAND = (r"(?:停止(?:播放)?" + _MUSIC_OBJECT + r"?(?:一下)?"
                 r"|停(?:下来|下|一下|一会儿|一会)?" + _MUSIC_OBJECT + r"?"
                 r"|暂停(?:播放)?" + _MUSIC_OBJECT + r"?(?:一下|一会儿)?"
                 r"|我?(?:别|不要|不用|不想)(?:放|播|听|唱)" + _MUSIC_OBJECT + r"?了?"
                 r"|不听" + _MUSIC_OBJECT + r"?了|关(?:掉|闭)?" + _MUSIC_OBJECT +
                 r"|把" + _MUSIC_OBJECT + r"(?:停(?:下|了|一下)?|关(?:掉|了)?|关闭)"
                 r"|" + _MUSIC_OBJECT + r"(?:停一下|暂停一下))")
_STOP_END = r"(?:吧|啊|呀|啦|好吗|行吗|谢谢|了|哦)*$"
_STOP_FULL = re.compile(r"^" + _STOP_PREFIX + _STOP_COMMAND + _STOP_END)
_STOP_SUFFIX = re.compile(_STOP_PREFIX + _STOP_COMMAND + _STOP_END)


def _text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"[\s，。！？、,.!?；;：:]+", "", unicodedata.normalize("NFKC", value).lower()).strip()


def is_music_request(text) -> bool:
    """Recognize a request to hear music, excluding refusals and discussion."""
    text = _text(text)
    if not text or _QUOTED.search(text) or _CONDITION.search(text):
        return False
    # A stop/close/pause sentence must reach ordinary chat when no song is
    # playing; the embedded word "播放" must never turn it into a song request.
    if is_stop_request(text) or _STOP_NEGATION.search(text):
        return False
    if re.search(r"你(?:喜欢|想|爱|会)|(?:音乐|唱歌|听歌)(?:是什么|什么意思)|(?:说|提到|提及|引用).*(?:听歌|音乐|放首歌)", text):
        return False
    requests = re.finditer(r"(?:想听|要听|听(?:一|首|点|些|个)?|来(?:一|首|点|些)|放|播|唱).*?(?:音乐|歌曲|歌|曲)", text)
    for match in requests:
        before = text[max(0, match.start() - 12):match.start()]
        if not re.search(r"(?:不想|不要|不|别|不能|不用)(?:给我|帮我|再|继续|去|让|把)*$", before):
            return True
    return False


def is_stop_request(text, partial=False) -> bool:
    """Classify music controls only; the caller must first check playback state."""
    text = _text(text)
    if not text or _QUOTED.search(text) or _CONDITION.search(text) or _MENTION.search(text) or _STOP_NEGATION.search(text):
        return False
    if _STOP_FULL.fullmatch(text):
        return True
    if not partial:
        return False
    match = _STOP_SUFFIX.search(text)
    if match is None:
        return False
    # A statement about something having stopped is not a stop command.
    return not re.search(r"(?:已经|刚才|会|曾经|歌曲|音乐|机器|心跳|火车|运动|工作|雨|风)$", text[:match.start()])


def _moods(text):
    text = _text(text)
    scores = Counter()
    for match in _MOOD_WORD.finditer(text):
        word = match.group()
        family = _FAMILY[word]
        before = text[max(0, match.start() - 12):match.start()]
        if _NEGATION.search(before) and not re.search(r"(?:不是|并非|没有)不$", before):
            # Explicit unhappiness carries evidence; other negations merely
            # remove the quoted positive/negative label instead of inverting it.
            if family == "欢乐":
                scores["悲伤"] += 1
            continue
        scores[family] += 1
    return {family: min(2., float(score)) for family, score in scores.items()}


def _normalized(scores):
    total = sum(scores.values())
    return {key: value / total for key, value in scores.items() if value > 0} if total else {}


def _tag_moods(tags):
    scores = Counter()
    items = tags.items() if isinstance(tags, dict) else ((tag, 1.) for tag in tags)
    for tag, weight in items:
        if not isinstance(tag, str) or isinstance(weight, bool) or not isinstance(weight, (float, int)) or not math.isfinite(weight) or weight <= 0:
            continue
        family = _FAMILY.get(tag)
        if family:
            scores[family] += float(weight)
    return _normalized(scores)


@dataclass(frozen=True)
class Selection:
    track: Any
    moods: dict[str, float]
    reason: str


class MusicContext:
    """Bounded successful dialogue evidence, interpreted by local rules only."""
    def __init__(self, max_turns=8, clock=time.monotonic):
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.max_turns, self.clock = max_turns, clock
        self.conversation_id = None
        self._turns = deque(maxlen=max_turns)
        self._observed_at = None
        self._last_track_id = None

    def observe_reply(self, conversation_id, user_text, reply_text, meme_id=None, meme_tags=()):
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError("conversation_id is required for successful dialogue evidence")
        if not isinstance(user_text, str) or not isinstance(reply_text, str):
            raise TypeError("successful user/reply text must be strings")
        if conversation_id != self.conversation_id:
            self.conversation_id = conversation_id
            self._turns.clear()
            self._observed_at = None
        if is_music_request(user_text) or is_stop_request(user_text):
            return
        # meme_id is only an identity hint. Optional human catalog tags may
        # support the texts; it is never treated as a planner mood label.
        self._turns.append((user_text, reply_text, _tag_moods(meme_tags or ())))
        self._observed_at = self.clock()

    def choose(self, catalog, request_text) -> Selection:
        tracks = tuple(catalog.tracks)
        if not tracks:
            raise ValueError("本地音乐目录没有可播放歌曲。")
        desired = _moods(request_text)
        reason = "本地规则：按本次点歌指定的风格选择。"
        if not desired:
            evidence = Counter()
            if self._observed_at is not None and self.clock() - self._observed_at < CONTEXT_TTL_SECONDS:
                for age, (user, reply, tags) in enumerate(reversed(self._turns)):
                    decay = .72 ** age
                    for source, weight in ((_moods(user), 2.), (_moods(reply), .65), (tags, .4)):
                        for mood, score in source.items():
                            evidence[mood] += decay * weight * score
            desired = dict(evidence)
            reason = "本地规则：按最近成功聊天的文字及素材标签选择。"
        moods = _normalized(desired) or {"中性": 1.}
        candidates = [track for track in tracks if track.id != self._last_track_id] or list(tracks)
        scored = [(sum(moods.get(mood, 0.) * value for mood, value in _tag_moods(track.tags).items()), track)
                  for track in candidates]
        best = max(score for score, _ in scored)
        if best <= 0:
            fallback = {"轻松": .6, "欢乐": .3, "中性": .1}
            scored = [(sum(fallback.get(mood, 0.) * value for mood, value in _tag_moods(track.tags).items()), track)
                      for track in candidates]
            best = max(score for score, _ in scored)
            reason = "本地规则：没有可用匹配，优先选择轻松、愉悦或中性歌曲。"
        track = random.choice([track for score, track in scored if math.isclose(score, best, rel_tol=1e-9, abs_tol=1e-12)])
        self._last_track_id = track.id
        return Selection(track, dict(moods), reason)


class StopCommandDetector:
    """Debounce local control partials; one signal until the next reset."""
    def __init__(self, interval=.2, stable_hits=2):
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        if isinstance(stable_hits, bool) or not isinstance(stable_hits, int) or stable_hits < 1:
            raise ValueError("stable_hits must be a positive integer")
        self.interval, self.stable_hits = float(interval), stable_hits
        self.reset()

    def reset(self):
        self._last_checked = None
        self._hits = 0
        self._triggered = False

    def feed(self, text, now=None) -> bool:
        now = time.monotonic() if now is None else now
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError("now must be finite monotonic seconds")
        if self._triggered:
            return False
        if not is_stop_request(text, partial=True):
            self._hits = 0
            return False
        if self._last_checked is not None:
            elapsed = now - self._last_checked
            if elapsed < self.interval - 1e-12:
                return False
            if elapsed > max(1., self.interval * 3):
                self._hits = 0
        self._last_checked = now
        self._hits += 1
        self._triggered = self._hits >= self.stable_hits
        return self._triggered
