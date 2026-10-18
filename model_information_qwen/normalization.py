"""Conservative ASR text repair; candidates never silently overwrite intent."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import math
import re

from .protocol import validate_text
from .phonetics import phonetic_edit_reason
from .role_names import ROLE_ALIASES, canonicalize_role_name


@dataclass(frozen=True)
class NormalizedInput:
    text: str
    status: str = "unchanged"
    clarification: str = ""
    reason: str = ""


_NUMBERS = re.compile(r"\d+(?:[./:\-]\d+)*|[零〇一二三四五六七八九十百千万亿两]+")
_NEGATIONS = re.compile(r"不是不|并非不|没有|不是|不想|不愿|不能|不会|不要|不|没|未|别|勿|无")
_INTENSITY = re.compile(r"有一点|有点|一点点|一点|非常|特别|十分|超级|极其|稍微|不太|还算|从来|总是|完全|很|太|挺")
_SURNAMES = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
_NAMES = re.compile(r"[" + _SURNAMES + r"][\u4e00-\u9fff]{1,2}(?=今天|昨天|明天|刚才|说|去|来|在|不|没|很|的|[，。！？,!?]|$)")
_PUNCTUATION = re.compile(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()…—\-]")
_EMOTIONS = ("不开心", "不想聊天", "开心", "高兴", "快乐", "喜欢", "讨厌", "难过", "难受", "伤心",
             "害怕", "焦虑", "紧张", "生气", "愤怒", "孤独", "委屈", "失落", "绝望", "想死", "自杀", "伤害自己")


def can_acknowledge_without_details(text):
    """An explicit feeling can be acknowledged without guessing its cause.

    An unresolved main action/object is different: do not suppress its needed
    clarification merely because the request also contains an emotion word.
    """
    if re.search(r"帮我|帮忙|请|替我|给我|处理|取消|去.{0,4}一下", text):
        return False
    feelings = (*_EMOTIONS, "压力", "鸭力", "疲惫", "很累", "太累", "好累", "真烦", "很烦", "么么哒", "谢谢")
    return any(word in text for word in feelings)


def unsafe_edit_reason(original: str, candidate: str) -> str:
    """Heuristic preservation checks, not a claim of complete semantic proof."""
    if _NUMBERS.findall(original) != _NUMBERS.findall(candidate):
        return "numbers_changed"
    # Unconverted aliases are ambiguous/protected entities. Only the narrow
    # deterministic role-name step may convert them to the canonical name.
    if any(original.count(alias) != candidate.count(alias) for alias in ROLE_ALIASES if alias in original):
        return "entity_changed"
    compact_original = re.sub(r"\s+", "", original)
    compact_candidate = re.sub(r"\s+", "", candidate)
    if _NEGATIONS.findall(compact_original) != _NEGATIONS.findall(compact_candidate):
        return "negation_changed"
    if _INTENSITY.findall(compact_original) != _INTENSITY.findall(compact_candidate):
        return "intensity_changed"
    for word in _EMOTIONS:
        if word in original and candidate.count(word) != original.count(word):
            return "emotion_changed"
    for pronoun in ("我", "你", "他", "她", "它"):
        if candidate.count(pronoun) != original.count(pronoun):
            return "person_changed"
    if [s.lower() for s in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", original)] != [s.lower() for s in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", candidate)]:
        return "entity_changed"
    # Explicit quoted entities and common personal nicknames remain literal.
    quoted = re.findall(r'["“「『]([^"”」』]+)["”」』]', original)
    nicknames = re.findall(r"(?:小|老)[\u4e00-\u9fff]", original)
    names = _NAMES.findall(original)
    if any(candidate.count(entity) != original.count(entity) for entity in quoted + nicknames + names):
        return "entity_changed"
    before, after = _PUNCTUATION.sub("", original), _PUNCTUATION.sub("", candidate)
    changed = [(i1, i2, j1, j2) for op, i1, i2, j1, j2 in SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if op != "equal"]
    if any(re.search(r"[不没未别勿无]", before[i1:i2] + after[j1:j2]) for i1, i2, j1, j2 in changed):
        return "negation_moved"
    edits = sum(max(i2 - i1, j2 - j1) for i1, i2, j1, j2 in changed)
    if edits > max(2, math.ceil(.15 * len(before))):
        return "too_many_edits"
    return phonetic_edit_reason(original, candidate)


def parse_repair(output: str, original: str) -> NormalizedInput:
    """Reject malformed/overreaching repairs and retain the original utterance."""
    try:
        if not isinstance(output, str):
            raise ValueError("not text")
        payload = json.loads(output.strip())
        if not isinstance(payload, dict) or set(payload) != {"status", "text", "clarification"}:
            raise ValueError("wrong schema")
        status, text, question = payload["status"], validate_text(payload["text"]), payload["clarification"]
        if status not in ("unchanged", "corrected", "uncertain") or not isinstance(question, str):
            raise ValueError("wrong status")
        if status in ("unchanged", "uncertain") and text != original:
            return NormalizedInput(original, "fallback", reason="original_not_preserved")
        if status == "uncertain":
            if can_acknowledge_without_details(original):
                return NormalizedInput(original, "unchanged", reason="nonessential_ambiguity")
            question = question.strip()
            question = re.sub(r"^(?:需要澄清|澄清问题|clarification)\s*[:：]\s*", "", question, flags=re.IGNORECASE)
            if not question or len(question) > 120 or "\n" in question or not question.endswith(("？", "?")):
                question = "这句话我有点没听清，你能换个说法再说一遍吗？"
            return NormalizedInput(original, status, question)
        if question:
            raise ValueError("unexpected question")
        if status == "corrected":
            reason = unsafe_edit_reason(original, text)
            if reason:
                return NormalizedInput(original, "fallback", reason=reason)
            return NormalizedInput(text, "corrected" if text != original else "unchanged")
        return NormalizedInput(original)
    except (ValueError, TypeError, RecursionError):
        return NormalizedInput(original, "fallback", reason="invalid_repair")


class ASRNormalizer:
    def __init__(self, generator, system_prompt, *, max_new_tokens=192, history_messages=6, enabled=True):
        self.generator = generator
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.history_messages = history_messages
        self.enabled = enabled

    def normalize(self, messages):
        received = validate_text(messages[-1]["content"])
        history = messages[:-1]
        try:
            original = validate_text(canonicalize_role_name(received, history))
        except Exception:
            # Do not let a name expansion exceed the request's text contract.
            original = received
        name_changed = original != received
        if not self.enabled:
            return NormalizedInput(original, "corrected" if name_changed else "unchanged",
                reason="role_name:disabled" if name_changed else "disabled")
        history = history[-self.history_messages:] if self.history_messages else []
        # One JSON data document, not synthetic A/B turns or executable text.
        data = json.dumps({"history": history, "text": original}, ensure_ascii=False)
        try:
            output = self.generator.complete(self.system_prompt, [{"role": "user", "content": data}],
                max_new_tokens=self.max_new_tokens, do_sample=False, no_repeat_ngram_size=0)
            result = parse_repair(output, original)
        except Exception:
            # A failed editor must not prevent a readable user's conversation.
            result = NormalizedInput(original, "fallback", reason="repair_model_failed")
        if name_changed:
            # Model failure cannot undo an already verified role-name alias.
            # Uncertainty about the remaining sentence still asks for clarity.
            status = "corrected" if result.status == "unchanged" else result.status
            reason = "role_name" + (":" + result.reason if result.reason else "")
            return NormalizedInput(result.text, status, result.clarification, reason)
        return result
