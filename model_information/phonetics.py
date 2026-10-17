"""A conservative phonetic guard for Chinese ASR correction candidates.

This guard never repairs text itself. It checks one-to-one Chinese character
positions using the existing ASR candidate; upstream guards still protect
negation, emotional meaning, names, Latin text and numbers. A pinyin match is
necessary for acceptance here, and does not by itself prove the same meaning.
"""

from __future__ import annotations

import re


_CHINESE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\U00020000-\U0002ee5f\U00030000-\U000323af]"
)
_SYLLABLE = re.compile(r"[a-z]+\Z")


def _syllable(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.lower().replace("ü", "v").replace("u:", "v")
    return value if _SYLLABLE.fullmatch(value) is not None else ""


def _same_or_close(left: str, right: str) -> bool:
    """Allow equality or one common accent feature, without combining them."""
    if left == right:
        return True
    if left.startswith("n") and right.startswith("l") and left[1:] == right[1:]:
        return True
    if left.startswith("l") and right.startswith("n") and left[1:] == right[1:]:
        return True
    for retroflex, plain in (("zh", "z"), ("ch", "c"), ("sh", "s")):
        if left.startswith(retroflex) and right.startswith(plain) and left[len(retroflex):] == right[len(plain):]:
            return True
        if right.startswith(retroflex) and left.startswith(plain) and right[len(retroflex):] == left[len(plain):]:
            return True
    for short, long in (("en", "eng"), ("in", "ing")):
        if left.endswith(short) and right == left[:-len(short)] + long:
            return True
        if right.endswith(short) and left == right[:-len(short)] + long:
            return True
    return False


def phonetic_edit_reason(original: str, candidate: str) -> str:
    """Return an empty string for a compatible edit, otherwise a safe reason.

    Non-Chinese characters, whitespace and punctuation do not participate in
    this phonetic check. Chinese insertions/deletions are rejected before a
    dependency is loaded. Identical Chinese sequences need no pinyin lookup.
    Unknown characters, dependency errors and malformed lookup output fail
    closed instead of raising or silently accepting an uncertain edit.
    """
    if not isinstance(original, str) or not isinstance(candidate, str):
        return "phonetic_invalid_text"
    before, after = _CHINESE.findall(original), _CHINESE.findall(candidate)
    if len(before) != len(after):
        return "phonetic_alignment_changed"
    changed = [index for index, (left, right) in enumerate(zip(before, after)) if left != right]
    if not changed:
        return ""
    try:
        from pypinyin import lazy_pinyin
    except Exception:
        return "phonetic_dependency_unavailable"
    try:
        # Whole phrases retain dictionary disambiguation of polyphonic words;
        # only changed character positions are compared afterward. The default
        # lazy_pinyin style is NORMAL, which intentionally omits tone marks.
        left_pinyin = lazy_pinyin("".join(before))
        right_pinyin = lazy_pinyin("".join(after))
    except Exception:
        return "phonetic_lookup_failed"
    if not isinstance(left_pinyin, (list, tuple)) or not isinstance(right_pinyin, (list, tuple)):
        return "phonetic_lookup_failed"
    if len(left_pinyin) != len(before) or len(right_pinyin) != len(after):
        return "phonetic_lookup_failed"
    for index in changed:
        left, right = _syllable(left_pinyin[index]), _syllable(right_pinyin[index])
        if not left or not right:
            return "phonetic_lookup_failed"
        if not _same_or_close(left, right):
            return "phonetic_mismatch"
    return ""
