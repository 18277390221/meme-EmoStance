"""Conservative, addressed standby commands without model or network calls.

Recognition is deliberately independent of player state.  A caller must only
act on a wake command while its standby state is active; an awake player keeps
the same utterance in ordinary conversation.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Literal


Command = Literal["sleep", "wake"]
ADDRESS_ALIASES = ("小meme", "小me", "meme", "小猫", "小咪", "小米", "小美", "小梅", "小妹")

# Preserve the difference between a direct command and a quoted/questioned
# command before removing punctuation and spacing for normal ASR variation.
_QUOTED_OR_QUESTION = re.compile(r'''[?？"'“”‘’「」『』《》‹›«»]''')
_ADDRESS = "(?:" + "|".join(re.escape(name) for name in ADDRESS_ALIASES) + ")"
_FRONT_ADDRESS = re.compile(r"^(?:喂|嘿|嗨)?" + _ADDRESS + r"(?:啊|呀|哦)?(?P<body>.+)$")
_BACK_ADDRESS = re.compile(r"^(?P<body>.+?)" + _ADDRESS + r"(?:啊|呀|哦)?$")

# Restrict optional politeness and urgency to a small grammar.  In particular,
# no arbitrary leading/trailing words are allowed to create a substring match.
_PREFIX = r"(?:你)?(?:请你|请|麻烦你)?(?:现在|赶紧|快点|快|先)?"
_SLEEP_BODY = (
    r"(?:睡觉去(?:吧|啦)?|去睡觉(?:吧|啦)?|睡觉(?:吧|啦|了)?"
    r"|该(?:去)?睡觉了|睡吧|去睡吧|休息吧|去休息吧|该休息了"
    r"|进入待机(?:模式)?|待机吧)"
)
_WAKE_BODY = (
    r"(?:别睡(?:觉)?了|不要睡(?:觉)?了|醒醒(?:吧|啦)?"
    r"|醒过来(?:吧|了)?|醒来吧|起床(?:了|啦|吧)?|该起床了|起来吧)"
)
_SLEEP = re.compile(r"^" + _PREFIX + _SLEEP_BODY + r"$")
# A common natural wake command contains both "别睡了" and "起床了".
# At most three direct wake clauses, with an optional "快/赶紧" per clause,
# are accepted.  Conditions, negations and explanatory tails cannot match.
_WAKE = re.compile(r"^" + _PREFIX + _WAKE_BODY + r"(?:" + _PREFIX + _WAKE_BODY + r"){0,2}$")


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def classify_command(text: object) -> Command | None:
    """Return a direct, named sleep/wake command, otherwise ``None``.

    Both name-first commands ("小me，睡觉。") and the installed legacy
    name-last command ("去睡觉吧meme") are supported.  An unaddressed "睡觉" is
    ordinary speech because it does not identify this player as the addressee.
    """
    if not isinstance(text, str) or not text or _QUOTED_OR_QUESTION.search(text):
        return None
    normalized = _normalize(text)
    address = _FRONT_ADDRESS.fullmatch(normalized) or _BACK_ADDRESS.fullmatch(normalized)
    if address is None:
        return None
    body = address.group("body")
    if _SLEEP.fullmatch(body):
        return "sleep"
    if _WAKE.fullmatch(body):
        return "wake"
    return None


def is_sleep_command(text: object) -> bool:
    """Whether text directly asks the named player to enter standby."""
    return classify_command(text) == "sleep"


def is_wake_command(text: object) -> bool:
    """Whether text directly asks the named player to wake (check state too)."""
    return classify_command(text) == "wake"
