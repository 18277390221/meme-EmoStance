"""Normalize approved voice spellings only when they address this assistant.

The fixed role name is application configuration, not an arbitrary entity
correction. Protected quotations, code, pets, people and naming intentions
take precedence. Uncertain mentions remain literal for the ASR editor's
ordinary entity guards; only the approved occurrence becomes a new baseline.
"""

from __future__ import annotations

from collections.abc import Mapping
import re


ROLE_NAME = "meme"
# The ASR guard imports these legacy/Chinese aliases to protect ambiguous mentions
# which this narrow canonicalizer intentionally leaves unchanged.
ROLE_ALIASES = ("meme猫", "咪咪猫", "米米猫", "妹妹猫", "美美猫", "迷迷猫", "梅梅猫", "米姆猫", "咪姆猫")

_LATIN_NAME = (
    r"(?<![A-Za-z0-9_])m[^\S\r\n]*e[^\S\r\n]*m[^\S\r\n]*e"
    r"(?:[^\S\r\n]*猫)?(?![A-Za-z0-9_]|[^\S\r\n]*猫)"
)
# The legacy Latin form must use the same identifier boundaries as bare meme;
# a plain alternative for meme猫 would also match the middle of xmeme猫.
_NAME = re.compile("|".join(re.escape(name) for name in ROLE_ALIASES if name != "meme猫") + "|" + _LATIN_NAME, re.IGNORECASE)
_PROTECTED = re.compile(
    r"```[\s\S]*?(?:```|$)|`[^`\n]*(?:`|$)|"
    r'"(?:\\.|[^"\\])*?(?:"|$)|'
    r"'(?:\\.|[^'\\])*?(?:'|$)|"
    r"“[^”]*?(?:”|$)|‘[^’]*?(?:’|$)|「[^」]*?(?:」|$)|『[^』]*?(?:』|$)"
)
_BOUNDARY = re.compile(r"[，,。.!！？?；;\n]")
_EDGE_PUNCTUATION = " \t\r\n\u3000，,。.!！？?；;：:、"
_ADDRESS_TAIL = re.compile(
    r"^(?:你|您|你好|您好|早安|晚安|在吗|在不在|陪我|帮我|听我|跟我|和我|请|我|"
    r"别|不要|能(?:不能)?(?:帮|陪|听)|可以(?:帮|陪|听))"
)
_GREETING_PREFIX = re.compile(r"(?:你好|您好|早安|晚安|谢谢(?:你)?|嗨|哈喽)[，,！!\s]*$")
_SELF_REFERENCE = re.compile(r"(?:你(?:是|叫)|你的名字(?:是|叫)|我(?:该|可以|能)?叫你)\s*$")
_REFERENCE_PREFIX = re.compile(r"(?:和|跟|找|问|谢谢|需要|想念)\s*$")
_PET_NAME_QUESTION = re.compile(r"(?:猫|宠物).{0,12}(?:叫什么|叫啥|名字|取名|起名|命名)")
_PERSON_NAME_QUESTION = re.compile(r"(?:你|他|她|朋友|同学|那个人).{0,8}(?:叫什么|叫啥|名字|网名|昵称)|(?:怎么|如何)称呼(?:你|他|她)")
_ASSISTANT_SELF_NAME_QUESTION = re.compile(r"我叫什么|我叫啥|我的名字|怎么称呼我")
_EXPLICIT_ASSISTANT_NAME = re.compile(
    r"(?:我(?:是|叫)|我的名字(?:是|叫))\s*[“\"‘']?" + _LATIN_NAME,
    re.IGNORECASE,
)
_ENTITY_PREFIX = re.compile(
    r"(?:家(?:的)?|养的|宠物(?:叫)?|同学(?:叫)?|朋友(?:叫)?|邻居(?:叫)?|"
    r"(?:小猫|猫咪|猫)(?:叫|名字叫|取名叫|起名叫))\s*$"
)
_LITERAL_PREFIX = re.compile(r"(?:原样输出|原样写|打印|输出|写下|写成|字符串|变量|字段|代码|文件|网名|昵称)\s*$", re.IGNORECASE)
_NAMING_PREFIX = re.compile(r"(?:取名|起名|命名|改名|改叫).{0,12}$")
_RENAME_PREFIX = re.compile(r"(?:以后|从今以后|从现在起).{0,8}你.{0,5}叫\s*$")
_ENTITY_SUFFIX = re.compile(
    r"^(?:是(?:我|你|他|她|朋友|同学|邻居|电影|游戏|书里)|"
    r"(?:是|的)?.{0,8}(?:网名|艺名|笔名|同学|邻居|宠物|品种|猫粮|猫砂|驱虫|绝育|疫苗)|"
    r"我(?:的|家)(?:宠物|猫|宝贝))"
)
_REPORTED_PREFIX = re.compile(r"(?:他说|她说|同学说|朋友说|书上|电影里|游戏里|提到)[^。！？!?\n]{0,18}$")


def _history(history) -> list[tuple[str, str]]:
    if history is None or isinstance(history, (str, bytes, Mapping)):
        return []
    try:
        rows = list(history)
    except TypeError:
        return []
    result = []
    for row in rows[-8:]:
        if not isinstance(row, Mapping):
            continue
        role = {"user": "user", "A": "user", "assistant": "assistant", "B": "assistant"}.get(row.get("role"))
        content = row.get("content", row.get("text"))
        if role is not None and isinstance(content, str):
            result.append((role, content))
    return result


def _local_context(text: str, start: int, end: int) -> tuple[str, str]:
    left = _BOUNDARY.split(text[:start])[-1]
    right = _BOUNDARY.split(text[end:], maxsplit=1)[0]
    return left, right


def _protected_mention(text: str, start: int, end: int) -> bool:
    left, right = _local_context(text, start, end)
    if _ENTITY_PREFIX.search(left) or _LITERAL_PREFIX.search(left):
        return True
    if _NAMING_PREFIX.search(left) or _RENAME_PREFIX.search(left):
        return True
    if _ENTITY_SUFFIX.search(right.lstrip()):
        return True
    if _REPORTED_PREFIX.search(text[:start]):
        return True
    # A human introducing their own name is not naming the assistant.
    if re.search(r"(?:我|他|她|朋友|同学)(?:的名字)?(?:是|叫)\s*$", left):
        return True
    return False


def canonicalize_role_name(text: str, history=None) -> str:
    """Return text with only clearly addressed, whitelisted names normalized."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    mentions = list(_NAME.finditer(text))
    if not mentions:
        return text
    protected = [(match.start(), match.end()) for match in _PROTECTED.finditer(text)]
    recent = _history(history)
    last_assistant = next((content for role, content in reversed(recent) if role == "assistant"), "")
    literal_name_answer = bool(_PET_NAME_QUESTION.search(last_assistant)) or bool(
        _PERSON_NAME_QUESTION.search(last_assistant) and not _ASSISTANT_SELF_NAME_QUESTION.search(last_assistant)
    )
    assistant_identified = any(role == "assistant" and _EXPLICIT_ASSISTANT_NAME.search(content) for role, content in recent)
    changes = []
    for match in mentions:
        start, end = match.span()
        if any(start < stop and end > beginning for beginning, stop in protected):
            continue
        if _protected_mention(text, start, end):
            continue
        before, after = text[:start], text[end:]
        isolated = not before.strip(_EDGE_PUNCTUATION) and not after.strip(_EDGE_PUNCTUATION)
        if isolated:
            if not literal_name_answer:
                changes.append((start, end))
            continue
        local_left, _ = _local_context(text, start, end)
        # Explicitly identifying "you" is stronger than an unqualified entity
        # mention, but naming/renaming intentions were already protected above.
        if _SELF_REFERENCE.search(local_left):
            changes.append((start, end))
            continue
        tail = after.lstrip(_EDGE_PUNCTUATION)
        address_position = not before.strip(_EDGE_PUNCTUATION) or bool(_GREETING_PREFIX.search(before))
        if address_position and (_ADDRESS_TAIL.match(tail) or _GREETING_PREFIX.search(before)):
            changes.append((start, end))
            continue
        if assistant_identified and _REFERENCE_PREFIX.search(local_left):
            changes.append((start, end))
    if not changes:
        return text
    parts = []
    previous = 0
    for start, end in changes:
        parts.extend((text[previous:start], ROLE_NAME))
        previous = end
    parts.append(text[previous:])
    return "".join(parts)
