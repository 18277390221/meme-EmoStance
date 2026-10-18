"""Business payloads inside the existing communication TCP envelopes."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


MAX_TEXT_LENGTH = 16000
_MEME_ID = re.compile(r"meme-[0-9]{4}\Z")


class ValidationError(ValueError):
    """A request or model result does not match the public business format."""


def validate_text(text: Any) -> str:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_LENGTH:
        raise ValidationError("文字必须是 1–16000 字符的非空字符串")
    return text


def validate_request_id(request_id: Any) -> str:
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ValidationError("请求 id 必须是 1–128 字符的字符串")
    return request_id


def validate_session_id(session_id: Any) -> str:
    if (
        not isinstance(session_id, str)
        or not session_id.strip()
        or len(session_id) > 128
        or not session_id.isprintable()
    ):
        raise ValidationError("session_id 必须是 1–128 个可打印字符的非空字符串")
    return session_id


def validate_reply(payload: Any) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ValidationError("模型返回必须是 JSON 对象")
    reply = validate_text(payload.get("reply"))
    meme_id = payload.get("meme_id")
    if not isinstance(meme_id, str) or _MEME_ID.fullmatch(meme_id) is None:
        raise ValidationError("meme_id 必须采用 meme-0000 格式")
    return {"reply": reply, "meme_id": meme_id}


def build_result(request_id: str, replydict: Mapping[str, Any]) -> dict[str, str]:
    """Keep result.text for old clients and expose two fields to new clients."""
    request_id = validate_request_id(request_id)
    result = validate_reply(replydict)
    return {
        "type": "result",
        "id": request_id,
        "text": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        **result,
        "reply_text": result["reply"],
    }


def decode_result(message: Mapping[str, Any]) -> dict[str, str]:
    """Read legacy JSON-in-text or direct fields; reject conflicting copies."""
    if not isinstance(message, Mapping) or message.get("type") != "result":
        raise ValidationError("需要 result 消息")
    validate_request_id(message.get("id"))
    decoded = None
    if "text" in message:
        if not isinstance(message["text"], str):
            raise ValidationError("result.text 必须为 JSON 字符串")
        try:
            decoded = validate_reply(json.loads(message["text"]))
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValidationError("result.text 中的回复格式无效") from exc
    if "reply" in message or "meme_id" in message:
        direct = validate_reply(message)
        if decoded is not None and decoded != direct:
            raise ValidationError("result.text 与直接回复字段不一致")
        decoded = direct
    if decoded is None:
        raise ValidationError("result 消息缺少 reply 和 meme_id")
    if "reply_text" in message and validate_text(message["reply_text"]) != decoded["reply"]:
        raise ValidationError("result.reply_text 与 reply 字段不一致")
    return decoded
