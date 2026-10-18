"""Private, line-oriented dialogue journal shared by server and terminal viewer.

Only the two application directions are accepted. Authentication, heartbeats,
and diagnostic lines never belong in this file.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import stat
import threading


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "runs" / "conversation.log"
_APPEND_LOCK = threading.Lock()
_LINE = re.compile(
    r"^\[(?P<timestamp>[^\]\r\n]+)\] "
    r"\[(?P<direction>树莓派 → 服务器|服务器 → 树莓派)\] "
    r"(?P<payload>\{[^\r\n]*\})$"
)
_COMMON_KEYS = {"device", "request_id"}
_INPUT_KEYS = _COMMON_KEYS | {"text", "end", "status"}
_OUTPUT_KEYS = _COMMON_KEYS | {"reply", "meme_id", "code", "message"}


def is_dialogue_line(line: str) -> bool:
    """Recognize one existing server dialogue line without exposing raw frames."""
    if not isinstance(line, str) or len(line) > 1024 * 1024:
        return False
    # The viewer reads newline-terminated lines; embedded physical line breaks
    # must not be able to introduce a second terminal/log entry.
    if line.endswith("\n"):
        line = line[:-1]
        if line.endswith("\r"):
            line = line[:-1]
    match = _LINE.fullmatch(line)
    if match is None:
        return False
    try:
        datetime.fromisoformat(match["timestamp"])
        payload = json.loads(match["payload"])
    except (ValueError, TypeError, RecursionError):
        return False
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("device"), str):
        return False
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        return False
    if match["direction"] == "树莓派 → 服务器":
        return payload.keys() <= _INPUT_KEYS and isinstance(payload.get("text"), str)
    if not payload.keys() <= _OUTPUT_KEYS:
        return False
    return (
        isinstance(payload.get("reply"), str)
        and isinstance(payload.get("meme_id"), str)
    ) or (
        isinstance(payload.get("code"), str)
        and isinstance(payload.get("message"), str)
    )


def append_dialogue(line: str, path: str | Path | None = None) -> bool:
    """Append one UTF-8 business line; journal failures never abort inference.

    The server has one writer process. A small lock prevents its worker threads
    from mixing records, and O_APPEND keeps writes at the end of the file.
    """
    if not is_dialogue_line(line):
        return False
    try:
        data = (line.rstrip("\r\n") + "\n").encode("utf-8")
        target = Path(path) if path is not None else DEFAULT_LOG_PATH
        with _APPEND_LOCK:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(target, flags, 0o600)
            try:
                metadata = os.fstat(fd)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    return False
                if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
                    return False
                os.fchmod(fd, 0o600)
                remaining = memoryview(data)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        return False
                    remaining = remaining[written:]
            finally:
                os.close(fd)
        return True
    except (OSError, UnicodeError, ValueError):
        return False
