"""Read complete dialogue lines without taking ownership of a running server."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time
from typing import Callable

from communication.common import FRAME_LIMIT
from .dialogue_log import DEFAULT_LOG_PATH, is_dialogue_line


HISTORY_BYTES = 4 * 1024 * 1024


def resolve_log_source(lock_path: Path, journal_path: Path = DEFAULT_LOG_PATH) -> tuple[Path, int]:
    """Prefer the runtime journal, or an older server's regular stdout file.

    Never read a terminal or pipe: doing so could consume input or steal another
    process's output. A PID-only background launcher remains attachable without
    restarting the loaded model.
    """
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].isdecimal() or int(lines[0]) < 1:
        raise RuntimeError("未找到有效的模型服务进程，无法显示实时对话")
    pid = int(lines[0])
    if len(lines) > 1:
        try:
            metadata = json.loads(lines[1])
        except (ValueError, TypeError) as exc:
            raise RuntimeError("模型服务的日志记录无效，请检查原启动终端") from exc
        if not isinstance(metadata, dict) or metadata.get("pid") != pid:
            raise RuntimeError("模型服务的日志记录与当前进程不一致")
        if "dialogue_log" in metadata:
            value = metadata["dialogue_log"]
            if not isinstance(value, str) or Path(value).resolve() != journal_path.resolve():
                raise RuntimeError("模型服务的对话日志路径与当前应用不一致")
            return journal_path, pid
    stdout = Path(f"/proc/{pid}/fd/1")
    try:
        regular = stat.S_ISREG(stdout.stat().st_mode)
    except OSError:
        regular = False
    if regular:
        return stdout, pid
    raise RuntimeError("已有服务的输出仍在原终端；在原终端查看，或重新启动一次模型服务后使用本命令")


def _decode_dialogue(data: bytes) -> str | None:
    try:
        line = data.decode("utf-8")
    except UnicodeError:
        return None
    return line if is_dialogue_line(line) else None


def _history(stream, tail: int) -> tuple[list[str], bytes]:
    end = stream.seek(0, os.SEEK_END)
    start = max(0, end - HISTORY_BYTES)
    stream.seek(start)
    data = stream.read(end - start)
    if start:
        boundary = data.find(b"\n")
        data = data[boundary + 1:] if boundary >= 0 else b""
    parts = data.split(b"\n")
    pending = parts.pop()
    lines = [line for part in parts if (line := _decode_dialogue(part)) is not None]
    return (lines[-tail:] if tail else []), pending


def follow_file(path: Path, running: Callable[[], bool], *, tail: int = 20,
                emit: Callable[[str], None] | None = None, poll_interval: float = .2,
                sleep: Callable[[float], None] = time.sleep) -> None:
    """Print recent and appended whole lines until the owning server exits.

    Partial UTF-8/JSON records wait for their newline. Named journals can be
    rotated or truncated; oversized or unrelated lines never reach the console.
    Ctrl+C propagates to the launcher, which only closes this viewer.
    """
    if isinstance(tail, bool) or not isinstance(tail, int) or tail < 0:
        raise ValueError("tail must be a nonnegative integer")
    emit = emit or (lambda line: print(line, flush=True))
    path = Path(path)
    stream = None
    pending = b""
    oversized = False
    first = True
    try:
        while True:
            if stream is None:
                try:
                    stream = path.open("rb")
                except FileNotFoundError:
                    if not running():
                        return
                    sleep(poll_interval)
                    continue
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise RuntimeError("对话日志不是普通文件，已停止查看")
                if first:
                    history, pending = _history(stream, tail)
                    for line in history:
                        emit(line)
                    first = False
                else:
                    pending = b""
                oversized = len(pending) > FRAME_LIMIT
                if oversized:
                    pending = b""
            data = stream.read(65536)
            if data:
                parts = data.split(b"\n")
                for index, part in enumerate(parts):
                    if not oversized:
                        pending += part
                        if len(pending) > FRAME_LIMIT:
                            pending = b""
                            oversized = True
                    if index < len(parts) - 1:
                        if not oversized and (line := _decode_dialogue(pending)) is not None:
                            emit(line)
                        pending = b""
                        oversized = False
                continue
            try:
                current = path.stat()
            except FileNotFoundError:
                current = None
            opened = os.fstat(stream.fileno())
            if current is not None and (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                stream.close()
                stream = None
                continue
            if current is not None and current.st_size < stream.tell():
                stream.seek(0)
                pending = b""
                oversized = False
                continue
            if not running():
                return
            sleep(poll_interval)
    finally:
        if stream is not None:
            stream.close()
