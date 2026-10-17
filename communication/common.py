"""Shared newline-delimited JSON protocol; Python 3.9+ and no dependencies."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import threading


FRAME_LIMIT = 1024 * 1024
READ_TIMEOUT = 60
AUTH_TIMEOUT = 10


class ProtocolError(Exception):
    pass


def log(message):
    print(message, flush=True)


def load_token(token_file):
    """Read a shared token without executing a shell configuration file."""
    token = os.environ.get("ZT_CHAT_TOKEN", "").strip()
    if not token:
        try:
            for line in Path(token_file).read_text(encoding="utf-8").splitlines():
                if line.startswith("ZT_CHAT_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
        except FileNotFoundError:
            pass
    if len(token) < 16:
        raise ValueError(
            "缺少通信口令：先运行 python3 init_config.py，"
            "并将生成的 .env 安全复制到树莓派；两端需使用相同口令。"
        )
    return token


def port_number(value):
    number = int(value)
    if not 1 <= number <= 65535:
        raise argparse.ArgumentTypeError("端口必须在 1 到 65535 之间")
    return number


def token_argument(parser):
    parser.add_argument(
        "--token-file",
        default=str(Path(__file__).with_name(".env")),
        help="含 ZT_CHAT_TOKEN=... 的文件；环境变量优先（默认脚本旁的 .env）",
    )


def encode_message(message):
    data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(data) > FRAME_LIMIT:
        raise ProtocolError("消息超过 1 MiB 限制")
    return data


async def read_message(reader, timeout=READ_TIMEOUT):
    try:
        line = await asyncio.wait_for(reader.readline(), timeout)
    except ValueError as exc:
        raise ProtocolError("消息超过 1 MiB 限制") from exc
    if not line:
        raise EOFError("对端已关闭连接")
    if len(line) > FRAME_LIMIT or not line.endswith(b"\n"):
        raise ProtocolError("消息过长或缺少换行分隔符")
    try:
        message = json.loads(line.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError("消息必须是 UTF-8 JSON") from exc
    if not isinstance(message, dict) or not isinstance(message.get("type"), str):
        raise ProtocolError("消息必须含有字符串 type 字段")
    return message


class Peer:
    def __init__(self, writer):
        self.writer = writer
        self.lock = asyncio.Lock()

    async def send(self, message):
        data = encode_message(message)
        async with self.lock:
            self.writer.write(data)
            await asyncio.wait_for(self.writer.drain(), 10)

    async def close(self):
        self.writer.close()
        try:
            await asyncio.wait_for(self.writer.wait_closed(), 2)
        except (ConnectionError, OSError, asyncio.TimeoutError):
            pass


class Console:
    """A daemon reader allows receiving messages while waiting for keyboard input."""

    def __init__(self):
        self.queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def read_stdin():
            try:
                for line in sys.stdin:
                    loop.call_soon_threadsafe(self.queue.put_nowait, line.rstrip("\r\n"))
                loop.call_soon_threadsafe(self.queue.put_nowait, None)
            except (RuntimeError, OSError, UnicodeError):
                pass

        threading.Thread(target=read_stdin, daemon=True, name="console-reader").start()

    async def readline(self):
        return await self.queue.get()
