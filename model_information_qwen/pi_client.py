"""Raspberry Pi business client for reply + meme_id over communication TCP.

Usage::

    client = InferenceClient(host, token=shared_token)
    await client.start()                 # connect() is an alias
    result = await client.request("你好")
    await client.close()

One instance keeps its randomly generated session_id across reconnects.
Requests whose delivery or reply is uncertain always fail; they are never
resent. Subsequent requests can wait for the next authenticated connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any
import uuid

from communication.common import (
    AUTH_TIMEOUT,
    FRAME_LIMIT,
    Console,
    Peer,
    ProtocolError,
    load_token,
    port_number,
    read_message,
)

from .protocol import decode_result, validate_session_id, validate_text


HEARTBEAT_INTERVAL = 15.0
MAX_PENDING = 8
DEFAULT_TOKEN_FILE = Path(__file__).resolve().parents[1] / "communication" / ".env"


class AuthenticationError(ConnectionError):
    pass


class RemoteError(RuntimeError):
    def __init__(self, message: str, *, code: str = "remote_error") -> None:
        super().__init__(message)
        self.code = code


class InferenceClient:
    def __init__(
        self,
        host: str,
        port: int = 9000,
        name: str = "raspberrypi",
        *,
        token: str,
        session_id: str | None = None,
    ) -> None:
        if not isinstance(host, str) or not host.strip():
            raise ValueError("服务器地址不能为空")
        self.host = host
        self.port = port_number(port)
        if not isinstance(name, str) or not 1 <= len(name) <= 64 or not name.isprintable():
            raise ValueError("设备名称须为 1–64 个可打印字符")
        if not isinstance(token, str) or len(token.strip()) < 16:
            raise ValueError("通信口令至少需要 16 个字符")
        self.name = name
        self._token = token.strip()
        self.session_id = validate_session_id(session_id if session_id is not None else uuid.uuid4().hex)
        self.client_id: str | None = None
        self.pushes: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._peer: Peer | None = None
        self._pending: dict[str, asyncio.Future[dict[str, str]]] = {}
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._fatal_error: Exception | None = None
        self._closed = False

    @property
    def connected(self) -> bool:
        return self._ready.is_set() and self._peer is not None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def start(self, timeout: float | None = 10.0) -> InferenceClient:
        """Start reconnect maintenance and wait for an authenticated link."""
        if self._closed:
            raise ConnectionError("客户端已关闭")
        if self._fatal_error is not None:
            raise self._fatal_error
        if self._runner is None:
            self._runner = asyncio.create_task(self._connection_loop())
        if self.connected:
            return self
        ready = asyncio.create_task(self._ready.wait())
        stopped = asyncio.create_task(self._stop.wait())
        try:
            done, _ = await asyncio.wait(
                (ready, stopped), timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if not done:
                raise TimeoutError("等待服务器连接超时；后台仍会继续重连")
            if self._fatal_error is not None:
                raise self._fatal_error
            if not self.connected:
                raise ConnectionError("客户端连接已停止")
            return self
        finally:
            ready.cancel()
            stopped.cancel()
            await asyncio.gather(ready, stopped, return_exceptions=True)

    async def connect(self, timeout: float | None = 10.0) -> InferenceClient:
        return await self.start(timeout=timeout)

    async def request(self, text: str, *, timeout: float | None = None) -> dict[str, str]:
        """Send once, then await its reply. A timeout never triggers a resend."""
        validate_text(text)
        await self.start()
        peer = self._peer
        if peer is None:
            raise ConnectionError("连接已中断，请等待重连后再发起新请求")
        if len(self._pending) >= MAX_PENDING:
            raise RuntimeError("已有 8 条请求等待结果，请等待后再发送")
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            try:
                await peer.send({"type": "request", "id": request_id, "text": text})
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                peer.writer.close()
                raise ConnectionError("发送中断，本轮送达状态未知，未自动重发") from exc
            if timeout is None:
                return await future
            return await asyncio.wait_for(future, timeout)
        finally:
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                # A disconnect can race a failed drain before this coroutine
                # awaits the future; consume that exception in either case.
                future.exception()

    def _fail_pending(self, reason: str) -> None:
        pending, self._pending = self._pending, {}
        for future in pending.values():
            if not future.done():
                future.set_exception(ConnectionError(reason))

    async def _receive(self, reader: asyncio.StreamReader, peer: Peer) -> None:
        while True:
            message = await read_message(reader)
            kind = message["type"]
            if kind == "result":
                future = self._pending.get(message.get("id")) if isinstance(message.get("id"), str) else None
                if future is None or future.done():
                    continue
                try:
                    result = decode_result(message)
                except ValueError as exc:
                    future.set_exception(exc)
                else:
                    future.set_result(result)
            elif kind == "error":
                future = self._pending.get(message.get("id")) if isinstance(message.get("id"), str) else None
                error = RemoteError(
                    str(message.get("message", "服务器处理失败")),
                    code=str(message.get("code", "remote_error")),
                )
                if future is not None and not future.done():
                    future.set_exception(error)
                elif "id" not in message:
                    raise error
            elif kind == "push":
                self.pushes.put_nowait(message)
            elif kind == "ping":
                await peer.send({"type": "pong"})
            elif kind != "pong":
                raise ProtocolError("服务器返回未知消息类型")

    async def _heartbeat(self, peer: Peer) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await peer.send({"type": "ping"})

    async def _connection_loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            peer = None
            tasks: list[asyncio.Task[None]] = []
            authenticated = False
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port, limit=FRAME_LIMIT), 10.0
                )
                peer = Peer(writer)
                await peer.send({
                    "type": "hello", "name": self.name,
                    "token": self._token, "session_id": self.session_id,
                })
                hello = await read_message(reader, AUTH_TIMEOUT)
                if hello["type"] == "error" and hello.get("code") == "auth_failed":
                    raise AuthenticationError("通信认证失败，请检查两端共享口令")
                if hello["type"] != "hello_ok":
                    raise ProtocolError("服务器未确认连接")
                if "session_id" in hello and hello["session_id"] != self.session_id:
                    raise ProtocolError("服务器返回的会话标识与请求不一致")
                client_id = hello.get("client_id")
                if not isinstance(client_id, str) or not client_id:
                    raise ProtocolError("服务器未返回连接标识")
                self.client_id = client_id
                self._peer = peer
                authenticated = True
                self._ready.set()
                tasks = [
                    asyncio.create_task(self._receive(reader, peer)),
                    asyncio.create_task(self._heartbeat(peer)),
                ]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except AuthenticationError as exc:
                self._fatal_error = exc
                self._stop.set()
            except (OSError, EOFError, ProtocolError, RemoteError, asyncio.TimeoutError):
                pass
            finally:
                self._ready.clear()
                self._peer = None
                self.client_id = None
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if peer is not None:
                    await peer.close()
                self._fail_pending("连接中断，未确认的请求不会自动重发")
            if not self._stop.is_set():
                backoff = 1.0 if authenticated else backoff
                try:
                    await asyncio.wait_for(self._stop.wait(), backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 15.0)

    async def close(self) -> None:
        self._closed = True
        self._stop.set()
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
            self._runner = None
        self._ready.clear()
        self._fail_pending("客户端已关闭，未确认的请求不会自动重发")

    async def __aenter__(self) -> InferenceClient:
        return await self.start()

    async def __aexit__(self, *_: object) -> None:
        await self.close()


async def _run_cli(args: argparse.Namespace) -> None:
    client = InferenceClient(
        args.host, args.port, args.name,
        token=load_token(args.token_file), session_id=args.session_id,
    )
    tasks: set[asyncio.Task[None]] = set()

    async def send(text: str) -> None:
        try:
            result = await client.request(text)
            print(json.dumps(result, ensure_ascii=False), flush=True)
        except (ValueError, RuntimeError, ConnectionError, asyncio.TimeoutError) as exc:
            print(f"请求失败：{exc}", file=sys.stderr, flush=True)

    try:
        await client.start(timeout=None)
        console = Console()
        while True:
            text = await console.readline()
            if text is None:
                await asyncio.gather(*tasks)
                break
            if text == "/quit":
                break
            if text == "/help":
                print("每行发送一次用户输入；/quit 退出。输出包含 reply 和 meme_id。", file=sys.stderr)
                continue
            if text.strip():
                task = asyncio.create_task(send(text))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="树莓派模型通信客户端：返回回复与 meme_id")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=port_number, default=9000)
    parser.add_argument("--name", default="raspberrypi")
    parser.add_argument("--session-id")
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run_cli(args))
    except (ValueError, ConnectionError, OSError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
