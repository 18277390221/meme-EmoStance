#!/usr/bin/env python3
"""Raspberry Pi endpoint with full-duplex text and automatic reconnect."""

import argparse
import asyncio
import signal
import uuid

from common import (
    AUTH_TIMEOUT, FRAME_LIMIT, Console, Peer, ProtocolError,
    load_token, log, port_number, read_message, token_argument,
)

# Shared defaults; the server address is supplied explicitly at startup.
DEFAULT_PORT = 9000
DEFAULT_NAME = "raspberrypi"


class AuthenticationError(Exception):
    pass


class ChatClient:
    def __init__(self, host, port, name, token):
        self.host, self.port, self.name, self.token = host, port, name, token
        self.peer = None
        self.pending = set()
        self.stop = asyncio.Event()
        self.exit_code = 0

    async def receive(self, reader):
        while True:
            message = await read_message(reader)
            kind = message["type"]
            if kind == "result":
                self.pending.discard(message.get("id"))
                log(f"[服务器返回 / {message.get('id')}] {message.get('text', '')}")
            elif kind == "push":
                log(f"[服务器主动发送] {message.get('text', '')}")
            elif kind == "error":
                self.pending.discard(message.get("id"))
                log(f"[服务端错误] {message.get('message', '未知错误')}")
            elif kind != "pong":
                raise ProtocolError(f"服务端消息类型未知：{kind}")

    async def heartbeat(self, peer):
        while True:
            await asyncio.sleep(15)
            await peer.send({"type": "ping"})

    async def connection_loop(self):
        backoff = 1
        while not self.stop.is_set():
            peer = None
            tasks = []
            authenticated = False
            try:
                log(f"[正在连接] {self.host}:{self.port}")
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port, limit=FRAME_LIMIT), 10
                )
                peer = Peer(writer)
                await peer.send({"type": "hello", "name": self.name, "token": self.token})
                reply = await read_message(reader, AUTH_TIMEOUT)
                if reply["type"] == "error" and reply.get("code") == "auth_failed":
                    raise AuthenticationError("通信口令不一致，请安全复制服务器的 .env 到树莓派")
                if reply["type"] != "hello_ok":
                    raise ProtocolError(f"握手失败：{reply.get('message', reply['type'])}")
                self.peer = peer
                authenticated = True
                log("[已连接服务器] 输入文字回车发送；/quit 退出")
                tasks = [asyncio.create_task(self.receive(reader)), asyncio.create_task(self.heartbeat(peer))]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            except AuthenticationError as exc:
                log(f"[认证失败] {exc}")
                self.exit_code = 1
                self.stop.set()
                return
            except (OSError, EOFError, ProtocolError, asyncio.TimeoutError) as exc:
                log(f"[连接中断] {type(exc).__name__}: {exc}")
            finally:
                self.peer = None
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if peer:
                    await peer.close()
                if self.pending:
                    log(
                        f"[结果未确认] {len(self.pending)} 条请求的返回结果未知；"
                        "不会自动重发，避免重复处理。需要时请手动重新发送。"
                    )
                    self.pending.clear()
            if not self.stop.is_set():
                backoff = 1 if authenticated else backoff
                log(f"[重连] {backoff} 秒后重试")
                try:
                    await asyncio.wait_for(self.stop.wait(), backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 15)

    async def console(self):
        console = Console()
        while True:
            line = await console.readline()
            if line is None:
                # Receive-only/background use still needs the network connection.
                return
            if line == "/quit":
                self.stop.set()
                return
            if line == "/help":
                log("输入文字发送到服务器；/quit 退出；断线期间的输入不会排队")
                continue
            if not line:
                continue
            peer = self.peer
            if not peer:
                log("[未发送] 尚未连接服务器，请等待连接成功后再输入")
                continue
            if len(self.pending) >= 8:
                log("[未发送] 已有 8 条请求等待结果，请等待后再发送")
                continue
            request_id = uuid.uuid4().hex
            self.pending.add(request_id)
            try:
                await peer.send({"type": "request", "id": request_id, "text": line})
                log(f"[已发送 / {request_id}] {line}")
            except ProtocolError as exc:
                self.pending.discard(request_id)
                log(f"[未发送] {exc}")
            except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
                # A failed drain can follow a partial write. Delivery is unknown.
                log(f"[发送中断] {type(exc).__name__}；本条是否送达未知")
                peer.writer.close()

    async def run(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        connection = asyncio.create_task(self.connection_loop())
        console = asyncio.create_task(self.console())
        try:
            await self.stop.wait()
        finally:
            connection.cancel()
            console.cancel()
            await asyncio.gather(connection, console, return_exceptions=True)
            log("客户端已停止")
        return self.exit_code


def main():
    parser = argparse.ArgumentParser(
        description="树莓派双向文字客户端（通过 ZeroTier 使用）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", required=True, help="服务器的 ZeroTier IP 或其他可达地址")
    parser.add_argument("--port", type=port_number, default=DEFAULT_PORT, help="服务端口")
    parser.add_argument("--name", default=DEFAULT_NAME, help="设备名称")
    token_argument(parser)
    args = parser.parse_args()
    try:
        token = load_token(args.token_file)
        code = asyncio.run(ChatClient(args.host, args.port, args.name, token).run())
    except (ValueError, OSError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        return
    raise SystemExit(code)


if __name__ == "__main__":
    main()
