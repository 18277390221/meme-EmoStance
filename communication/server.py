#!/usr/bin/env python3
"""TCP endpoint: authenticate Raspberry Pi peers and exchange UTF-8 text."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hmac
import signal
import uuid

from common import (
    AUTH_TIMEOUT, FRAME_LIMIT, Console, Peer, ProtocolError,
    encode_message, load_token, log, port_number, read_message, token_argument,
)


def acknowledge(text, client_id):
    """Return a transport acknowledgement without invoking business logic."""
    return f"服务器已收到：{text}"


class ChatServer:
    def __init__(self, token, on_text=None, name="server"):
        self.token = token
        # Optional application callback is supplied by the caller, outside
        # this communication module. The standalone CLI only acknowledges.
        self.on_text = on_text if on_text is not None else acknowledge
        self.name = name
        self.clients = {}
        self.connections = set()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="communication")
        self.stop = asyncio.Event()

    async def process_requests(self, peer, name, queue):
        while True:
            message = await queue.get()
            request_id = message["id"]
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.on_text, message["text"], name
                )
                if not isinstance(result, str):
                    raise TypeError("通信回调必须返回字符串")
                response = {"type": "result", "id": request_id, "text": result}
                encode_message(response)  # Validate the reply before entering network error handling.
            except Exception as exc:
                log(f"[处理失败 {name} / {request_id}] {type(exc).__name__}: {exc}")
                response = {
                    "type": "error", "id": request_id,
                    "code": "handler_error", "message": "处理失败，请查看服务器日志",
                }
            try:
                await peer.send(response)
                if response["type"] == "result":
                    log(f"[已返回 {name} / {request_id}] {result!r}")
            except (OSError, ConnectionError, asyncio.TimeoutError):
                peer.writer.close()
                return

    async def connected(self, reader, writer):
        task = asyncio.current_task()
        self.connections.add(task)
        peer = Peer(writer)
        key = uuid.uuid4().hex
        name = "未认证客户端"
        worker = None
        try:
            hello = await read_message(reader, AUTH_TIMEOUT)
            supplied = hello.get("token")
            if (
                hello["type"] != "hello" or not isinstance(supplied, str)
                or not hmac.compare_digest(supplied.encode("utf-8"), self.token.encode("utf-8"))
            ):
                await peer.send({"type": "error", "code": "auth_failed", "message": "通信口令错误"})
                return
            name = hello.get("name", "raspberrypi")
            if not isinstance(name, str) or not 1 <= len(name) <= 64 or not name.isprintable():
                raise ProtocolError("客户端名称须为 1 到 64 个可打印字符")
            if len(self.clients) >= 16:
                await peer.send({"type": "error", "code": "busy", "message": "连接数已达上限"})
                return
            # Do not expose this peer to broadcasts before its handshake response.
            await peer.send({"type": "hello_ok", "name": self.name, "client_id": key})
            self.clients[key] = (name, peer)
            log(f"[已连接] {name}，地址 {writer.get_extra_info('peername')}")
            queue = asyncio.Queue(maxsize=8)
            worker = asyncio.create_task(self.process_requests(peer, name, queue))
            while True:
                message = await read_message(reader)
                if message["type"] == "ping":
                    await peer.send({"type": "pong"})
                elif message["type"] == "request":
                    request_id, text = message.get("id"), message.get("text")
                    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128 or not isinstance(text, str):
                        await peer.send({"type": "error", "code": "bad_request", "message": "request 需要字符串 id 和 text"})
                        continue
                    if queue.full():
                        await peer.send({"type": "error", "id": request_id, "code": "busy", "message": "请求队列已满，请稍后重新发送"})
                        continue
                    log(f"[收到 {name} / {request_id}] {text!r}")
                    queue.put_nowait(message)
                else:
                    await peer.send({"type": "error", "code": "bad_type", "message": "仅支持 request 或 ping"})
        except EOFError:
            pass
        except ProtocolError as exc:
            log(f"[协议错误] {name}: {exc}")
            try:
                await peer.send({"type": "error", "code": "protocol_error", "message": str(exc)})
            except (OSError, ConnectionError, asyncio.TimeoutError):
                pass
        except (OSError, ConnectionError, asyncio.TimeoutError) as exc:
            log(f"[连接中断] {name}: {type(exc).__name__}")
        finally:
            existed = self.clients.pop(key, None)
            if worker:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
            await peer.close()
            self.connections.discard(task)
            if existed:
                log(f"[已断开] {name}")

    async def broadcast(self, text):
        clients = list(self.clients.values())
        if not clients:
            log("[未发送] 尚无树莓派连接；请先启动树莓派 client.py")
            return
        outcomes = await asyncio.gather(
            *(peer.send({"type": "push", "text": text, "name": self.name}) for _, peer in clients),
            return_exceptions=True,
        )
        sent = 0
        for (name, peer), outcome in zip(clients, outcomes):
            if isinstance(outcome, BaseException):
                log(f"[发送失败] {name}: {type(outcome).__name__}")
                peer.writer.close()
            else:
                sent += 1
        log(f"[已发送到 {sent} 个客户端] {text!r}")

    async def console(self):
        console = Console()
        while True:
            line = await console.readline()
            if line is None:
                # stdin closed: keep the network server available for background use.
                return
            if line == "/quit":
                self.stop.set()
                return
            if line == "/clients":
                log(f"[在线 {len(self.clients)}] " + ", ".join(name for name, _ in self.clients.values()))
            elif line == "/help":
                log("输入文字发送给所有在线树莓派；/clients 查看连接；/quit 退出")
            elif line:
                try:
                    await self.broadcast(line)
                except ProtocolError as exc:
                    log(f"[未发送] {exc}")

    async def run(self, host, port):
        server = await asyncio.start_server(self.connected, host, port, limit=FRAME_LIMIT)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_running_loop().add_signal_handler(sig, self.stop.set)
            except (NotImplementedError, RuntimeError):
                pass
        addresses = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        log(f"服务已启动：{addresses}")
        log("树莓派连接后，输入文字回车即可发送；/clients 查看连接；/quit 退出")
        console = asyncio.create_task(self.console())
        try:
            async with server:
                await self.stop.wait()
        finally:
            console.cancel()
            await asyncio.gather(console, return_exceptions=True)
            connections = list(self.connections)
            for task in connections:
                task.cancel()
            await asyncio.gather(*connections, return_exceptions=True)
            self.executor.shutdown(wait=False, cancel_futures=True)
            log("服务已停止")


def main():
    parser = argparse.ArgumentParser(description="服务器与树莓派双向文字通信（可通过 ZeroTier 使用）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；跨设备使用时填服务器的 ZeroTier IP")
    parser.add_argument("--port", type=port_number, default=9000)
    parser.add_argument("--name", default="server")
    token_argument(parser)
    args = parser.parse_args()
    try:
        token = load_token(args.token_file)
        asyncio.run(ChatServer(token, name=args.name).run(args.host, args.port))
    except (ValueError, OSError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
