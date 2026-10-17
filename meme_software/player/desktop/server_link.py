"""Application-side adapter for communication/ JSONL TCP protocol; no audio upload."""
from collections import deque
import json
import os
from pathlib import Path
import queue
import re
import socket
import threading
import time
import uuid

FRAME_LIMIT = 1024 * 1024


def load_token(path):
    token = os.environ.get('ZT_CHAT_TOKEN', '').strip()
    if not token:
        try:
            with Path(path).expanduser().open(encoding='utf-8') as stream:
                for line in stream:
                    if line.startswith('ZT_CHAT_TOKEN='):
                        token = line.split('=', 1)[1].strip()
                        break
        except OSError:
            raise ValueError('通信口令文件不可读，请在语音设置中选择已有.env文件。') from None
    if len(token) < 16:
        raise ValueError('缺少有效的通信口令，请安全使用服务器现有.env，不要发送口令到聊天。')
    return token


def encode(message):
    data = (json.dumps(message, ensure_ascii=False, separators=(',', ':')) + '\n').encode('utf-8')
    if len(data) > FRAME_LIMIT:
        raise ValueError('文字消息超过协议限制')
    return data


def parse_reply(message):
    """Return a structured cat reply; ordinary legacy acknowledgements return None."""
    if 'meme_id' in message or 'reply_text' in message:
        payload = message
    else:
        text = message.get('text')
        if not isinstance(text, str):
            raise ValueError('服务器result需要text字符串或meme_id/reply_text字段。')
        # The team's callback returns str. JSON inside result.text requires no
        # changes to their TCP module. Plain echoes must never become replies.
        if not text.lstrip().startswith(('{', '[')):
            return None
        try:
            payload = json.loads(text)
        except (ValueError, RecursionError):
            raise ValueError('服务器回复不是有效的JSON对象。') from None
    if not isinstance(payload, dict):
        raise ValueError('服务器回复需要包含meme_id和reply_text的JSON对象。')
    key, text = payload.get('meme_id'), payload.get('reply_text')
    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', key):
        raise ValueError('服务器meme_id需要是本地素材的完整编号，例如meme-0157。')
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError('服务器reply_text需要是1到2000字的回复文字。')
    return {'meme_id': key, 'reply_text': text.strip()}


class ServerLink:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.socket = None
        self.connected = False
        self.waiting = {}
        self.outgoing = queue.Queue(maxsize=8)
        self.events = deque(maxlen=256)
        self.last_reply_at = 0.0

    def event(self, kind, **values):
        with self.lock:
            self.events.append({'type': kind, **values})

    def drain(self):
        with self.lock:
            result = list(self.events)
            self.events.clear()
        return result

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            self.event('unsent', message='识别文字为空或过长，未发送。')
            return None
        with self.lock:
            if not self.connected:
                self.events.append({'type': 'unsent', 'message': '服务器未连接，本句话未发送；连接后请重说。'})
                return None
            if len(self.waiting) >= 8:
                self.events.append({'type': 'unsent', 'message': '有8句话等待服务器确认，请稍后重说。'})
                return None
            key = uuid.uuid4().hex
            self.waiting[key] = time.monotonic()
            self.outgoing.put_nowait({'type': 'request', 'id': key, 'text': text})
        return key

    def start(self, host, port, token_file):
        if self.thread and self.thread.is_alive():
            raise ValueError('通信仍在退出，请稍后重试。')
        if not host or len(host) > 253 or any(x.isspace() for x in host):
            raise ValueError('请填写有效的服务器地址')
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError('服务器端口必须为1到65535')
        token = load_token(token_file)
        self.stop.clear()
        self.thread = threading.Thread(target=self._run, args=(host, port, token), daemon=True)
        self.thread.start()

    def _run(self, host, port, token):
        backoff = 1
        while not self.stop.is_set():
            sock = None
            authenticated = False
            try:
                self.event('connecting')
                sock = socket.create_connection((host, port), timeout=3)
                self.socket = sock
                sock.settimeout(.1)
                buffer = bytearray()

                def read_frame(deadline):
                    while not self.stop.is_set():
                        if b'\n' in buffer:
                            end = buffer.index(b'\n') + 1
                            if end > FRAME_LIMIT:
                                raise ValueError('frame limit')
                            line = bytes(buffer[:end])
                            del buffer[:end]
                            message = json.loads(line.decode('utf-8'))
                            if not isinstance(message, dict) or not isinstance(message.get('type'), str):
                                raise ValueError('bad frame')
                            return message
                        if len(buffer) >= FRAME_LIMIT:
                            raise ValueError('frame limit')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('read timeout')
                        try:
                            data = sock.recv(65536)
                        except socket.timeout:
                            continue
                        if not data:
                            raise EOFError()
                        buffer.extend(data)
                    raise EOFError()

                def send(message):
                    # Only this thread writes socket frames; heartbeats cannot interleave.
                    sock.settimeout(3)
                    try:
                        sock.sendall(encode(message))
                    finally:
                        sock.settimeout(.1)

                send({'type': 'hello', 'name': 'meme-player', 'token': token})
                hello = read_frame(time.monotonic() + 10)
                if hello['type'] == 'error' and hello.get('code') == 'auth_failed':
                    self.event('auth_error', message='服务器认证失败，请核对两端是否使用同一份.env。')
                    return
                if hello['type'] != 'hello_ok':
                    raise ValueError('handshake')
                with self.lock:
                    self.connected = True
                authenticated = True
                backoff = 1
                self.event('connected')
                ping_at = time.monotonic() + 15
                last_read = time.monotonic()
                while not self.stop.is_set():
                    try:
                        request = self.outgoing.get_nowait()
                    except queue.Empty:
                        request = None
                    if request:
                        send(request)
                        self.event('sent', id=request['id'])
                    now = time.monotonic()
                    if now >= ping_at:
                        send({'type': 'ping'})
                        ping_at = now + 15
                    with self.lock:
                        expired = [key for key, when in self.waiting.items() if now - when >= 60]
                        for key in expired:
                            del self.waiting[key]
                    if expired:
                        self.event('unconfirmed', count=len(expired), message='服务器60秒未确认，未自动重发；可稍后重新说。')
                    if now - last_read >= 60:
                        raise TimeoutError('idle')
                    # Try one receive cycle; retain fragmented/multiple frames.
                    try:
                        message = read_frame(min(time.monotonic() + .12, last_read + 60))
                    except TimeoutError:
                        continue
                    last_read = time.monotonic()
                    kind, key = message['type'], message.get('id')
                    if kind in ('result', 'error'):
                        if not isinstance(key, str):
                            raise ValueError('bad result id')
                        with self.lock:
                            requested_at = self.waiting.pop(key, None)
                        if requested_at is not None:
                            if kind == 'result':
                                try:
                                    reply = parse_reply(message)
                                except ValueError as error:
                                    self.event('reply_error', message=str(error))
                                    continue
                                if reply is not None:
                                    if requested_at < self.last_reply_at:
                                        self.event('stale_reply', message='较早一句的回复晚于新回复到达，已忽略。')
                                        continue
                                    self.last_reply_at = requested_at
                                    self.event('reply', id=key, **reply)
                                    continue
                                self.event('ack', id=key)
                            else:
                                self.event('server_error', message='服务器未能接收/处理本句话，请稍后重说。')
                    elif kind == 'ping':
                        send({'type': 'pong'})
                    elif kind not in ('pong', 'push'):
                        raise ValueError('unexpected server frame')
            except (OSError, EOFError, ValueError, RecursionError):
                if not self.stop.is_set():
                    self.event('disconnected', message='服务器未连接，正在重连。请确认TCP 9000服务已启动。')
            finally:
                with self.lock:
                    self.connected = False
                    count = len(self.waiting)
                    self.waiting.clear()
                    while True:
                        try:
                            self.outgoing.get_nowait()
                        except queue.Empty:
                            break
                if count and not self.stop.is_set():
                    self.event('unconfirmed', count=count, message='连接中断，部分文字是否送达未知，不自动重发。')
                if sock:
                    sock.close()
                self.socket = None
            if self.stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15)
        self.event('closed')

    def close(self, grace=2):
        # ASR is stopped/drained first by the caller; allow its final sentence to be confirmed.
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            with self.lock:
                waiting = bool(self.waiting) and self.connected
            if not waiting:
                break
            time.sleep(.03)
        self.stop.set()
        sock = self.socket
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread:
            self.thread.join(4)
