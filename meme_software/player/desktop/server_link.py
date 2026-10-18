"""Version 2 JSONL client: readiness, one request, reconnect status and playback ACK."""
from collections import deque
import hashlib
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
    if 'meme_id' in message or 'reply_text' in message:
        payload = message
    else:
        text = message.get('text')
        if not isinstance(text, str):
            raise ValueError('服务器result需要text字符串或meme_id/reply_text字段。')
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
    conversation_id = message.get('conversation_id', payload.get('conversation_id'))
    has_conversation_id = 'conversation_id' in message or 'conversation_id' in payload
    if has_conversation_id and (not isinstance(conversation_id, str) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', conversation_id)):
        raise ValueError('服务器conversation_id格式不正确。')
    revisions = [source['background_revision'] for source in (message, payload)
                 if 'background_revision' in source]
    if any(type(value) is not int or value < 0 for value in revisions):
        raise ValueError('服务器background_revision需要是非负整数。')
    if len(set(revisions)) > 1:
        raise ValueError('服务器background_revision字段不一致。')
    background_revision = revisions[0] if revisions else 0
    reply = {'meme_id': key, 'reply_text': text.strip(), 'background_revision': background_revision}
    if conversation_id is not None:
        reply['conversation_id'] = conversation_id
    return reply


class ServerLink:
    def __init__(self, on_ready=None):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = self.socket = None
        self.connected = False
        self.model_ready = False
        self.on_ready = on_ready
        self.session_id = uuid.uuid4().hex
        self.server_instance_id = None
        self.model_version = None
        self.conversation_id = None
        self.background_revision = 0
        self._conversation_revisions = {}
        self.waiting = {}
        self.awaiting_playback = None
        self.playback_receipts = {}
        self.outgoing = queue.Queue(maxsize=32)
        self.events = deque(maxlen=256)
        self.completed = deque(maxlen=1024)
        self.completed_ids = set()

    @property
    def recovering(self):
        with self.lock:
            return bool(self.waiting)

    def _ready(self, value):
        with self.lock:
            self.connected = self.model_ready = bool(value)
        # Never call application callbacks with the network lock held.
        if self.on_ready:
            self.on_ready(bool(value))

    def event(self, kind, **values):
        with self.lock:
            self.events.append({'type': kind, **values})

    def drain(self):
        with self.lock:
            result = list(self.events)
            self.events.clear()
        return result

    def _queue_control(self, message):
        with self.lock:
            if not self.connected or not self.model_ready:
                return False
            try:
                self.outgoing.put_nowait(message)
            except queue.Full:
                self.events.append({'type': 'reply_error', 'message': '通信控制队列已满，正在重新连接。'})
                sock = self.socket
                if sock:
                    try:
                        sock.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                return False
            return True

    def speech_start(self):
        return self._queue_control({'type': 'speech_start'})

    def speech_cancel(self):
        return self._queue_control({'type': 'speech_cancel'})

    def playback_done(self, request_id, failed=False):
        if not isinstance(request_id, str) or not request_id:
            return False
        control = 'playback_failed' if failed else 'playback_done'
        with self.lock:
            if request_id not in self.completed_ids and request_id != self.awaiting_playback:
                return False  # An old boot's playback is no longer a server lease.
            self.playback_receipts[request_id] = control
            if self.awaiting_playback == request_id:
                self.awaiting_playback = None
        # Keep this receipt until control_ack, including across a dropped send.
        return self._queue_control({'type': control, 'id': request_id})

    def submit(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            self.event('unsent', message='识别文字为空或过长，未发送。')
            return None
        with self.lock:
            if not self.connected or not self.model_ready:
                self.events.append({'type': 'unsent', 'message': '服务器尚未就绪，本句话未发送；就绪后请重说。'})
                return None
            if self.waiting or self.awaiting_playback:
                self.events.append({'type': 'unsent', 'message': '上一轮还未完成，请稍候。'})
                return None
            key = uuid.uuid4().hex
            self.waiting[key] = {'at': time.monotonic(), 'query_at': time.monotonic(), 'accepted': False}
            try:
                self.outgoing.put_nowait({'type': 'request', 'id': key, 'text': text.strip(), 'end': True})
            except queue.Full:
                del self.waiting[key]
                self.events.append({'type': 'unsent', 'message': '通信繁忙，本句话未发送，请重说。'})
                return None
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

    def _handshake(self, hello, send):
        if (hello.get('type') != 'hello_ok' or hello.get('protocol_version') != 2 or
                hello.get('model_ready') is not True or hello.get('max_inflight') != 1 or
                not isinstance(hello.get('server_instance_id'), str) or not hello['server_instance_id'] or
                not isinstance(hello.get('model_version'), str) or
                hello.get('session_id') != self.session_id):
            raise ValueError('服务器尚未提供版本2模型就绪握手，请更新服务器后重试。')
        epoch = hello['server_instance_id']
        with self.lock:
            changed = self.server_instance_id is not None and self.server_instance_id != epoch
            old_ids = list(self.waiting)
            if changed:
                self.waiting.clear()
                self.playback_receipts.clear()
                self.awaiting_playback = None
                self.completed.clear()
                self.completed_ids.clear()
                self.conversation_id = None
                self.background_revision = 0
                self._conversation_revisions.clear()
            self.server_instance_id = epoch
            self.model_version = hello['model_version']
            pending = list(self.waiting)
            receipts = list(self.playback_receipts.items())
        if changed:
            self.event('server_restarted', ids=old_ids,
                       message='服务器已重新启动，旧请求不自动重发；请在收音恢复后重说。')
        for key, control in receipts:
            send({'type': control, 'id': key})
        for key in pending:
            send({'type': 'request_status', 'id': key})
        if not pending and self.awaiting_playback is None:
            send({'type': 'speech_cancel'})  # Release a pre-disconnect unfinished phrase.
        self._ready(True)
        self.event('connected', model_version=self.model_version, recovering=bool(pending))

    def _received(self, message, send):
        kind, key = message['type'], message.get('id')
        if kind == 'accepted':
            if not isinstance(key, str):
                raise ValueError('bad accepted id')
            with self.lock:
                item = self.waiting.get(key)
                if item:
                    item['accepted'] = True
                    item['query_at'] = time.monotonic()
            if item:
                self.event('accepted', id=key, status=message.get('status', 'processing'))
        elif kind in ('result', 'error'):
            if not isinstance(key, str):
                # Control failures have no request id; show them without dropping an inference.
                if kind == 'error':
                    self.event('control_error', message='服务器未能执行收音/播放状态更新。', code=message.get('code'))
                    return
                raise ValueError('bad result id')
            with self.lock:
                if kind == 'error' and key in self.playback_receipts and key not in self.waiting:
                    self.playback_receipts.pop(key, None)
                item = self.waiting.pop(key, None)
                duplicate = key in self.completed_ids
                if item is not None and not duplicate:
                    if len(self.completed) == self.completed.maxlen:
                        self.completed_ids.discard(self.completed[0])
                    self.completed.append(key)
                    self.completed_ids.add(key)
                    if kind == 'result':
                        self.awaiting_playback = key
            if item is None or duplicate:
                return
            if kind == 'error':
                unknown = message.get('code') == 'unknown_request'
                self.event('server_error', id=key, code=message.get('code'),
                           message=('服务器已无本句话的处理记录，未自动重发，请重说。' if unknown else
                                    '服务器未能处理本句话，请稍后重说。'))
                return
            try:
                reply = parse_reply(message)
                if reply is None:
                    raise ValueError('服务器只返回了普通确认，没有模型回复。')
                if 'conversation_id' not in reply:
                    # Older v2 servers cannot identify an idle reset. Keep one
                    # stable background for their whole boot, never guess by time.
                    identity = str(self.server_instance_id) + ':' + self.session_id
                    reply['conversation_id'] = 'legacy-' + hashlib.sha256(identity.encode()).hexdigest()
                with self.lock:
                    identity, revision = reply['conversation_id'], reply['background_revision']
                    previous = self._conversation_revisions.get(identity)
                    if previous is not None and (identity != self.conversation_id or revision < previous):
                        raise ValueError('服务器回复属于已过期的对话或背景版本，已忽略。')
                    self.conversation_id, self.background_revision = identity, revision
                    self._conversation_revisions[identity] = revision
                    if len(self._conversation_revisions) > 1024:
                        del self._conversation_revisions[next(iter(self._conversation_revisions))]
            except ValueError as error:
                self.playback_done(key, failed=True)
                self.event('reply_error', id=key, message=str(error))
                return
            self.event('reply', id=key, **reply)
        elif kind == 'control_ack':
            if message.get('control') in ('playback_done', 'playback_failed'):
                with self.lock:
                    self.playback_receipts.pop(key, None)
        elif kind == 'ping':
            send({'type': 'pong'})
        elif kind not in ('pong', 'push'):
            raise ValueError('unexpected server frame')

    def _run(self, host, port, token):
        backoff = 1
        while not self.stop.is_set():
            sock = None
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
                    # One socket owner serializes request/control/heartbeat frames.
                    sock.settimeout(3)
                    try:
                        sock.sendall(encode(message))
                    finally:
                        sock.settimeout(.1)

                send({'type': 'hello', 'name': 'meme-player', 'token': token,
                      'protocol_version': 2, 'session_id': self.session_id})
                hello = read_frame(time.monotonic() + 10)
                if hello['type'] == 'error' and hello.get('code') == 'auth_failed':
                    self.event('auth_error', message='服务器认证失败，请核对两端是否使用同一份.env。')
                    return
                self._handshake(hello, send)
                backoff = 1
                ping_at = time.monotonic() + 15
                last_read = time.monotonic()
                while not self.stop.is_set():
                    try:
                        request = self.outgoing.get_nowait()
                    except queue.Empty:
                        request = None
                    if request:
                        send(request)
                        if request['type'] == 'request':
                            self.event('sent', id=request['id'])
                    now = time.monotonic()
                    if now >= ping_at:
                        send({'type': 'ping'})
                        ping_at = now + 15
                    with self.lock:
                        queries = [key for key, item in self.waiting.items() if now - item['query_at'] >= 30]
                        for key in queries:
                            self.waiting[key]['query_at'] = now
                    for key in queries:
                        send({'type': 'request_status', 'id': key})
                    if now - last_read >= 60:
                        raise TimeoutError('idle')
                    try:
                        message = read_frame(min(time.monotonic() + .12, last_read + 60))
                    except TimeoutError:
                        continue
                    last_read = time.monotonic()
                    self._received(message, send)
            except (OSError, EOFError, ValueError, RecursionError) as error:
                if not self.stop.is_set():
                    text = str(error) if isinstance(error, ValueError) and '版本2' in str(error) else '服务器未连接，麦克风已暂停，正在重连。'
                    self.event('disconnected', message=text)
            finally:
                self._ready(False)
                # Request IDs survive disconnect; query status instead of resending text.
                with self.lock:
                    while True:
                        try:
                            self.outgoing.get_nowait()
                        except queue.Empty:
                            break
                if sock:
                    sock.close()
                self.socket = None
            if self.stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15)
        self.event('closed')

    def close(self, grace=2):
        self._ready(False)
        self.stop.set()
        sock = self.socket
        if sock:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.thread:
            self.thread.join(4)
