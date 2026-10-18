"""Local song routing and playback ownership, independent of model requests."""
from collections import OrderedDict, deque
from pathlib import Path
import threading

from .catalog import MusicCatalog
from .core import MusicContext, is_music_request, is_stop_request
from .playback import MusicPlayer


class MusicController:
    def __init__(self, turn, directory, speech_control, microphone_paused,
                 player=None, catalog_factory=MusicCatalog, context=None):
        self.turn = turn
        self.directory = Path(directory).expanduser()
        self.speech_control = speech_control
        self.microphone_paused = microphone_paused
        self.player = player if player is not None else MusicPlayer(volume=30)
        self.catalog_factory = catalog_factory
        self.context = context if context is not None else MusicContext()
        self.lock = threading.RLock()
        self.state = 'idle'
        self.token = None
        self.track_id = self.track_title = None
        self.last_error = None
        self.pending_song = None
        self.pending_normal = None
        self.pending_chat = OrderedDict()
        self.events = deque()
        self.closed = False

    @property
    def active(self):
        with self.lock:
            return self.state != 'idle'

    def status(self):
        with self.lock:
            return {'state': self.state, 'playing': self.state == 'playing' and self.player.is_playing(),
                    'track_id': self.track_id, 'last_error': self.last_error}

    def command(self, body):
        """Narrow localhost API: intents only, never paths or caller-picked files."""
        if not isinstance(body, dict) or set(body) - {'action', 'text'}:
            raise ValueError('音乐请求只接受action和点歌text字段。')
        action = body.get('action')
        if action not in ('play', 'stop'):
            raise ValueError('音乐action只支持play或stop。')
        with self.lock, self.turn.lock:
            if self.closed:
                raise ValueError('音乐功能已关闭。')
            if action == 'stop':
                if 'text' in body:
                    raise ValueError('停止音乐不需要text字段。')
                self._reconcile_exit()
                self._stop()
            else:
                text = body.get('text', '我想听音乐')
                if (not isinstance(text, str) or not text.strip() or len(text) > 2000 or
                        not is_music_request(text)):
                    raise ValueError('text需要是有效的点歌请求。')
                if (self.state != 'idle' or not self.turn.enabled or not self.turn.ready or
                        self.turn.phase != 'listening' or self.turn.media_active):
                    raise ValueError('请等待服务器就绪并结束当前回复后再点歌。')
                self.pending_normal = None
                self.handle_final(text)
            return self.status()

    def handle_final(self, text):
        """IPC callback: no Tk work, no model call for songs or captured lyrics."""
        with self.lock:
            if self.closed:
                return None
            ended = self._reconcile_exit()
            if self.state != 'idle':
                if self.state == 'playing' and self.player.is_playing() and is_stop_request(text):
                    self._stop()
                elif self.state == 'playing':
                    self.turn.music_rearm()
                return None
            if ended and not is_music_request(text):
                # This is an actual normal-final callback, not a music partial.
                # Keep it through the short output/capture handover cooldown.
                self.pending_normal = text
                return None
            if is_music_request(text):
                if self.turn.music_prepare():
                    self.pending_song = text
                    self.state = 'preparing'
                    self.last_error = None
                    self.events.append({'type': 'preparing', 'message': '正在选一首适合现在心情的歌。'})
                return None
            return self._submit_chat(text)

    def _submit_chat(self, text):
        key = self.turn.submit_final(text)
        if key is not None:
            self.pending_chat[key] = text
            while len(self.pending_chat) > 256:
                self.pending_chat.popitem(last=False)
        return key

    def _reconcile_exit(self):
        if self.state == 'playing' and not self.player.is_playing():
            self._complete('finished', '这首歌播完啦。')
            return True
        return False

    def observe_reply(self, event):
        with self.lock:
            user = self.pending_chat.pop(event.get('id'), None)
            if user is not None:
                self.context.observe_reply(event.get('conversation_id'), user, event.get('reply_text', ''),
                                           meme_id=event.get('meme_id'))

    def forget_request(self, request_id):
        with self.lock:
            self.pending_chat.pop(request_id, None)

    def music_stop(self, token=None):
        with self.lock:
            if self.state == 'playing' and token == self.token:
                if not self._reconcile_exit():
                    self._stop()

    def music_utterance(self, token=None):
        with self.lock:
            if self.state == 'playing' and token == self.token:
                if self._reconcile_exit():
                    return
                # Each endpoint closed its capture window. Lyrics stay local;
                # start another control-only window so later stop remains audible.
                self.turn.music_rearm()

    def _complete(self, kind, message=None):
        self.speech_control(False, None)
        self.state = 'idle'
        self.pending_song = None
        self.token = None
        self.track_id = self.track_title = None
        self.turn.music_finished()
        if message:
            self.events.append({'type': kind, 'message': message})

    def _stop(self):
        if self.state in ('idle', 'stopping'):
            return
        self.pending_song = None
        self.speech_control(False, None)
        if self.token is None:
            self._complete('stopped', '音乐已停止。')
            return
        self.state = 'stopping'
        self.turn.music_stopping()
        self.player.stop()

    def set_ready(self, ready):
        with self.lock:
            self.turn.set_ready(ready)
            if not ready:
                self.pending_normal = None
                self._stop()

    def disable(self):
        with self.lock:
            self.pending_normal = None
            self.turn.disable()
            self._stop()

    def poll(self):
        """UI poll: wait for capture close before starting any speaker output."""
        with self.lock:
            if self.closed:
                return []
            if self.pending_normal is not None and self.turn.phase == 'listening':
                text, self.pending_normal = self.pending_normal, None
                key = self._submit_chat(text)
                if key is not None:
                    self.events.append({'type': 'chat_sent', 'id': key,
                                        'message': '文字已发出，等待服务器回复。'})
            if self.pending_song is not None and self.microphone_paused():
                request, self.pending_song = self.pending_song, None
                try:
                    catalog = self.catalog_factory(self.directory)
                    selection = self.context.choose(catalog, request)
                    self.track_id, self.track_title = selection.track.id, selection.track.title
                    self.token = self.player.start(selection.track)
                except (OSError, ValueError, RuntimeError) as error:
                    self.last_error = str(error)[:300]
                    self._complete('error', '音乐播放失败：' + self.last_error)
            for event in self.player.drain():
                if self.token is None or event.get('token') != self.token:
                    continue
                kind = event.get('type')
                if kind == 'started':
                    if self.state != 'preparing':
                        continue
                    self.state = 'playing'
                    # Pipe commands are ordered: control mode is enabled before
                    # resume, so the next stream cannot leak lyrics into chat.
                    self.speech_control(True, self.token)
                    self.turn.music_started()
                    self.events.append({'type': 'started', 'message': '正在播放：' + self.track_title})
                elif kind == 'error':
                    self.last_error = str(event.get('message', '本地播放器异常'))[:300]
                    self._complete('error', '音乐播放失败：' + self.last_error)
                elif kind in ('finished', 'stopped'):
                    self._complete(kind, '这首歌播完啦。' if kind == 'finished' else '音乐已停止。')
            result = list(self.events)
            self.events.clear()
            return result

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.turn.disable()
            self.pending_normal = None
            self._stop()
            self.closed = True
        self.player.close()
