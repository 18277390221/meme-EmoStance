"""Listen during playback; queue one final reply until current media completes."""
import threading
import time


class ConversationTurn:
    def __init__(self, send, pause, resume, cooldown=.25):
        self.send, self.pause, self.resume = send, pause, resume
        self.cooldown = cooldown
        self.lock = threading.RLock()
        self.enabled = False
        self.phase = 'idle'
        self.request_id = None
        self.resume_at = None
        self.media_active = False
        self.listen_requested = False

    def _listen(self):
        if self.enabled and not self.listen_requested:
            self.listen_requested = True
            self.resume()

    def _pause(self):
        self.listen_requested = False
        self.pause()

    def enable(self):
        with self.lock:
            self.enabled = True
            if self.phase not in ('preparing', 'playing', 'waiting'):
                self.phase = 'listening'
            if self.phase in ('listening', 'playing'):
                self._listen()

    def disable(self):
        with self.lock:
            self.enabled = False
            self.request_id = None
            self.resume_at = None
            if self.phase not in ('preparing', 'playing'):
                self.phase = 'idle'
            self._pause()

    def submit_final(self, text):
        with self.lock:
            if not self.enabled or self.phase not in ('listening', 'playing'):
                return None
            if not isinstance(text, str) or not text.strip():
                return None
            self.phase = 'waiting'
            self._pause()
            self.request_id = self.send(text)
            if self.request_id is None:
                self.failure()
            return self.request_id

    def respond(self, body, handler, prior_requests):
        """Gate before changing the player's event, including local HTTP/manual replies."""
        with self.lock:
            request_id = body.get('request_id') if isinstance(body, dict) else None
            if isinstance(request_id, str) and request_id in prior_requests:
                return handler(body)  # Preserve idempotence without replay/re-pausing.
            if self.phase == 'waiting':
                if request_id != 'server-' + str(self.request_id):
                    raise ValueError('正在等待本轮服务器回复，请稍候。')
            elif self.phase not in ('idle', 'listening'):
                raise ValueError('小猫正在回复；新一轮只接受用户说完后的对应服务器回复。')
            elif isinstance(request_id, str) and request_id.startswith('server-'):
                raise ValueError('此服务器回复不属于当前对话，已忽略。')
            result = handler(body)  # Validation failure must not reserve the turn.
            self.phase = 'preparing'
            self.request_id = None
            self._pause()
            return result

    def can_start_media(self):
        with self.lock:
            return not self.media_active

    def media_started(self):
        with self.lock:
            self.media_active = True
            if self.phase == 'preparing':
                self.phase = 'playing'
                self._listen()

    def media_finished(self, now=None):
        with self.lock:
            self.media_active = False
            # waiting: final already sent; preparing: its validated response is
            # queued in player.event. Neither may reopen the microphone here.
            if self.phase == 'playing':
                self.phase = 'listening' if self.enabled else 'idle'
                self._listen()  # Already listening: do not reset a partial phrase.

    def media_cancelled(self):
        with self.lock:
            self.media_active = False
            self.request_id = None
            self.phase = 'listening' if self.enabled else 'idle'
            self._listen()

    def failure(self):
        with self.lock:
            if self.phase != 'waiting':
                return
            self.request_id = None
            self.phase = ('playing' if self.media_active else 'listening') if self.enabled else 'idle'
            self._listen()

    def tick(self, now=None):
        # Retained for the existing UI interface. Playback no longer gates
        # listening, so it must not reset or cool down an active ASR stream.
        pass
