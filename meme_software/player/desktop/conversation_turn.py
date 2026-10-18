"""Strict half-duplex: ready -> listen -> infer -> prepare -> play -> listen."""
import threading
import time

STANDBY_PHASES = ('standby_preparing', 'standby', 'standby_waking')


class ConversationTurn:
    def __init__(self, send, pause, resume, cooldown=.25, speech_start=None,
                 speech_cancel=None, playback_done=None):
        self.send, self.pause, self.resume = send, pause, resume
        self.send_speech_start = speech_start or (lambda: None)
        self.send_speech_cancel = speech_cancel or (lambda: None)
        self.send_playback_done = playback_done or (lambda request_id, failed=False: None)
        self.cooldown = cooldown
        self.lock = threading.RLock()
        self.enabled = False
        self.ready = False
        self.phase = 'idle'
        self.request_id = None
        self.resume_at = None
        self.media_active = False
        self.listen_requested = False
        self.speech_active = False

    def _listen(self):
        if (self.enabled and self.ready and not self.media_active and
                self.phase in ('listening', 'music', 'standby') and not self.listen_requested):
            self.listen_requested = True
            self.resume()

    def _pause(self):
        self.listen_requested = False
        self.pause()

    def _available(self):
        return 'listening' if self.enabled and self.ready else ('connecting' if self.enabled else 'idle')

    def set_ready(self, ready):
        """Called directly by the socket thread, so disconnect immediately closes ASR."""
        with self.lock:
            self.ready = bool(ready)
            if not self.ready:
                self.speech_active = False
                self._pause()
            if self.phase in ('idle', 'listening', 'connecting'):
                self.phase = self._available()
                self._listen()

    def enable(self):
        with self.lock:
            self.enabled = True
            if self.phase not in ('preparing', 'playing', 'waiting', 'cooldown', 'music_preparing', 'music', 'music_stopping') + STANDBY_PHASES:
                self.phase = self._available()
            self._listen()

    def disable(self):
        with self.lock:
            self.enabled = False
            self.resume_at = None
            if self.speech_active:
                self.send_speech_cancel()
            self.speech_active = False
            # An already submitted request survives a voice-toggle/reconnect.
            if self.phase not in ('preparing', 'playing', 'waiting', 'music_preparing', 'music', 'music_stopping') + STANDBY_PHASES:
                self.phase = 'idle'
            self._pause()

    def speech_started(self):
        with self.lock:
            if self.enabled and self.ready and self.phase == 'listening' and not self.speech_active:
                self.speech_active = True
                self.send_speech_start()

    def speech_cancelled(self):
        with self.lock:
            if self.speech_active:
                self.send_speech_cancel()
            self.speech_active = False
            if self.phase in ('listening', 'music'):
                # Worker has already released ALSA before cancellation is emitted.
                self.listen_requested = False
                self._listen()

    def submit_final(self, text):
        with self.lock:
            if not self.enabled or not self.ready or self.phase != 'listening':
                return None
            if not isinstance(text, str) or not text.strip():
                self.speech_cancelled()
                return None
            self.phase = 'waiting'
            self.speech_active = False
            self._pause()  # Worker emitted final only after closing capture.
            self.request_id = self.send(text)
            if self.request_id is None:
                self.send_speech_cancel()
                self.failure()
            return self.request_id

    def respond(self, body, handler, prior_requests):
        with self.lock:
            request_id = body.get('request_id') if isinstance(body, dict) else None
            if isinstance(request_id, str) and request_id in prior_requests:
                return handler(body)
            if self.phase == 'waiting':
                if request_id != 'server-' + str(self.request_id):
                    raise ValueError('正在等待本轮服务器回复，请稍候。')
            elif self.phase not in ('idle', 'listening', 'connecting'):
                raise ValueError('小猫正在回复，请等本轮播放结束。')
            elif isinstance(request_id, str) and request_id.startswith('server-'):
                raise ValueError('此服务器回复不属于当前对话，已忽略。')
            result = handler(body)
            self.phase = 'preparing'
            self._pause()
            return result

    def can_start_media(self):
        with self.lock:
            return not self.media_active and self.phase not in ('music_preparing', 'music', 'music_stopping') + STANDBY_PHASES

    def standby_prepare(self):
        with self.lock:
            if not self.enabled or not self.ready or self.phase != 'listening' or self.media_active:
                return False
            if self.speech_active:
                self.send_speech_cancel()
            self.speech_active = False
            self.phase = 'standby_preparing'
            self._pause()
            return True

    def standby_started(self):
        with self.lock:
            if self.phase != 'standby_preparing' or not self.enabled or not self.ready:
                return False
            self.phase = 'standby'
            self._listen()
            return True

    def standby_rearm(self):
        with self.lock:
            if self.phase == 'standby':
                self.listen_requested = False
                self._listen()

    def standby_waking(self):
        with self.lock:
            if self.phase != 'standby':
                return False
            self.phase = 'standby_waking'
            self._pause()
            return True

    def standby_finished(self):
        with self.lock:
            if self.phase in STANDBY_PHASES:
                self._media_complete()

    def standby_abort(self):
        with self.lock:
            if self.phase in STANDBY_PHASES:
                self.resume_at = None
                self.speech_active = False
                self.media_active = False
                self.phase = self._available()
                self._pause()

    def music_prepare(self):
        """Reserve local output without submitting a model request or chat turn."""
        with self.lock:
            if not self.enabled or not self.ready or self.phase != 'listening' or self.media_active:
                return False
            if self.speech_active:
                self.send_speech_cancel()
            self.speech_active = False
            self.phase = 'music_preparing'
            self._pause()
            return True

    def music_started(self):
        with self.lock:
            if self.phase == 'music_preparing' and self.enabled and self.ready:
                self.phase = 'music'
                self._listen()

    def music_rearm(self):
        with self.lock:
            if self.phase == 'music':
                self.speech_active = False
                self.listen_requested = False
                self._listen()

    def music_finished(self):
        with self.lock:
            if self.phase not in ('music_preparing', 'music', 'music_stopping'):
                return
            self._media_complete()

    def music_stopping(self):
        with self.lock:
            if self.phase in ('music_preparing', 'music'):
                self.phase = 'music_stopping'
                self._pause()

    def media_started(self):
        with self.lock:
            self.media_active = True
            if self.phase != 'waiting':
                self.phase = 'playing'
            self._pause()

    def _media_complete(self, failed=False, now=None):
        self.media_active = False
        if self.request_id is not None:
            self.send_playback_done(self.request_id, failed=failed)
            self.request_id = None
        self.speech_active = False
        if self.enabled:
            self.phase = 'cooldown'
            self.resume_at = (time.monotonic() if now is None else now) + self.cooldown
        else:
            self.phase = 'idle'
            self.resume_at = None
        self._pause()

    def media_finished(self, now=None):
        with self.lock:
            if self.phase == 'playing':
                self._media_complete(now=now)
            else:
                self.media_active = False

    def media_cancelled(self):
        with self.lock:
            if self.phase in ('preparing', 'playing'):
                self._media_complete(failed=True)
            else:
                self.media_active = False

    def failure(self):
        with self.lock:
            if self.phase != 'waiting':
                return
            self.request_id = None
            self.speech_active = False
            self.phase = self._available()
            self._listen()

    def tick(self, now=None):
        with self.lock:
            now = time.monotonic() if now is None else now
            if self.phase == 'cooldown' and self.resume_at is not None and now >= self.resume_at:
                self.resume_at = None
                self.phase = self._available()
                self._listen()
