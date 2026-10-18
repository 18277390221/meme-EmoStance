"""Idle sleep policy; recording heartbeat is deliberately not speech activity."""
import threading
import time
import unicodedata


def is_sleep_command(text):
    if not isinstance(text, str):
        return False
    normalized = unicodedata.normalize('NFKC', text).casefold()
    normalized = ''.join(c for c in normalized
                         if not c.isspace() and not unicodedata.category(c).startswith('P'))
    return normalized == '去睡觉吧meme'


class SleepIdle:
    def __init__(self, seconds=20, clock=time.monotonic):
        self.seconds, self.clock = seconds, clock
        self.lock = threading.RLock()
        self.last = clock()
        self.speaking = False
        self.stage = 'awake'

    def activity(self, speaking=None):
        with self.lock:
            self.last = self.clock()
            if speaking is not None:
                self.speaking = speaking

    def due(self, eligible):
        with self.lock:
            now = self.clock()
            if not eligible or self.speaking:
                self.last = now
                return False
            return self.stage == 'awake' and now - self.last >= self.seconds

    def install_callbacks(self, speech, on_sleep=None):
        # Callbacks run in the IPC reader, before GUI polling can enter sleep.
        for name, speaking in [('on_speech_start', True), ('on_speech_cancel', False), ('on_final', False),
                               ('on_music_stop', False), ('on_music_utterance', False)]:
            original = getattr(speech, name, None)
            def callback(*args, _original=original, _speaking=speaking,
                         _final=(name == 'on_final'), **kwargs):
                self.activity(_speaking)
                text = args[0] if args else kwargs.get('text')
                if _final and on_sleep is not None and is_sleep_command(text):
                    on_sleep()
                    return None  # Local command must not create a server request.
                if _original:
                    return _original(*args, **kwargs)
            setattr(speech, name, callback)


class SleepController:
    def __init__(self, window, desktop, muted=False, screen_factory=None):
        self.window, self.desktop = window, desktop
        self.idle = SleepIdle()
        self.closed = False
        self.voice_requested = False
        self.idle.install_callbacks(desktop.speech, self.request_sleep)
        self.muted, self.screen_factory = muted, screen_factory
        self.screen = None
        self.disabled = False
        self.pause_at = 0.0
        self.resume_owned = False
        self.job = window.after(100, self.tick)
        window.bind('<ButtonPress-1>', lambda event: self.idle.activity(), add='+')

    def request_sleep(self):
        # IPC reader thread: record intent only; Tk and playback stay on UI thread.
        with self.idle.lock:
            if not self.closed:
                self.voice_requested = True

    def busy(self):
        d = self.desktop
        return (d.turn.phase != 'listening' or d.turn.media_active or d.playing or
                d.audio.is_playing() or d.pending_event is not None or d.ready_media is not None or
                d.restarting or (d.speech_pane is not None and d.speech_pane.winfo_exists()) or
                (d.manager is not None and d.manager.winfo_exists()))

    def tick(self):
        self.job = None
        if self.closed or self.desktop.closed:
            return
        d = self.desktop
        try:
            healthy = d.speech.running() and d.turn.enabled and d.link.connected
            if self.idle.stage == 'awake' and self.voice_requested:
                with self.idle.lock:
                    self.voice_requested = False
                self.resume_owned = True
                if self.disabled or not healthy:
                    self.leave()
                else:
                    self.idle.stage = 'command_wait'
                    self.pause_at = time.monotonic()
                    d.speech.pause()
                    d.notice = '已识别待机指令，等当前表情和声音完整结束后进入待机。'
            elif self.idle.stage == 'command_wait':
                if self.idle.speaking or not healthy:
                    self.leave()
                elif not self.busy():
                    self.idle.stage = 'pausing'
                    self.pause_at = time.monotonic()
                elif not d.speech.paused and time.monotonic() - self.pause_at > 5:
                    raise ValueError('待机指令等待麦克风关闭超时，已取消。')
            elif self.idle.stage == 'awake':
                eligible = (not self.disabled and healthy and not self.busy() and
                            d.speech_state == 'listening' and not d.speech.paused and
                            time.monotonic() - d.speech.last_activity < 1.5)
                if self.idle.due(eligible):
                    with self.idle.lock:
                        if not self.idle.speaking:
                            self.idle.stage = 'pausing'
                            self.pause_at = time.monotonic()
                            self.resume_owned = True
                            d.speech.pause()
            elif self.idle.stage == 'pausing':
                # Wait for the worker's matching ALSA-closed acknowledgement.
                if self.idle.speaking or self.busy() or not healthy:
                    self.leave()
                elif d.speech.paused:
                    if self.screen_factory is None:
                        from sleep_overlay import SleepScreen
                        factory = SleepScreen
                    else:
                        factory = self.screen_factory
                    self.screen = factory(self.window, on_wake=self.leave, muted=self.muted)
                    self.idle.stage = 'sleeping'
                elif time.monotonic() - self.pause_at > 5:
                    raise ValueError('息屏等待麦克风关闭超时，已取消息屏。')
            elif self.idle.stage == 'sleeping':
                if self.idle.speaking or self.busy() or not healthy:
                    self.leave()
        except (OSError, ValueError, RuntimeError) as error:
            self.disabled = True
            d.notice = '息屏已停用：' + str(error)
            self.leave()
        if not self.closed:
            self.job = self.window.after(100, self.tick)

    def leave(self):
        with self.idle.lock:
            self.voice_requested = False
        if self.screen is not None:
            screen, self.screen = self.screen, None
            screen.close()
        self.idle.stage = 'awake'
        self.idle.activity()
        if self.resume_owned:
            self.resume_owned = False
            d = self.desktop
            if (not self.closed and not d.closed and d.turn.enabled and
                    d.turn.phase in ('listening', 'playing')):
                d.speech.resume()

    def close(self):
        self.closed = True
        if self.job is not None:
            self.window.after_cancel(self.job)
            self.job = None
        self.leave()
