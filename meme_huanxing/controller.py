"""Local standby ownership: a wake animation finishes before normal ASR resumes.

Speech callbacks only record intent.  ``poll`` (or its Tk ``tick`` wrapper) owns
all screen operations, speaker output and mode transitions.
"""
from collections import deque
from pathlib import Path
import threading
import time
import uuid

from .commands import is_sleep_command


class _PauseTimeout(RuntimeError):
    pass


class IdlePolicy:
    """A microphone heartbeat is deliberately not user activity."""

    def __init__(self, seconds=20, clock=time.monotonic):
        self.seconds, self.clock = seconds, clock
        self.last = clock()
        self.speaking = False
        self.stage = 'awake'

    def activity(self, speaking=None):
        self.last = self.clock()
        if speaking is not None:
            self.speaking = bool(speaking)

    def due(self, eligible):
        if not eligible or self.speaking:
            self.last = self.clock()
            return False
        return self.stage == 'awake' and self.clock() - self.last >= self.seconds


class StandbyController:
    """Wrap a desktop's local music/chat router with stateful voice standby.

    ``desktop.turn`` supplies standby_prepare/started/rearm/waking/finished/abort.
    ``desktop.speech`` supplies standby_control, pause and the acknowledged paused
    flag.  The real ASR callbacks must use the session token supplied to the worker.
    ``schedule=False`` lets an existing desktop UI poll drive this controller.
    """

    def __init__(self, window, desktop, directory, *, muted=False, sink_name=None,
                 screen_factory=None, clock=time.monotonic, idle_seconds=20,
                 pause_timeout=5, schedule=True):
        self.window, self.desktop = window, desktop
        self.directory = Path(directory)
        self.muted, self.sink_name = muted, sink_name
        self.screen_factory = screen_factory
        self.clock, self.pause_timeout = clock, pause_timeout
        self.lock = threading.RLock()
        self.idle = IdlePolicy(idle_seconds, clock)
        self.state = 'awake'
        self.token = None
        self.screen = None
        self.closed = False
        self.pending_sleep = False
        self.pending_wake = False
        self.pending_rearm = False
        self.pending_complete = False
        self.pending_error = None
        self.last_error = None
        self.pause_at = 0.0
        self.events = deque()
        self.job = window.after(100, self.tick) if schedule else None
        if hasattr(window, 'bind'):
            window.bind('<ButtonPress-1>', lambda event: self.activity(), add='+')

    @property
    def active(self):
        with self.lock:
            return self.state != 'awake' or self.pending_sleep

    def _state(self, state):
        self.state = self.idle.stage = state

    def status(self):
        with self.lock:
            # Health requests can run outside Tk's thread. Read only the plain
            # presentation snapshot, never widget dimensions or Tk image data.
            shown = getattr(self.screen, 'shown', None)
            frame = shown[0] if isinstance(shown, tuple) and shown else None
            if type(frame) is not int:
                frame = None
            return {'state': self.state, 'sleeping': self.state == 'sleeping',
                    'requested': self.pending_sleep, 'last_error': self.last_error,
                    'frame': frame,
                    'animation_waking': bool(getattr(self.screen, 'waking', False))}

    def activity(self, speaking=None):
        with self.lock:
            self.idle.activity(speaking)

    def speech_started(self):
        with self.lock:
            if self.closed or self.state != 'awake' or self.pending_sleep:
                return
            self.idle.activity(True)
            self.desktop.turn.speech_started()

    def speech_cancelled(self):
        with self.lock:
            self.idle.activity(False)
            if not self.closed and self.state == 'awake' and not self.pending_sleep:
                self.desktop.turn.speech_cancelled()

    def handle_final(self, text):
        """IPC thread: consume local sleep, never emit sleeping speech to chat."""
        with self.lock:
            if self.closed:
                return None
            self.idle.activity(False)
            if self.state != 'awake' or self.pending_sleep:
                # The worker normally emits only token-tagged standby controls.
                # A final from the previous capture window must stay local.
                return None
            if is_sleep_command(text):
                self.pending_sleep = True
                return None
            return self.desktop.music.handle_final(text)

    def request_sleep(self):
        with self.lock:
            if not self.closed and self.state == 'awake':
                self.idle.activity(False)
                self.pending_sleep = True
                return True
            return False

    def on_wake(self, token=None):
        with self.lock:
            if (not self.closed and self.state == 'sleeping' and
                    self.token is not None and token == self.token):
                self.pending_wake = True
                return True
            return False

    def on_sleep(self, token=None):
        """A sleep command heard in music mode belongs to that music session."""
        with self.lock:
            music = getattr(self.desktop, 'music', None)
            if (not self.closed and self.state == 'awake' and music is not None and
                    music.active and token is not None and token == music.token):
                self.idle.activity(False)
                self.pending_sleep = True
                return True
            return False

    def on_utterance(self, token=None):
        with self.lock:
            if (not self.closed and self.state == 'sleeping' and
                    self.token is not None and token == self.token):
                self.pending_rearm = True

    def command(self, body):
        """Narrow localhost intent API, also useful for controlled live checks."""
        if not isinstance(body, dict) or set(body) != {'action'}:
            raise ValueError('待机请求只接受action字段。')
        if body['action'] not in ('sleep', 'wake'):
            raise ValueError('待机action只支持sleep或wake。')
        with self.lock:
            if self.closed:
                raise ValueError('待机功能已关闭。')
            if body['action'] == 'sleep':
                if not self._healthy():
                    raise ValueError('请等待语音、设备和服务器就绪后再待机。')
                self.request_sleep()
            else:
                self.on_wake(self.token)
            return self.status()

    def busy(self):
        d = self.desktop
        music = getattr(d, 'music', None)
        if (d.turn.phase != 'listening' or d.turn.media_active or
                bool(music is not None and music.active) or
                getattr(d, 'playing', False) or d.audio.is_playing() or
                getattr(d, 'pending_event', None) is not None or
                getattr(d, 'ready_media', None) is not None or
                getattr(d, 'restarting', False)):
            return True
        for name in ('speech_pane', 'manager'):
            pane = getattr(d, name, None)
            if pane is not None and pane.winfo_exists():
                return True
        return False

    def _healthy(self):
        d = self.desktop
        return (not self.closed and not d.closed and d.turn.enabled and
                d.turn.ready and d.link.connected and d.speech.running())

    def _can_recover(self):
        d = self.desktop
        output = getattr(d, 'audio_output', None)
        return (self._healthy() and getattr(d, 'voice_wanted', True) and
                not getattr(d, 'restarting', False) and
                (self.muted or output is None or output.ready()))

    def _notice(self, kind, message):
        self.desktop.notice = message
        self.events.append({'type': kind, 'message': message})

    def _screen_complete(self, token):
        with self.lock:
            if self.state == 'waking' and self.token == token:
                self.pending_complete = True

    def _screen_error(self, error, token):
        with self.lock:
            if self.token == token and self.state != 'awake':
                self.pending_error = str(error)[:300]

    def _begin_sleep(self):
        if not self.desktop.turn.standby_prepare():
            return False
        self.token = uuid.uuid4().hex
        self.pause_at = self.clock()
        self._state('pausing')
        self._notice('preparing', '正在进入待机。')
        return True

    def _enter_screen(self):
        if self.screen_factory is None:
            from .screen import TkSleepScreen
            factory = TkSleepScreen
        else:
            factory = self.screen_factory
        token = self.token
        self.screen = factory(self.window, directory=self.directory,
            muted=self.muted, sink_name=self.sink_name,
            on_wake_request=lambda: self.on_wake(token),
            on_wake_complete=lambda: self._screen_complete(token),
            on_error=lambda error: self._screen_error(error, token))
        # Commands are ordered before resume so no sleeping stream enters chat.
        self.desktop.speech.standby_control(True, self.token)
        self._state('sleeping')
        if not self.desktop.turn.standby_started():
            raise RuntimeError('语音或服务器状态变化，已取消待机。')
        self._notice('sleeping', '小me正在待机，说“小me，别睡了”可以唤醒。')

    def poll(self):
        """UI thread only: microphone ACK gates all screen state transitions."""
        with self.lock:
            if self.closed:
                return []
            try:
                if self.pending_error is not None:
                    message, self.pending_error = self.pending_error, None
                    raise RuntimeError(message)
                if self.state != 'awake' and not self._healthy():
                    self.abort()
                elif self.state == 'awake':
                    if self.pending_sleep:
                        self.pending_sleep = False
                        if self._healthy():
                            self._state('waiting')
                            self.desktop.speech.pause()
                            music = getattr(self.desktop, 'music', None)
                            if music is not None and music.active:
                                music.command({'action': 'stop'})
                            self._notice('waiting', '等当前回复和声音完整结束后进入待机。')
                    else:
                        d = self.desktop
                        eligible = (self._healthy() and not self.busy() and
                            d.speech_state == 'listening' and not d.speech.paused and
                            self.clock() - d.speech.last_activity < 1.5)
                        if self.idle.due(eligible):
                            self._begin_sleep()
                elif self.state == 'waiting':
                    if not self.busy():
                        self._begin_sleep()
                elif self.state == 'pausing':
                    if self.desktop.speech.paused:
                        self._enter_screen()
                    elif self.clock() - self.pause_at >= self.pause_timeout:
                        raise _PauseTimeout('待机等待麦克风关闭超时，请暂停后重新开启语音。')
                elif self.state == 'sleeping':
                    if self.pending_wake:
                        self.pending_wake = False
                        self.pending_rearm = False
                        self.pause_at = self.clock()
                        self._state('wake_pausing')
                        self.desktop.turn.standby_waking()
                        # Keep wake-only routing installed until ALSA actually
                        # closes. A late capture event remains token-tagged.
                    elif self.pending_rearm:
                        self.pending_rearm = False
                        self.desktop.turn.standby_rearm()
                elif self.state == 'wake_pausing':
                    if self.desktop.speech.paused:
                        self.desktop.speech.standby_control(False, None)
                        self._state('waking')
                        self.screen.wake()
                        self._notice('waking', '小me正在起床。')
                    elif self.clock() - self.pause_at >= self.pause_timeout:
                        raise _PauseTimeout('唤醒等待麦克风关闭超时，请暂停后重新开启语音。')
                elif self.state == 'waking' and self.pending_complete:
                    self.pending_complete = False
                    if self.screen is not None:
                        self.screen.close()
                        self.screen = None
                    self.token = None
                    self._state('awake')
                    self.idle.activity(False)
                    self.desktop.turn.standby_finished()
                    self._notice('awake', '小me醒啦，可以继续聊天。')
                elif self.state == 'recovering':
                    if not self._can_recover():
                        self.abort()
                    elif self.desktop.speech.paused:
                        self._state('awake')
                        self.idle.activity(False)
                        self.desktop.turn.enable()
                        self._notice('recovered', '待机素材未能播放，已恢复普通聊天。')
                    elif self.clock() - self.pause_at >= self.pause_timeout:
                        raise _PauseTimeout('待机异常后等待麦克风关闭超时，请暂停后重新开启语音。')
            except (OSError, ValueError, RuntimeError) as error:
                self.last_error = str(error)[:300]
                recover = not isinstance(error, _PauseTimeout) and self._can_recover()
                self.abort()
                self._notice('error', '待机已取消：' + self.last_error)
                if recover:
                    # Resource/render errors must not strand otherwise healthy
                    # ASR. Wait for capture closure before normal mode resumes.
                    self._state('recovering')
                    self.pause_at = self.clock()
            result = list(self.events)
            self.events.clear()
            return result

    def tick(self):
        self.job = None
        if self.closed:
            return
        self.poll()
        if not self.closed and not self.desktop.closed:
            self.job = self.window.after(100, self.tick)

    def abort(self):
        """UI-only cancellation; never owns a microphone resume operation."""
        with self.lock:
            owned = self.state != 'awake' or self.pending_sleep or self.token is not None
            self.pending_sleep = self.pending_wake = self.pending_rearm = False
            self.pending_complete = False
            self.pending_error = None
            if owned:
                self.desktop.speech.standby_control(False, None)
                self.desktop.speech.pause()
                self.desktop.turn.standby_abort()
            if self.screen is not None:
                screen, self.screen = self.screen, None
                screen.close()
            self.token = None
            self._state('awake')
            self.idle.activity(False)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.job is not None:
                self.window.after_cancel(self.job)
                self.job = None
            self.abort()
