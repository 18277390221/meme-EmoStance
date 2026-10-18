"""Tk standby GIF: sleep frames 0..14 loop, wake frames 15..end once.

Decode and scale run off the UI thread with a bounded cache.  Touch only requests
wake; the controller closes capture before calling ``wake``.  Completion happens
after the final wake frame's own duration, never on a speech or touch event.
"""
from pathlib import Path
import threading
import time


class TkSleepScreen:
    def __init__(self, parent, *, directory, on_wake_request=None,
                 on_wake_complete=None, on_error=None, muted=False,
                 sink_name=None):
        import tkinter as tk
        from native_core import Audio

        self.parent = parent
        self.directory = Path(directory)
        for name in ('sleep.gif', 'sleep.wav'):
            if not (self.directory / name).is_file():
                raise ValueError('缺少待机素材：' + name)
        self.on_wake_request = on_wake_request
        self.on_wake_complete = on_wake_complete
        self.on_error = on_error
        self.closed = self.waking = False
        self.number = 0
        self.shown = None
        self.deadline = 0.0
        self.photo = None
        self.audio = Audio(muted, sink_name=sink_name)
        self.wav = None
        self.condition = threading.Condition()
        self.request = None
        self.request_serial = 0
        self.result = None
        self.requested = None
        self.overlay = tk.Frame(parent, bg='black')
        self.image = tk.Label(self.overlay, bg='black', bd=0, highlightthickness=0)
        self.image.pack(fill='both', expand=True)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay.lift()
        for widget in (self.overlay, self.image):
            widget.bind('<ButtonPress-1>', self.request_wake)
        self.overlay.bind('<Escape>', self.request_wake)
        self.overlay.focus_set()
        self.overlay.bind('<Destroy>', self.destroyed)
        self.worker = threading.Thread(target=self.decode, daemon=True,
                                       name='standby-render')
        self.worker.start()
        self.job = parent.after(15, self.poll)

    def decode(self):
        reader = None
        try:
            from PIL import Image
            from render_pipeline import CachedFrames

            gif_bytes = (self.directory / 'sleep.gif').read_bytes()
            with Image.open(self.directory / 'sleep.gif') as gif:
                count = gif.n_frames
            if count < 16:
                raise ValueError('待机GIF不足16帧，无法播放唤醒片段。')
            wav = (self.directory / 'sleep.wav').read_bytes()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or self.request is not None)
                    if self.closed:
                        return
                    serial, number, size = self.request
                    self.request = None
                if reader is None or reader.size != size:
                    if reader is not None:
                        reader.close()
                    reader = CachedFrames(gif_bytes, *size, budget=16 * 1024 * 1024)
                _, image, delay = reader.frame(number)
                with self.condition:
                    if not self.closed and serial == self.request_serial:
                        self.result = (serial, number, size, image, delay, count, wav)
        except Exception as error:
            with self.condition:
                if not self.closed:
                    self.result = error
        finally:
            if reader is not None:
                reader.close()

    def poll(self):
        from PIL import ImageTk

        self.job = None
        if self.closed:
            return
        try:
            now = time.monotonic()
            with self.condition:
                result, self.result = self.result, None
            if isinstance(result, Exception):
                raise result
            if result is not None:
                serial, number, size, frame, delay, count, wav = result
                if serial == self.request_serial:
                    self.photo = ImageTk.PhotoImage(frame, master=self.overlay)
                    self.image.configure(image=self.photo)
                    self.shown = (number, size)
                    self.wav, self.count = wav, count
                    self.deadline = now + delay
            if self.shown is not None and now >= self.deadline:
                if self.waking:
                    self.number += 1
                    if self.number >= self.count:
                        self.close()
                        if self.on_wake_complete is not None:
                            self.on_wake_complete()
                        return
                else:
                    self.number = (self.number + 1) % 15
                self.shown = None
            size = (max(1, self.overlay.winfo_width()),
                    max(1, self.overlay.winfo_height()))
            wanted = (self.number, size)
            if wanted != self.requested:
                with self.condition:
                    self.request_serial += 1
                    self.request = (self.request_serial, self.number, size)
                    self.requested = wanted
                    self.condition.notify_all()
            if self.wav is not None and not self.waking and not self.audio.muted:
                errors = self.audio.pop_errors()
                if errors:
                    raise RuntimeError(errors[0])
                if not self.audio.is_playing():
                    self.audio.play(self.wav)
            self.job = self.parent.after(15, self.poll)
        except Exception as error:
            self.close()
            if self.on_error is not None:
                self.on_error(error)

    def request_wake(self, event=None):
        if not self.closed and not self.waking and self.on_wake_request is not None:
            self.on_wake_request()
        return 'break'

    def wake(self):
        """Called by the UI controller only after the microphone-closed ACK."""
        if self.closed or self.waking:
            return
        self.waking = True
        self.audio.stop()
        self.number = 15
        self.shown = None
        self.requested = None
        with self.condition:
            self.request_serial += 1
            self.result = None

    def destroyed(self, event):
        if event.widget is self.overlay and not self.closed:
            self.close()
            if self.on_error is not None:
                self.on_error(RuntimeError('待机画面意外关闭，已取消待机。'))

    def close(self):
        if self.closed:
            return
        with self.condition:
            self.closed = True
            self.request = self.result = None
            self.condition.notify_all()
        if self.job is not None:
            self.parent.after_cancel(self.job)
            self.job = None
        self.audio.stop()
        if self.overlay.winfo_exists():
            self.overlay.destroy()
