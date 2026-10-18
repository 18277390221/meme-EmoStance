"""Pi adaptation of repository SleepScreen: original frames 0..14 / wake 15+.

GIF decode/scale runs on one worker with a bounded cache and latest-only mailbox.
The supplied sleep WAV loops only after microphone pause acknowledgement.
"""
from pathlib import Path
import threading
import time
import tkinter as tk

from PIL import Image, ImageTk
from native_core import Audio
from render_pipeline import CachedFrames


class SleepScreen:
    def __init__(self, parent, *, on_wake=None, muted=False):
        self.parent, self.on_wake = parent, on_wake
        self.folder = Path(__file__).resolve().parent
        for name in ('sleep.gif', 'sleep.wav'):
            if not (self.folder / name).is_file():
                raise ValueError('缺少睡眠素材：' + name)
        self.closed = False
        self.waking = False
        self.number = 0
        self.shown = None
        self.deadline = 0.0
        self.photo = None
        self.audio = Audio(muted)
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
            widget.bind('<ButtonPress-1>', self.wake)
        self.overlay.bind('<Escape>', self.wake)
        self.overlay.focus_set()
        self.overlay.bind('<Destroy>', self.destroyed)
        self.worker = threading.Thread(target=self.decode, daemon=True, name='sleep-render')
        self.worker.start()
        self.job = parent.after(15, self.poll)

    def decode(self):
        reader = None
        try:
            with Image.open(self.folder / 'sleep.gif') as gif:
                count = gif.n_frames
            if count < 16:
                raise ValueError('睡眠GIF不足16帧，无法播放原唤醒片段。')
            wav = (self.folder / 'sleep.wav').read_bytes()
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
                    reader = CachedFrames((self.folder / 'sleep.gif').read_bytes(), *size,
                                          budget=16 * 1024 * 1024)
                _, image, delay = reader.frame(number)
                with self.condition:
                    if not self.closed and serial == self.request_serial:
                        self.result = (serial, number, size, image, delay, count, wav)
        except Exception as error:
            with self.condition:
                self.result = error
        finally:
            if reader is not None:
                reader.close()

    def poll(self):
        self.job = None
        if self.closed:
            return
        now = time.monotonic()
        with self.condition:
            result, self.result = self.result, None
        if isinstance(result, Exception):
            self.close(woken=True)
            return
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
                    self.close(woken=True)
                    return
            else:
                self.number = (self.number + 1) % 15
            self.shown = None
        size = (max(1, self.overlay.winfo_width()), max(1, self.overlay.winfo_height()))
        wanted = (self.number, size)
        if wanted != self.requested:
            with self.condition:
                self.request_serial += 1
                self.request = (self.request_serial, self.number, size)
                self.requested = wanted
                self.condition.notify_all()
        if self.wav is not None and not self.waking and not self.audio.muted and not self.audio.is_playing():
            try:
                if self.audio.pop_errors():
                    self.wav = None
                else:
                    self.audio.play(self.wav)
            except (ValueError, OSError):
                self.wav = None
        self.job = self.parent.after(15, self.poll)

    def wake(self, event=None):
        if self.closed or self.waking:
            return 'break'
        self.waking = True
        self.audio.stop()
        self.number = 15
        self.shown = None
        self.requested = None
        with self.condition:
            self.request_serial += 1
            self.result = None
        return 'break'

    def destroyed(self, event):
        if event.widget is self.overlay:
            self.close()

    def close(self, *, woken=False):
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
        if woken and self.on_wake is not None:
            self.on_wake()
