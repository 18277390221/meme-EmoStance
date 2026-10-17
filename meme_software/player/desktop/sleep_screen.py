"""在现有 Tkinter 窗口内显示可点击唤醒的睡猫画面。"""

from __future__ import annotations

import tkinter as tk
import winsound
from pathlib import Path
from typing import Callable
from weakref import WeakKeyDictionary

from PIL import Image, ImageOps, ImageTk


ASSET_DIR = Path(__file__).resolve().parent
GIF_PATH = ASSET_DIR / "sleep.gif"
WAV_PATH = ASSET_DIR / "sleep.wav"

# 原 GIF 的 0～14 帧闭眼；点击后播放第 15 帧到最后一帧（含绿色特效）。
SLEEP_FRAMES = tuple(range(15))
WAKE_START_FRAME = 15


class SleepScreen:
    """铺满父窗口；点击后播放睁眼片段并移除画面。"""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_wake: Callable[[], None] | None = None,
    ) -> None:
        for path in (GIF_PATH, WAV_PATH):
            if not path.is_file():
                raise FileNotFoundError(f"找不到素材：{path}")

        self.parent = parent
        self.on_wake = on_wake
        self._closed = False
        self._waking = False
        self._sleep_position = 1
        self._wake_position = 0
        self._shown_frame = SLEEP_FRAMES[0]
        self._after_id: str | None = None
        self._resize_after_id: str | None = None
        self._render_size: tuple[int, int] | None = None

        # Pillow 一次打开 GIF，并正确合成每帧；不再反复让 Tk 解码整个文件。
        self._source_frames: dict[int, Image.Image] = {}
        self._durations: dict[int, int] = {}
        with Image.open(GIF_PATH) as gif:
            self._wake_frames = tuple(range(WAKE_START_FRAME, gif.n_frames))
            for number in (*SLEEP_FRAMES, *self._wake_frames):
                gif.seek(number)
                self._source_frames[number] = gif.convert("RGB").copy()
                self._durations[number] = max(20, gif.info.get("duration", 67))

        self.overlay = tk.Frame(parent, bg="black", cursor="hand2")
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay.lift()
        self.image = tk.Label(
            self.overlay, bg="black", bd=0, highlightthickness=0, cursor="hand2"
        )
        self.image.pack(fill="both", expand=True)

        # 点击图片和图片周围的区域都能唤醒。
        self.overlay.bind("<Button-1>", self._start_waking)
        self.image.bind("<Button-1>", self._start_waking)
        self.overlay.bind("<Escape>", lambda _event: self.close())
        self.overlay.bind("<Destroy>", self._on_destroy)
        self.overlay.bind("<Configure>", self._schedule_resize)
        self.overlay.focus_set()

        self._frames: dict[int, ImageTk.PhotoImage] = {}
        self._render_frames()
        winsound.PlaySound(
            str(WAV_PATH),
            winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
        )
        self._after_id = self.overlay.after(
            self._durations[self._shown_frame], self._show_sleep_frame
        )

    def _render_frames(self) -> None:
        self._resize_after_id = None
        if self._closed:
            return
        width = self.overlay.winfo_width()
        height = self.overlay.winfo_height()
        if width < 2 or height < 2:
            width = max(1, self.parent.winfo_width())
            height = max(1, self.parent.winfo_height())
        size = (width, height)
        if size == self._render_size:
            return

        # 等比铺满窗口，只裁掉边缘，避免拉伸猫脸。
        self._frames = {
            number: ImageTk.PhotoImage(
                ImageOps.fit(frame, size, Image.Resampling.LANCZOS, centering=(0.5, 0.35)),
                master=self.overlay,
            )
            for number, frame in self._source_frames.items()
        }
        self._render_size = size
        self.image.configure(image=self._frames[self._shown_frame])

    def _schedule_resize(self, event: tk.Event) -> None:
        if self._closed or (event.width, event.height) == self._render_size:
            return
        if self._resize_after_id is not None:
            self.overlay.after_cancel(self._resize_after_id)
        self._resize_after_id = self.overlay.after(120, self._render_frames)

    def _show_frame(self, number: int) -> None:
        self._shown_frame = number
        self.image.configure(image=self._frames[number])

    def _show_sleep_frame(self) -> None:
        self._after_id = None
        if self._closed or self._waking:
            return
        number = SLEEP_FRAMES[self._sleep_position]
        self._show_frame(number)
        self._sleep_position = (self._sleep_position + 1) % len(SLEEP_FRAMES)
        self._after_id = self.overlay.after(
            self._durations[number], self._show_sleep_frame
        )

    def _start_waking(self, _event: tk.Event) -> None:
        if self._closed or self._waking:
            return
        self._waking = True
        winsound.PlaySound(None, 0)
        if self._after_id is not None:
            self.overlay.after_cancel(self._after_id)
            self._after_id = None
        self._show_next_wake_frame()

    def _show_next_wake_frame(self) -> None:
        self._after_id = None
        if self._closed:
            return
        if self._wake_position == len(self._wake_frames):
            self.close(woken=True)
            return
        number = self._wake_frames[self._wake_position]
        self._show_frame(number)
        self._wake_position += 1
        self._after_id = self.overlay.after(
            self._durations[number], self._show_next_wake_frame
        )

    def _on_destroy(self, event: tk.Event) -> None:
        if event.widget is self.overlay:
            self.close()

    def close(self, *, woken: bool = False) -> None:
        """关闭息屏画面；睁眼片段播完时触发 on_wake。"""
        if self._closed:
            return
        self._closed = True
        for after_id in (self._after_id, self._resize_after_id):
            if after_id is not None:
                self.overlay.after_cancel(after_id)
        winsound.PlaySound(None, 0)
        if self.overlay.winfo_exists():
            self.overlay.destroy()
        if _screens.get(self.parent) is self:
            del _screens[self.parent]
        if woken and self.on_wake is not None:
            self.on_wake()


_screens: WeakKeyDictionary[tk.Misc, SleepScreen] = WeakKeyDictionary()


def set_sleep_screen(
    parent: tk.Misc,
    enabled: bool,
    *,
    on_wake: Callable[[], None] | None = None,
) -> SleepScreen | None:
    """显示或移除睡猫画面；重复启用不会创建多个画面。"""
    current = _screens.get(parent)
    if enabled:
        if current is not None and not current._closed:
            return current
        screen = SleepScreen(parent, on_wake=on_wake)
        _screens[parent] = screen
        return screen
    if current is not None:
        current.close()
    return None


if __name__ == "__main__":
    from sleep_demo import main

    main()
