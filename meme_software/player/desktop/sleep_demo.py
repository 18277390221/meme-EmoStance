"""睡猫息屏模块的独立演示：运行后点击按钮进入息屏。"""

import tkinter as tk

from sleep_screen import set_sleep_screen


def main() -> None:
    root = tk.Tk()
    root.title("睡猫息屏示例")
    root.geometry("520x520")
    root.minsize(360, 360)

    status = tk.StringVar(value="点击下方按钮，猫咪会铺满这个窗口。")

    tk.Label(root, text="睡猫息屏示例", font=("Microsoft YaHei UI", 18, "bold")).pack(
        pady=(28, 8)
    )
    tk.Label(root, textvariable=status, font=("Microsoft YaHei UI", 10)).pack()

    def enter_sleep_screen() -> None:
        status.set("息屏中：点击猫咪，睁眼后自动返回。")
        set_sleep_screen(root, True, on_wake=lambda: status.set("猫咪醒了！可以再次进入息屏。"))

    tk.Button(
        root,
        text="进入睡猫息屏",
        font=("Microsoft YaHei UI", 12),
        command=enter_sleep_screen,
        padx=20,
        pady=7,
    ).pack(pady=24)

    tk.Label(
        root,
        text="按 Esc 可以提前退出息屏。",
        font=("Microsoft YaHei UI", 9),
        fg="#666666",
    ).pack()

    root.mainloop()


if __name__ == "__main__":
    main()
