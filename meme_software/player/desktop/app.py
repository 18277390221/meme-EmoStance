#!/usr/bin/env python3
"""Meme猫原生桌面程序：Tk窗口，非浏览器/WebView。"""
import argparse
import base64
from contextlib import closing
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description='Meme猫原生桌面播放器（不打开浏览器）')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--windowed', action='store_true', help='以普通窗口启动；默认全屏')
    parser.add_argument('--mute', action='store_true', help='仅画面测试，不播放声音')
    parser.add_argument('--no-api', action='store_true', help='不启动本机回复接口')
    parser.add_argument('--port', type=int, default=8784)
    parser.add_argument('--check', action='store_true', help='检查依赖和数据，不打开窗口')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在1到65535之间')
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog, simpledialog
        from PIL import Image, ImageTk
        from native_core import Audio, Clip, check_library, read_asset, validate_wav
        sys.path.insert(0, str(args.root.resolve()))
        from launch import Player, Handler
        count = check_library(args.root)
        if args.check:
            print(json.dumps({'assets': count, 'tk': tk.TkVersion, 'pillow': Image.__version__,
                              'aplay': bool(shutil.which('aplay')), 'api_port': args.port,
                              'browser_required': False},
                             ensure_ascii=False))
            return 0
        player = Player(args.root)
        window = tk.Tk()
    except Exception as error:
        print('桌面播放器无法启动：' + str(error), file=sys.stderr)
        print('需要系统 Python3、python3-tk、python3-pil.imagetk；请在树莓派桌面启动。', file=sys.stderr)
        return 1

    class Desktop:
        BG = '#071e19'

        def __init__(self):
            self.closed = False
            self.clip = None
            self.photo = None
            self.animation = None
            self.resize_job = None
            self.deadline = 0
            self.next_due = 0
            self.last_revision = -1
            self.key = None
            self.reply = '你好呀，我在这里。'
            self.audio = Audio(args.mute)
            self.server = None
            self.manager = None
            self.fullscreen = not args.windowed
            window.title('Meme猫 · 小剧场')
            window.geometry('800x540')
            window.minsize(320, 240)
            window.configure(bg=self.BG)
            window.protocol('WM_DELETE_WINDOW', self.close)
            window.bind('<Escape>', lambda event: self.set_fullscreen(False))
            window.bind('<F11>', lambda event: self.set_fullscreen(not self.fullscreen))
            window.bind('<space>', lambda event: self.replay() if event.widget is self.canvas else None)
            self.canvas = tk.Canvas(window, bg=self.BG, bd=0, highlightthickness=0)
            self.canvas.pack(fill='both', expand=True)
            self.canvas.bind('<Configure>', self.resize)
            bar = tk.Frame(window, bg='#102e26')
            bar.pack(fill='x')
            for text, action in [('表情', self.scenes), ('重播', self.replay),
                                 ('管理', self.manage), ('全屏', lambda: self.set_fullscreen(not self.fullscreen)),
                                 ('退出', self.close)]:
                tk.Button(bar, text=text, command=action, bg='#163c31', fg='#ebfff5',
                          activebackground='#27634e', activeforeground='white', relief='flat',
                          font=('sans-serif', 10), padx=7, pady=5).pack(side='left', expand=True, fill='x')
            window.update_idletasks()
            self.set_fullscreen(self.fullscreen)
            if not args.no_api:
                try:
                    self.server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, player=player))
                    threading.Thread(target=self.server.serve_forever, daemon=True).start()
                except OSError as error:
                    messagebox.showerror('接口端口被占用',
                        f'本机 {args.port} 端口不可用：{error}\n请正常退出占用该端口的旧实例，或指定其他空闲端口（例如 --port 8785）。', parent=window)
                    self.close()
                    return
            scenario = self.available_scenes()[0] if self.available_scenes() else None
            if scenario:
                self.request(scenario['meme_id'], scenario['reply_text'])
            else:
                self.draw()
            window.after(30, self.poll)

        def available_scenes(self):
            keys = {item['id'] for item in player.assets()}
            return [s for s in player.library()['scenarios'] if s['meme_id'] in keys]

        def set_fullscreen(self, enabled):
            self.fullscreen = enabled
            window.attributes('-fullscreen', enabled)

        def request(self, key, text):
            try:
                player.respond({'meme_id': key, 'reply_text': text,
                                'request_id': 'desktop-' + uuid.uuid4().hex})
            except Exception as error:
                messagebox.showerror('无法播放', str(error), parent=window)

        def replay(self):
            if self.key:
                self.request(self.key, self.reply)

        def resize(self, event=None):
            if self.resize_job:
                window.after_cancel(self.resize_job)
            self.resize_job = window.after(80, self.resize_draw)

        def resize_draw(self):
            self.resize_job = None
            self.draw()

        def draw(self):
            self.canvas.delete('all')
            width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
            left, top, right, bottom = 0, 0, width, height
            if self.clip and self.clip.current is not None:
                frame = self.clip.fit(width, height)
                self.photo = ImageTk.PhotoImage(frame, master=window)
                self.canvas.create_image(width / 2, height / 2, image=self.photo)
                left, top = (width - frame.width) / 2, (height - frame.height) / 2
                right, bottom = left + frame.width, top + frame.height
            # Reply is inside the bottom of the visual, with no media name/timer.
            font = ('sans-serif', max(12, min(22, int(width / 33))), 'bold')
            text = self.canvas.create_text((left + right) / 2, bottom - 8, anchor='s',
                text=self.reply, fill='white', width=max(50, right - left - 20), font=font,
                justify='center')
            box = self.canvas.bbox(text)
            if box:
                shade = self.canvas.create_rectangle(left, max(top, box[1] - 8), right, bottom,
                                                     fill='#10251f', outline='')
                self.canvas.tag_lower(shade, text)

        def stop(self):
            if self.animation:
                window.after_cancel(self.animation)
                self.animation = None
            self.audio.stop()

        def play(self, event):
            self.stop()
            if self.clip:
                self.clip.close()
                self.clip = None
            self.key, self.reply = event['meme_id'], event['reply_text']
            try:
                row = read_asset(player, self.key)
                self.clip = Clip(row['visual_blob'], row['duration_seconds'])
                delay = self.clip.next()
                self.draw()
                self.audio.play(row['audio_blob'])
                started = time.monotonic()
                self.deadline = started + self.clip.duration
                self.next_due = started
                self.next_due += delay / 1000
                self.animation = window.after(max(1, min(delay, round(self.clip.duration * 1000))), self.tick)
            except Exception as error:
                self.stop()
                self.draw()
                messagebox.showerror('素材播放失败', str(error), parent=window)

        def tick(self):
            self.animation = None
            if self.closed or not self.clip:
                return
            if time.monotonic() >= self.deadline:
                self.audio.stop()
                return
            try:
                delay = self.clip.next()
                self.draw()
                self.next_due += delay / 1000
                self.animation = window.after(max(1, round((min(self.next_due, self.deadline) - time.monotonic()) * 1000)), self.tick)
            except Exception as error:
                self.stop()
                messagebox.showerror('画面解码失败', str(error), parent=window)

        def poll(self):
            if self.closed:
                return
            with player.lock:
                event = player.event.copy()
            if event['revision'] != self.last_revision:
                self.last_revision = event['revision']
                if event['type'] == 'response':
                    self.play(event)
                elif event['type'] == 'stop':
                    self.stop()
            for error in self.audio.pop_errors():
                messagebox.showwarning('声音未能播放', error, parent=window)
            window.after(50, self.poll)

        def scenes(self):
            pane = tk.Toplevel(window)
            pane.title('选择表情')
            pane.geometry('300x300')
            for scenario in self.available_scenes():
                def select(s=scenario):
                    self.request(s['meme_id'], s['reply_text'])
                    pane.destroy()
                ttk.Button(pane, text=scenario['label'], command=select).pack(fill='x', padx=16, pady=5)

        def backup(self):
            folder = args.root / 'data/backups'
            folder.mkdir(exist_ok=True)
            path = folder / ('assets-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6] + '.sqlite3')
            with closing(player.connect()) as source, closing(sqlite3.connect(path)) as target:
                source.backup(target)

        def import_pair(self):
            try:
                visual = filedialog.askopenfilename(parent=self.manager, title='选择表情',
                    filetypes=[('表情图片', '*.gif *.png *.jpg *.jpeg *.webp')])
                if not visual:
                    return
                audio = filedialog.askopenfilename(parent=self.manager, title='选择配套WAV声音',
                    filetypes=[('WAV声音', '*.wav')])
                if not audio:
                    return
                vp, ap = Path(visual), Path(audio)
                if vp.stat().st_size > 20 * 1024 * 1024 or ap.stat().st_size > 20 * 1024 * 1024:
                    raise ValueError('每个素材文件不能超过20MB。')
                vb, ab = vp.read_bytes(), ap.read_bytes()
                duration = validate_wav(ab)
                with Image.open(BytesIO(vb)) as image:
                    kind = image.format
                    image.verify()
                types = {'GIF': 'image/gif', 'PNG': 'image/png', 'JPEG': 'image/jpeg', 'WEBP': 'image/webp'}
                if kind not in types or not .5 <= duration <= 120:
                    raise ValueError('格式不支持，或声音长度不在0.5到120秒之间。')
                name = simpledialog.askstring('素材名称', '给这套表情取个名字：', initialvalue=vp.stem[:40], parent=self.manager)
                if not name:
                    return
                item = {'id': 'user-' + uuid.uuid4().hex, 'label': name,
                        'visual_type': 'gif' if kind == 'GIF' else 'image', 'duration_seconds': duration,
                        'scene_index': None,
                        'visual': {'name': vp.name, 'type': types[kind], 'data': base64.b64encode(vb).decode('ascii')},
                        'audio': {'name': ap.name, 'type': 'audio/wav', 'data': base64.b64encode(ab).decode('ascii')}}
                self.backup()
                player.import_pack({'format': 'meme-player-library', 'version': 1, 'items': [item]})
                self.refresh_list()
            except Exception as error:
                messagebox.showerror('导入失败', str(error), parent=self.manager)

        def refresh_list(self):
            self.rows = player.assets()
            self.listbox.delete(0, 'end')
            for row in self.rows:
                self.listbox.insert('end', row['id'] + '  ' + row['label'])

        def manage(self):
            if self.manager and self.manager.winfo_exists():
                self.manager.lift()
                return
            self.manager = tk.Toplevel(window)
            self.manager.title('素材与回复')
            self.manager.geometry('460x300')
            self.manager.minsize(300, 220)
            ttk.Label(self.manager, text='选择表情，输入小猫的回复').pack(anchor='w', padx=8)
            frame = ttk.Frame(self.manager)
            frame.pack(fill='both', expand=True, padx=8)
            scrollbar = ttk.Scrollbar(frame)
            scrollbar.pack(side='right', fill='y')
            self.listbox = tk.Listbox(frame, yscrollcommand=scrollbar.set, exportselection=False)
            self.listbox.pack(fill='both', expand=True)
            scrollbar.config(command=self.listbox.yview)
            self.entry = ttk.Entry(self.manager)
            self.entry.insert(0, self.reply)
            self.entry.pack(fill='x', padx=8, pady=5)
            bar = ttk.Frame(self.manager)
            bar.pack(fill='x', padx=8, pady=5)
            def play_selected():
                selection = self.listbox.curselection()
                if selection:
                    self.request(self.rows[selection[0]]['id'], self.entry.get())
            ttk.Button(bar, text='播放回复', command=play_selected).pack(side='left')
            ttk.Button(bar, text='导入表情与声音', command=self.import_pair).pack(side='left', padx=4)
            ttk.Button(bar, text='关闭', command=self.manager.destroy).pack(side='right')
            self.refresh_list()

        def close(self):
            if self.closed:
                return
            self.closed = True
            self.stop()
            if self.clip:
                self.clip.close()
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            window.destroy()

    desktop = Desktop()
    try:
        if not desktop.closed:
            window.mainloop()
    except KeyboardInterrupt:
        desktop.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
