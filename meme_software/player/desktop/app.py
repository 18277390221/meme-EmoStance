#!/usr/bin/env python3
"""Meme猫原生桌面程序：Tk窗口，非浏览器/WebView。"""
import argparse
import base64
from collections import OrderedDict
from contextlib import closing
from functools import partial
from http.server import ThreadingHTTPServer
from io import BytesIO
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import threading
import time
import uuid

from stt_bridge import SpeechProcess, find_asr_root
from server_link import ServerLink
from render_pipeline import RenderEngine


def speech_defaults():
    return {'asr_root': '', 'device': 'plughw:CARD=UACDemoV10,DEV=0',
            'host': '127.0.0.1', 'port': 9000,
            'token_file': str(Path.home() / 'zerotier_chat/.env')}


def main():
    parser = argparse.ArgumentParser(description='Meme猫原生桌面播放器（不打开浏览器）')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--windowed', action='store_true', help='以普通窗口启动；默认全屏')
    parser.add_argument('--mute', action='store_true', help='仅画面测试，不播放声音')
    parser.add_argument('--no-api', action='store_true', help='不启动本机回复接口')
    parser.add_argument('--port', type=int, default=8784)
    parser.add_argument('--check', action='store_true', help='检查依赖和数据，不打开窗口')
    parser.add_argument('--no-stt', action='store_true', help='只播放表情，不启动收音/服务器通信')
    parser.add_argument('--stt-root', type=Path, help='明确选择已有离线ASR安装目录')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在1到65535之间')
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog, simpledialog
        from PIL import Image, ImageTk
        from native_core import Audio, check_library, read_asset, validate_wav
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
            self.frame = None
            self.photo = None
            self.photos = OrderedDict()
            self.photo_bytes = 0
            self.photo_profile = None
            self.frame_key = None
            self.subtitle_layout = None
            self.picture_layout = None
            self.indicator_layout = None
            self.current_event = None
            self.pending_event = None
            self.render_generation = 0
            self.renderer = RenderEngine(lambda key: read_asset(player, key))
            self.render_stats = {}
            self.render_summary = ''
            self.drawn_frames = 0
            self.draw_ms = []
            self.playing = False
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
            self.link = ServerLink()
            self.speech = SpeechProcess(on_final=self.link.submit)
            self.speech_state = 'paused'
            self.net_state = 'closed'
            self.notice = '语音未开启'
            self.speech_error = ''
            self.stats = {}
            self.sent_at = 0.0
            self.quiet_intro = not args.no_stt and sys.platform == 'linux'
            self.speech_pane = None
            self.settings = speech_defaults()
            self.config_path = args.root / 'data/speech-config.json'
            try:
                if self.config_path.is_file():
                    saved = json.loads(self.config_path.read_text(encoding='utf-8'))
                    self.settings.update({k: saved[k] for k in self.settings if k in saved})
            except (ValueError, OSError, TypeError):
                self.notice = '语音配置读取失败，请打开语音设置重新保存。'
            if args.stt_root:
                self.settings['asr_root'] = str(args.stt_root)
            self.io_done = threading.Event()
            self.restarting = False
            self.next_settings = None
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
            self.picture_item = self.canvas.create_image(0, 0, anchor='center')
            self.subtitle_shade = self.canvas.create_rectangle(0, 0, 0, 0, fill='#10251f', outline='', state='hidden')
            self.subtitle_item = self.canvas.create_text(0, 0, anchor='s', fill='white', justify='center')
            self.canvas.bind('<Configure>', self.resize)
            bar = tk.Frame(window, bg='#102e26')
            bar.pack(fill='x')
            for text, action in [('表情', self.scenes), ('重播', self.replay),
                                 ('管理', self.manage), ('语音', self.voice_settings),
                                 ('全屏', lambda: self.set_fullscreen(not self.fullscreen)),
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
            window.after(120, self.animate_listening)
            if not args.no_stt and sys.platform == 'linux':
                window.after(200, self.start_voice)

        def start_voice(self):
            if self.closed or self.restarting:
                return
            try:
                self.speech_error = ''
                root = find_asr_root(args.root, self.settings['asr_root'])
                if not self.link.thread or not self.link.thread.is_alive():
                    self.link.start(self.settings['host'], self.settings['port'], self.settings['token_file'])
                self.speech.start(root, self.settings['device'])
                self.speech_state = 'loading'
                self.notice = '正在加载本地中文识别，请稍候。'
            except (ValueError, OSError) as error:
                self.speech_state = 'error'
                self.notice = str(error)
                self.speech_error = str(error)

        def draw_listening(self):
            active = (self.speech_state == 'listening' and self.speech.running() and
                      time.monotonic() - self.speech.last_activity < 1.5)
            if active:
                label = '正在听' if self.link.connected else '正在听 · 未连接'
                color = '#74e2b8' if self.link.connected else '#ffcc78'
                if self.link.connected and time.monotonic() - self.sent_at < 1:
                    label, color = '发送中', '#8fc9ff'
                if self.net_state in ('unsent', 'unconfirmed', 'server_error', 'auth_error', 'reply_error'):
                    label, color = '发送异常', '#ffcc78'
            else:
                label = {'loading': '正在准备', 'paused': '收音已停',
                         'stopping': '正在停止', 'error': '语音异常'}.get(self.speech_state, '收音等待')
                color = '#ffcc78' if self.speech_state == 'error' else '#a4b7b0'
            tag = 'listening-indicator'
            signature = (label, color, active)
            if not active and signature == self.indicator_layout:
                return
            self.indicator_layout = signature
            self.canvas.delete(tag)
            self.canvas.create_rectangle(6, 6, 176, 39, fill='#10251f', outline='', tags=tag)
            for index in range(4):
                height = (6 + 11 * (1 + math.sin(time.monotonic() * 7 + index))) if active else 7
                x = 14 + index * 5
                self.canvas.create_line(x, 22 - height / 2, x, 22 + height / 2,
                                        fill=color, width=3, tags=tag)
            self.canvas.create_text(40, 22, text=label, anchor='w', fill=color,
                                    font=('sans-serif', 10, 'bold'), tags=tag)

        def animate_listening(self):
            if not self.closed:
                self.draw_listening()
                window.after(120, self.animate_listening)

        def poll_voice(self):
            for event in self.speech.drain():
                kind = event['type']
                if kind in ('loading', 'listening'):
                    self.speech_state = kind
                    self.notice = '正在收音，完整句子将发给服务器。' if kind == 'listening' else '正在加载本地中文模型。'
                elif kind == 'final':
                    if event.get('request_id'):
                        self.sent_at = time.monotonic()
                        self.notice = '识别完成，正在发送文字。'
                elif kind == 'error':
                    self.speech_state = 'error'
                    self.notice = event.get('message', '收音异常')
                    self.speech_error = self.notice
                elif kind == 'stats':
                    self.stats = event.get('data', {})
                elif kind == 'exited' and self.speech_state != 'error':
                    self.speech_state = 'paused'
                    self.notice = '收音已停止，麦克风已释放。'
            for event in self.link.drain():
                self.net_state = event['type']
                messages = {'connecting': '正在连接服务器。', 'connected': '服务器已连接，正在收音。',
                            'sent': '文字已发出，等待服务器确认。',
                            'ack': '服务器已确认收到文字，尚未返回表情编号和回复。'}
                self.notice = event.get('message', messages.get(event['type'], self.notice))
                if event['type'] == 'reply':
                    try:
                        # Only local paired media can be selected. Server text
                        # is never interpreted as a path, command or audio URL.
                        if event['meme_id'] not in {row['id'] for row in player.assets()}:
                            raise ValueError('服务器返回的表情编号未导入本机：' + event['meme_id'])
                        player.respond({'meme_id': event['meme_id'], 'reply_text': event['reply_text'],
                                        'request_id': 'server-' + event['id']})
                        self.notice = '已收到小猫回复，正在播放本地表情与配套声音。'
                    except (ValueError, sqlite3.Error) as error:
                        self.net_state = 'reply_error'
                        self.notice = str(error)
            if self.restarting and self.io_done.is_set():
                self.restarting = False
                if self.next_settings:
                    self.settings = self.next_settings
                    self.next_settings = None
                    self.start_voice()
                else:
                    self.speech_state = 'paused'
            if self.speech_pane and self.speech_pane.winfo_exists():
                self.voice_notice.set(self.speech_error or self.notice)
                capture = self.stats.get('capture', {})
                self.voice_stats.set('输入样本 %s / 已处理 %s / 溢出 %s' % (
                    capture.get('input_samples', '—'), capture.get('delivered_samples', '—'),
                    capture.get('overflow_events', '—')) + ('\n' + self.render_summary if self.render_summary else ''))

        def stop_voice(self, restart=None):
            if self.restarting:
                return
            self.restarting = True
            self.next_settings = restart
            self.speech_state = 'stopping'
            self.notice = '正在停止收音并释放麦克风。'
            self.io_done.clear()
            def stop():
                try:
                    self.speech.close()
                    self.link.close()
                finally:
                    self.io_done.set()
            threading.Thread(target=stop, daemon=True).start()

        def voice_settings(self):
            if self.speech_pane and self.speech_pane.winfo_exists():
                self.speech_pane.lift()
                return
            pane = self.speech_pane = tk.Toplevel(window)
            pane.title('语音与服务器')
            pane.geometry('%dx%d' % (min(460, window.winfo_screenwidth()),
                                    max(240, min(310, window.winfo_screenheight() - 50))))
            pane.minsize(300, 240)
            frame = ttk.Frame(pane, padding=8)
            frame.pack(fill='both', expand=True)
            frame.columnconfigure(1, weight=1)
            fields = {}
            for row, (name, label) in enumerate((('asr_root', 'STT目录'),
                    ('device', '麦克风'), ('host', '服务器地址'), ('port', '端口'), ('token_file', '现有口令文件'))):
                ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=3)
                fields[name] = ttk.Entry(frame)
                fields[name].insert(0, str(self.settings[name]))
                fields[name].grid(row=row, column=1, sticky='ew', padx=(5, 0))
            self.voice_notice = tk.StringVar(value=self.notice)
            self.voice_stats = tk.StringVar(value='')
            ttk.Label(frame, textvariable=self.voice_notice,
                      wraplength=max(240, min(400, window.winfo_screenwidth() - 30))).grid(row=5, columnspan=2, sticky='w', pady=7)
            ttk.Label(frame, textvariable=self.voice_stats).grid(row=6, columnspan=2, sticky='w')
            bar = ttk.Frame(frame)
            bar.grid(row=7, columnspan=2, sticky='ew', pady=8)
            def apply():
                if self.restarting:
                    return
                try:
                    config = {name: entry.get().strip() for name, entry in fields.items()}
                    config['port'] = int(config['port'])
                    if not 1 <= config['port'] <= 65535 or not config['host'] or not config['device']:
                        raise ValueError('请填写有效的服务器地址、端口和麦克风。')
                    find_asr_root(args.root, config['asr_root'])
                    self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8', newline='\n')
                    self.stop_voice(restart=config)
                except (ValueError, OSError) as error:
                    messagebox.showerror('语音设置', str(error), parent=pane)
            ttk.Button(bar, text='保存并开始', command=apply).pack(side='left')
            ttk.Button(bar, text='暂停收音', command=self.stop_voice).pack(side='left', padx=5)
            ttk.Button(bar, text='关闭', command=pane.destroy).pack(side='right')

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
            event = self.pending_event or self.current_event
            if event:
                self.render_generation = self.renderer.request(event, self.render_size(),
                    reflow=self.pending_event is None)
            self.draw()

        def render_size(self):
            return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

        def draw(self):
            before = time.monotonic()
            width, height = self.render_size()
            left, top, right, bottom = 0, 0, width, height
            if self.frame is not None:
                frame = self.frame
                key = self.frame_key
                if key in self.photos:
                    self.photo = self.photos.pop(key)
                    self.photos[key] = self.photo
                else:
                    self.photo = ImageTk.PhotoImage(frame, master=window)
                    cost = frame.width * frame.height * 4
                    while self.photos and self.photo_bytes + cost > 32 * 1024 * 1024:
                        _, old = self.photos.popitem(last=False)
                        self.photo_bytes -= old.width() * old.height() * 4
                    if cost <= 32 * 1024 * 1024:
                        self.photos[key] = self.photo
                        self.photo_bytes += cost
                picture = (str(self.photo), width, height)
                if picture != self.picture_layout:
                    self.picture_layout = picture
                    self.canvas.itemconfigure(self.picture_item, image=self.photo)
                    self.canvas.coords(self.picture_item, width / 2, height / 2)
                left, top = (width - frame.width) / 2, (height - frame.height) / 2
                right, bottom = left + frame.width, top + frame.height
            # Reply is inside the bottom of the visual, with no media name/timer.
            layout = (self.reply, width, height, left, top, right, bottom)
            if layout != self.subtitle_layout:
                self.subtitle_layout = layout
                font = ('sans-serif', max(12, min(22, int(width / 33))), 'bold')
                self.canvas.coords(self.subtitle_item, (left + right) / 2, bottom - 8)
                self.canvas.itemconfigure(self.subtitle_item, text=self.reply,
                                          width=max(50, right - left - 20), font=font)
                box = self.canvas.bbox(self.subtitle_item)
                if box:
                    self.canvas.coords(self.subtitle_shade, left, max(top, box[1] - 8), right, bottom)
                    self.canvas.itemconfigure(self.subtitle_shade, state='normal')
            self.drawn_frames += 1
            self.draw_ms.append((time.monotonic() - before) * 1000)
            self.draw_ms = self.draw_ms[-512:]

        def stop(self, cancel=True):
            if self.animation:
                window.after_cancel(self.animation)
                self.animation = None
            if cancel:
                self.renderer.cancel()
                self.render_generation = self.renderer.generation
                self.pending_event = None
            self.playing = False
            self.audio.stop()

        def play(self, event):
            self.stop(cancel=False)
            self.pending_event = dict(event)
            self.render_generation = self.renderer.request(event, self.render_size())

        def poll_render(self):
            for item in self.renderer.take():
                if item['generation'] != self.render_generation:
                    continue
                kind = item['type']
                if kind == 'ready':
                    if self.photo_profile != item['cache_key']:
                        self.photos.clear()
                        self.photo_bytes = 0
                        self.photo_profile = item['cache_key']
                    self.current_event = item['event']
                    self.key, self.reply = self.current_event['meme_id'], self.current_event['reply_text']
                    self.pending_event = None
                    self.frame, self.frame_key = item['image'], (item['cache_key'], item['index'])
                    self.draw()
                    if not item['reflow']:
                        self.drawn_frames = 1
                        self.draw_ms = self.draw_ms[-1:]
                        self.started = time.monotonic()
                        self.deadline = self.started + item['duration']
                        self.playing = True
                        if self.quiet_intro:
                            self.quiet_intro = False
                        else:
                            try:
                                self.audio.play(item['audio'])
                            except (ValueError, OSError) as error:
                                self.notice = str(error)
                        self.render_stats = {'prepare_ms': item['prepare_ms']}
                    self.renderer.begin(self.render_generation, self.started if self.playing else time.monotonic() - item['duration'])
                elif kind == 'frame':
                    if self.playing and time.monotonic() < self.deadline:
                        self.frame, self.frame_key = item['image'], (item['cache_key'], item['index'])
                        self.draw()
                elif kind == 'finished':
                    self.playing = False
                    self.audio.stop()
                    self.render_stats.update({k: item[k] for k in ('cache_hits', 'cache_misses', 'cache_bytes')})
                    mean = sum(self.draw_ms) / max(1, len(self.draw_ms))
                    self.render_summary = '画面提交 %s 帧 / 界面处理均值 %.1fms（非屏幕实测刷新率）' % (self.drawn_frames, mean)
                elif kind == 'render_error':
                    self.stop()
                    self.notice = item['message']
                    self.render_summary = item['message']

        def poll(self):
            if self.closed:
                return
            self.poll_voice()
            with player.lock:
                event = player.event.copy()
            if event['revision'] != self.last_revision:
                self.last_revision = event['revision']
                if event['type'] == 'response':
                    self.play(event)
                elif event['type'] == 'stop':
                    self.stop()
            self.poll_render()
            for error in self.audio.pop_errors():
                messagebox.showwarning('声音未能播放', error, parent=window)
            window.after(16, self.poll)

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
            self.renderer.close()
            if self.server:
                self.server.shutdown()
                self.server.server_close()
            self.stop_voice()
            def finish_close():
                if self.io_done.is_set():
                    window.destroy()
                else:
                    window.after(50, finish_close)
            window.after(50, finish_close)

    desktop = Desktop()
    try:
        window.mainloop()
    except KeyboardInterrupt:
        desktop.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
