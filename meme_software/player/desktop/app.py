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
from conversation_turn import ConversationTurn
from voice_recovery import VoiceRecovery, configured_capture_available
from audio_output import UsbAudioOutput, DEFAULT_USB_SINK

if not (Path(__file__).resolve().parent / 'meme_sing').is_dir():
    repository = Path(__file__).resolve().parents[3]
    if (repository / 'meme_sing').is_dir():
        sys.path.insert(0, str(repository))
from meme_sing.controller import MusicController
from meme_sing.playback import MusicPlayer
from meme_huanxing.controller import StandbyController


def speech_defaults():
    return {'asr_root': '', 'device': 'plughw:CARD=UACDemoV10,DEV=0',
            'host': '127.0.0.1', 'port': 9000,
            'token_file': str(Path.home() / 'zerotier_chat/.env')}


def main():
    parser = argparse.ArgumentParser(description='Meme猫原生桌面播放器（不打开浏览器）')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--windowed', action='store_true', help='以普通窗口启动；默认全屏')
    parser.add_argument('--mute', action='store_true', help='仅画面测试，不播放声音')
    parser.add_argument('--audio-sink', default=DEFAULT_USB_SINK,
                        help='固定使用的USB音箱Pulse输出名称，重连后自动恢复')
    parser.add_argument('--no-api', action='store_true', help='不启动本机回复接口')
    parser.add_argument('--port', type=int, default=8784)
    parser.add_argument('--check', action='store_true', help='检查依赖和数据，不打开窗口')
    parser.add_argument('--no-stt', action='store_true', help='只播放表情，不启动收音/服务器通信')
    parser.add_argument('--stt-root', type=Path, help='明确选择已有离线ASR安装目录')
    parser.add_argument('--meme-dir', type=Path, help='sucai_meme/meme_gif本地目录；按文件名前缀映射编号')
    parser.add_argument('--end-silence', type=float, default=.7, help='停顿多少秒提交整句，0.5到5，默认0.7')
    parser.add_argument('--music-dir', type=Path,
                        default=Path.home() / 'meme/meme-EmoStance/sucai_meme/meme_music',
                        help='本地点歌MP3目录；不上传、不下载音乐')
    parser.add_argument('--legacy-library', action='store_true', help='明确选择旧SQLite素材，仅用于回退对照')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在1到65535之间')
    if not math.isfinite(args.end_silence) or not .5 <= args.end_silence <= 5:
        parser.error('--end-silence需要在0.5到5秒之间')
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog, simpledialog
        from PIL import Image, ImageTk
        from native_core import Audio, check_library, read_asset, validate_wav
        sys.path.insert(0, str(args.root.resolve()))
        from launch import Player, Handler
        if args.legacy_library and args.meme_dir:
            raise ValueError('--legacy-library不能与--meme-dir同时使用')
        if args.legacy_library:
            count = check_library(args.root)
            player = Player(args.root)
            meme_directory = None
        else:
            from directory_library import DirectoryPlayer, find_directory
            meme_directory = find_directory(args.root, args.meme_dir)
            player = DirectoryPlayer(args.root, meme_directory)
            count = len(player.assets())
        if args.check:
            print(json.dumps({'assets': count, 'tk': tk.TkVersion, 'pillow': Image.__version__,
                              'aplay': bool(shutil.which('aplay')), 'api_port': args.port,
                              'browser_required': False, 'meme_directory': str(meme_directory) if meme_directory else None},
                             ensure_ascii=False))
            return 0
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
            self.current_visual_name = None
            self.pending_event = None
            self.render_generation = 0
            self.renderer = RenderEngine(lambda key, background_id=None:
                                         read_asset(player, key, background_id=background_id))
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
            self.audio = Audio(args.mute, sink_name=args.audio_sink)
            self.server = None
            self.manager = None
            self.link = ServerLink()
            self.speech = SpeechProcess()
            self.voice_recovery = VoiceRecovery()
            self.recovery_ticket = None
            self.voice_wanted = False
            self.waiting_audio_start = False
            self.audio_gate_held = False
            self.standby_connection_lost = False
            self.turn = ConversationTurn(self.link.submit, self.speech.pause, self.speech.resume,
                speech_start=self.link.speech_start, speech_cancel=self.link.speech_cancel,
                playback_done=self.link.playback_done)
            self.music = MusicController(self.turn, args.music_dir, self.speech.music_control,
                lambda: not self.speech.running() or self.speech.paused,
                player=MusicPlayer(volume=0 if args.mute else 30, sink_name=args.audio_sink))
            self.audio_output = UsbAudioOutput(args.audio_sink, get_owned_pids=self.audio_process_ids,
                can_repair=self.audio_repair_allowed)
            self.audio_output.start()
            self.link.on_ready = self.server_ready
            self.speech.on_music_stop = self.music.music_stop
            self.speech.on_music_utterance = self.music.music_utterance
            player.communication_status = self.communication_status
            player.music_command = self.music.command
            original_respond = player.respond
            player.respond = lambda body: self.turn.respond(body, original_respond, player.requests)
            self.ready_media = None
            self.visual_done = False
            self.speech_state = 'paused'
            self.net_state = 'closed'
            self.notice = '语音未开启'
            self.speech_error = ''
            self.stats = {}
            self.sent_at = 0.0
            self.quiet_intro = False
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
            self.standby = StandbyController(window, self, Path(__file__).resolve().parent,
                muted=args.mute, sink_name=args.audio_sink, schedule=False)
            self.speech.on_speech_start = self.standby.speech_started
            self.speech.on_speech_cancel = self.standby.speech_cancelled
            self.speech.on_final = self.standby.handle_final
            self.speech.on_standby_wake = self.standby.on_wake
            self.speech.on_standby_utterance = self.standby.on_utterance
            self.speech.on_standby_sleep = self.standby.on_sleep
            player.standby_command = self.standby.command
            self.fullscreen = True
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
            self.subtitle_shade = self.canvas.create_image(0, 0, anchor='nw', state='hidden')
            self.subtitle_item = self.canvas.create_text(0, 0, anchor='s', fill='white', justify='center')
            self.canvas.bind('<Configure>', self.resize)
            self.touch_controls_visible = False
            self.touch_hide_job = None
            self.canvas.bind('<ButtonPress-1>', self.touch_pressed)
            self.canvas.bind('<ButtonRelease-1>', self.touch_released)
            self.canvas.bind('<Configure>', lambda event: self.draw_exit_control(), add='+')
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

        def communication_status(self):
            with self.turn.lock:
                status = {'protocol_version': 2, 'phase': self.turn.phase,
                          'voice_enabled': self.turn.enabled, 'media_active': self.turn.media_active}
            with self.link.lock:
                status.update(server_connected=self.link.connected, model_ready=self.link.model_ready,
                              model_version=self.link.model_version, pending_requests=len(self.link.waiting),
                              playback_pending=self.link.awaiting_playback is not None,
                              pending_playback_receipts=len(self.link.playback_receipts))
            status.update(asr_running=self.speech.running(), microphone_paused=self.speech.paused,
                          end_silence_seconds=args.end_silence,
                          asr_state=self.speech_state, asr_error=self.speech_error[:300],
                          voice_recovery=self.voice_recovery.status())
            displayed = self.current_event or {}
            status.update(conversation_id=displayed.get('conversation_id'),
                          background_id=displayed.get('background_id', getattr(player, 'background_id', None)),
                          selected_visual=self.current_visual_name)
            status['music'] = self.music.status()
            status['audio_output'] = self.audio_output.status()
            status['standby'] = self.standby.status()
            return status

        def server_ready(self, ready):
            self.music.set_ready(ready)
            if not ready:
                self.standby_connection_lost = True

        def poll_standby(self):
            if self.standby_connection_lost:
                self.standby_connection_lost = False
                was_active = self.standby.active
                self.standby.abort()
                if (was_active and self.voice_wanted and not self.restarting and
                        self.turn.ready and self.link.connected and self.speech.running()
                        and self.voice_recovery.status()['state'] == 'idle'
                        and (args.mute or self.audio_output.ready())):
                    self.turn.enable()
            self.standby.poll()

        def audio_process_ids(self):
            pids = set(self.music.player.process_ids())
            process = self.audio.process
            if process is not None and process.poll() is None:
                pids.add(process.pid)
            standby = getattr(self, 'standby', None)
            screen = getattr(standby, 'screen', None)
            process = getattr(getattr(screen, 'audio', None), 'process', None)
            if process is not None and process.poll() is None:
                pids.add(process.pid)
            return pids

        def audio_repair_allowed(self):
            standby = getattr(self, 'standby', None)
            return ((not self.speech.running() or self.speech.paused)
                    and not self.music.active and not self.audio.is_playing()
                    and not (standby is not None and standby.active))

        def start_voice(self, recovery_ticket=None):
            if self.closed or self.restarting:
                return
            if recovery_ticket is None:
                self.voice_recovery.cancel()
                self.voice_wanted = True
            elif not self.voice_wanted:
                return
            self.recovery_ticket = recovery_ticket
            try:
                self.speech_error = ''
                root = find_asr_root(args.root, self.settings['asr_root'])
                if not self.link.thread or not self.link.thread.is_alive():
                    self.link.start(self.settings['host'], self.settings['port'], self.settings['token_file'])
                if not args.mute and not self.audio_output.ready():
                    self.waiting_audio_start = True
                    self.audio_output.request_recovery()
                    self.speech_state = 'recovering'
                    self.notice = '正在恢复USB音箱输出，准备好后自动开启收音。'
                    return
                self.waiting_audio_start = False
                self.speech.start(root, self.settings['device'], paused=True, end_silence=args.end_silence)
                self.turn.enable()
                self.speech_state = 'loading'
                self.notice = '正在加载本地语音识别，请稍候。'
            except (ValueError, OSError) as error:
                self.music.disable()
                self.voice_recovery.fail({'reason': 'asr_start_failed', 'recoverable': False,
                                          'message': str(error)})
                self.recovery_ticket = None
                self.waiting_audio_start = False
                self.speech_state = 'error'
                self.notice = str(error)
                self.speech_error = str(error)

        def draw_listening(self):
            active = (self.turn.phase == 'listening' and self.speech_state == 'listening' and self.speech.running() and
                      time.monotonic() - self.speech.last_activity < 1.5)
            if self.turn.phase in ('waiting', 'preparing', 'playing', 'cooldown', 'music_preparing', 'music', 'music_stopping'):
                label = {'waiting':'等待回复', 'preparing':'准备回复',
                         'playing':'正在回复', 'cooldown':'准备聆听',
                         'music_preparing':'正在选歌', 'music':'音乐播放中',
                         'music_stopping':'正在停止音乐'}[self.turn.phase]
                if self.turn.phase == 'preparing' and self.turn.media_active:
                    label = '回复待播放'
                color = '#8fc9ff'
            elif self.turn.phase in ('standby_preparing', 'standby', 'standby_waking'):
                label = {'standby_preparing': '准备待机', 'standby': '正在待机',
                         'standby_waking': '正在起床'}[self.turn.phase]
                color = '#8fc9ff'
            elif active:
                label = '正在听' if self.link.connected else '正在听 · 未连接'
                color = '#74e2b8' if self.link.connected else '#ffcc78'
                if self.link.connected and time.monotonic() - self.sent_at < 1:
                    label, color = '发送中', '#8fc9ff'
                if self.net_state in ('unsent', 'unconfirmed', 'server_error', 'auth_error', 'reply_error'):
                    label, color = '发送异常', '#ffcc78'
            else:
                label = {'loading': '正在准备', 'paused': '等待服务器就绪' if self.turn.phase == 'connecting' else '收音已停',
                         'stopping': '正在停止', 'error': '语音异常',
                         'recovering': '麦克风恢复中'}.get(self.speech_state, '收音等待')
                color = '#ffcc78' if self.speech_state == 'error' else '#a4b7b0'
            tag = 'listening-indicator'
            signature = (label, color, active)
            if not active and signature == self.indicator_layout:
                return
            self.indicator_layout = signature
            self.canvas.delete(tag)
            shade = self.canvas.create_rectangle(6, 6, 6, 39, fill='#10251f', outline='', tags=tag)
            for index in range(4):
                height = (6 + 11 * (1 + math.sin(time.monotonic() * 7 + index))) if active else 7
                x = 14 + index * 5
                self.canvas.create_line(x, 22 - height / 2, x, 22 + height / 2,
                                        fill=color, width=3, tags=tag)
            text_item = self.canvas.create_text(40, 22, text=label, anchor='w', fill=color,
                                    font=('sans-serif', 10, 'bold'), tags=tag)
            text_bounds = self.canvas.bbox(text_item)
            self.canvas.coords(shade, 6, 6, (text_bounds[2] if text_bounds else 40) + 8, 39)

        def animate_listening(self):
            if not self.closed:
                self.draw_listening()
                window.after(120, self.animate_listening)

        def poll_voice(self):
            for event in self.speech.drain():
                kind = event['type']
                if kind in ('loading', 'listening', 'paused'):
                    self.speech_state = kind
                    if kind == 'listening':
                        if self.recovery_ticket is not None:
                            self.voice_recovery.recovered(self.recovery_ticket)
                            self.recovery_ticket = None
                        self.speech_error = ''
                        self.notice = '正在听，说完后停顿约%s秒再发送。' % args.end_silence
                    elif kind == 'loading':
                        self.notice = '正在加载本地中文模型。'
                elif kind == 'utterance_cancelled':
                    self.notice = event.get('message', '本句话未完整结束，请重说。')
                elif kind == 'final':
                    if event.get('request_id'):
                        self.sent_at = time.monotonic()
                        self.notice = '识别完成，正在发送文字。'
                elif kind == 'error':
                    if not self.voice_wanted or self.closed:
                        continue  # A late device error cannot undo a manual pause.
                    self.standby.abort()
                    self.music.disable()
                    self.speech_state = 'error'
                    self.notice = event.get('message', '收音异常')
                    self.speech_error = self.notice
                    recover = (self.voice_recovery.fail(event)
                               if self.voice_wanted and not self.restarting and not self.closed else False)
                    if recover:
                        self.speech_state = 'recovering'
                        self.audio_output.request_recovery()
                        self.notice = '录音设备连接中断，已停止音乐；等待设备恢复后自动重新收音。'
                    elif self.voice_recovery.status()['state'] == 'exhausted':
                        self.notice = '麦克风多次恢复失败，请检查设备后在语音设置中重新开启。'
                    self.recovery_ticket = None
                    print(json.dumps({'event': 'asr_error', 'reason': event.get('reason', 'unknown'),
                                      'recoverable': event.get('recoverable') is True,
                                      'recovery': self.voice_recovery.status()}, ensure_ascii=False), flush=True)
                elif kind == 'stats':
                    self.stats = event.get('data', {})
                elif kind == 'exited' and self.speech_state not in ('error', 'recovering'):
                    self.standby.abort()
                    self.music.disable()
                    self.speech_state = 'paused'
                    self.notice = '收音已停止，麦克风已释放。'
            for event in self.link.drain():
                self.net_state = event['type']
                messages = {'connecting': '正在连接服务器。', 'connected': '服务器模型已就绪。',
                            'sent': '文字已发出，等待服务器确认。',
                            'accepted': '服务器已接收完整文字，正在推理。'}
                self.notice = event.get('message', messages.get(event['type'], self.notice))
                if event['type'] == 'reply':
                    try:
                        self.music.observe_reply(event)
                        # Only local paired media can be selected. Server text
                        # is never interpreted as a path, command or audio URL.
                        background_id = (player.begin_conversation(event['conversation_id'])
                                         if hasattr(player, 'begin_conversation') else None)
                        if event['meme_id'] not in {row['id'] for row in player.assets()}:
                            raise ValueError('服务器返回的表情编号不在当前素材来源中或已剔除：' + event['meme_id'])
                        player.respond({'meme_id': event['meme_id'], 'reply_text': event['reply_text'],
                                        'request_id': 'server-' + event['id'],
                                        'conversation_id': event['conversation_id'],
                                        'background_id': background_id})
                        self.notice = ('已收到新回复，等当前表情与声音完整播完再播放。' if self.turn.media_active
                                       else '已收到小猫回复，正在准备本地表情与配套声音。')
                    except (ValueError, sqlite3.Error) as error:
                        self.net_state = 'reply_error'
                        self.notice = str(error)
                        self.link.playback_done(event['id'], failed=True)
                        self.turn.failure()
                elif event['type'] in ('unsent', 'server_error', 'reply_error', 'server_restarted'):
                    self.music.forget_request(event.get('id'))
                    self.turn.failure()
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
            self.voice_wanted = False
            self.waiting_audio_start = False
            self.audio_gate_held = False
            self.voice_recovery.cancel()
            self.recovery_ticket = None
            self.next_settings = restart
            self.standby.abort()
            if self.restarting:
                return
            self.music.disable()
            self.restarting = True
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

        def poll_voice_recovery(self):
            if self.closed or not self.voice_wanted:
                return
            if self.waiting_audio_start:
                if args.mute or self.audio_output.ready():
                    ticket = self.recovery_ticket
                    self.waiting_audio_start = False
                    self.start_voice(recovery_ticket=ticket)
                return
            if self.voice_recovery.status()['state'] not in (
                    'backoff', 'waiting_device', 'device_stabilizing'):
                return
            ticket = self.voice_recovery.poll(
                worker_running=self.speech.running(), music_active=self.music.active,
                restarting=self.restarting or (not args.mute and not self.audio_output.ready()),
                device_available=configured_capture_available(self.settings['device']))
            if ticket is not None and self.voice_recovery.attempt_started(ticket):
                print(json.dumps({'event': 'asr_recovery_attempt',
                                  'recovery': self.voice_recovery.status()}, ensure_ascii=False), flush=True)
                self.start_voice(recovery_ticket=ticket)

        def poll_audio_output(self):
            if self.closed or args.mute or not self.voice_wanted or self.restarting:
                return
            if not self.audio_output.ready():
                if not self.audio_gate_held:
                    self.audio_gate_held = True
                    self.standby.abort()
                    self.music.disable()
                    self.stop()
                elif self.audio.is_playing():
                    self.stop()  # A pending reply can arrive during output recovery.
                if self.voice_recovery.status()['state'] not in ('error', 'exhausted'):
                    self.speech_state = 'recovering'
                    self.notice = 'USB音箱输出恢复中，准备好后自动继续对话。'
                return
            if self.audio_gate_held:
                self.audio_gate_held = False
                recovery_state = self.voice_recovery.status()['state']
                may_resume = recovery_state == 'idle' or (
                    recovery_state == 'recovering' and self.recovery_ticket is not None)
                if (self.speech.running() and not self.waiting_audio_start
                        and may_resume):
                    self.turn.enable()

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

        def draw_exit_control(self):
            tag = 'touch-exit-control'
            self.canvas.delete(tag)
            if not self.touch_controls_visible:
                return
            right = max(64, self.canvas.winfo_width()) - 8
            cx, cy = right - 28, 36
            # Open door and outward arrow, with a narrow contrast outline.
            # Only strokes are drawn: the expression remains visible behind it.
            paths = [
                (cx + 1, cy - 14, cx - 13, cy - 14, cx - 13, cy + 14, cx + 1, cy + 14),
                (cx - 2, cy, cx + 16, cy),
                (cx + 9, cy - 7, cx + 16, cy, cx + 9, cy + 7),
            ]
            for color, width in [('#16342d', 6), ('#f3fff9', 3)]:
                for points in paths:
                    self.canvas.create_line(*points, fill=color, width=width,
                                            capstyle='round', joinstyle='round', tags=tag)
            self.canvas.tag_raise(tag)

        def hide_touch_controls(self):
            self.touch_hide_job = None
            if self.closed:
                return
            self.touch_controls_visible = False
            self.draw_exit_control()
            self.set_fullscreen(True)

        def touch_pressed(self, event):
            # A first touch reveals the arrow; only a later touch can exit.
            right = max(64, self.canvas.winfo_width()) - 8
            if self.touch_controls_visible and right - 56 <= event.x <= right and 8 <= event.y <= 64:
                self.close()
                return 'break'
            if self.touch_hide_job is not None:
                window.after_cancel(self.touch_hide_job)
                self.touch_hide_job = None
            self.touch_controls_visible = True
            self.draw_exit_control()

        def touch_released(self, event):
            if self.closed:
                return
            if self.touch_hide_job is not None:
                window.after_cancel(self.touch_hide_job)
            self.touch_hide_job = window.after(3000, self.hide_touch_controls)

        def set_fullscreen(self, enabled):
            self.fullscreen = enabled
            window.attributes('-fullscreen', enabled)

        def request(self, key, text, background_id=None, conversation_id=None):
            try:
                body = {'meme_id': key, 'reply_text': text,
                        'request_id': 'desktop-' + uuid.uuid4().hex}
                if background_id is not None:
                    body['background_id'] = background_id
                if conversation_id is not None:
                    body['conversation_id'] = conversation_id
                player.respond(body)
            except Exception as error:
                messagebox.showerror('无法播放', str(error), parent=window)

        def replay(self):
            if self.key:
                snapshot = self.current_event or {}
                self.request(self.key, self.reply, snapshot.get('background_id'), snapshot.get('conversation_id'))

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
                    shade_top = max(top, box[1] - 8)
                    # One bounded RGBA overlay per subtitle layout; Tk blends it
                    # over each current frame without modifying the expression.
                    overlay = Image.new('RGBA', (max(1, math.ceil(right - left)),
                                                max(1, math.ceil(bottom - shade_top))),
                                        (115, 115, 115, 112))
                    self.subtitle_shade_photo = ImageTk.PhotoImage(overlay, master=window)
                    self.canvas.coords(self.subtitle_shade, left, shade_top)
                    self.canvas.itemconfigure(self.subtitle_shade,
                                              image=self.subtitle_shade_photo, state='normal')
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
                self.ready_media = None
                self.turn.media_cancelled()
            self.visual_done = False
            self.playing = False
            self.audio.stop()

        def play(self, event):
            self.stop(cancel=False)
            self.pending_event = dict(event)
            self.render_generation = self.renderer.request(event, self.render_size())

        def poll_render(self):
            events = self.renderer.take()
            if self.ready_media is not None:
                events.insert(0, self.ready_media)
                self.ready_media = None
            for item in events:
                if item['generation'] != self.render_generation:
                    continue
                kind = item['type']
                if kind == 'ready':
                    if not item['reflow'] and self.speech.running() and not self.speech.paused:
                        self.ready_media = item
                        continue  # Confirm ALSA has closed before starting speaker output.
                    if self.photo_profile != item['cache_key']:
                        self.photos.clear()
                        self.photo_bytes = 0
                        self.photo_profile = item['cache_key']
                    self.current_event = item['event']
                    self.current_visual_name = item.get('visual_name')
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
                        self.visual_done = False
                        self.turn.media_started()
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
                    self.visual_done = True
                    self.render_stats.update({k: item[k] for k in ('cache_hits', 'cache_misses', 'cache_bytes')})
                    mean = sum(self.draw_ms) / max(1, len(self.draw_ms))
                    self.render_summary = '画面提交 %s 帧 / 界面处理均值 %.1fms（非屏幕实测刷新率）' % (self.drawn_frames, mean)
                elif kind == 'render_error':
                    self.stop()
                    self.notice = item['message']
                    self.render_summary = item['message']
            if self.visual_done and not self.audio.is_playing():
                self.visual_done = False
                self.playing = False
                self.audio.stop()
                self.turn.media_finished()

        def poll(self):
            if self.closed:
                return
            self.poll_voice()
            for event in self.music.poll():
                self.notice = event['message']
                if event['type'] == 'started':
                    self.reply = event['message']
                    self.draw()
            self.poll_audio_output()
            self.poll_voice_recovery()
            self.poll_standby()
            with player.lock:
                event = player.event.copy()
            if event['revision'] != self.last_revision:
                if event['type'] == 'response' and self.turn.can_start_media():
                    self.last_revision = event['revision']
                    self.play(event)
                elif event['type'] == 'stop':
                    self.last_revision = event['revision']
                    self.stop()
            self.poll_render()
            self.turn.tick()
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
            self.audio_output.close()
            self.standby.close()
            self.music.close()
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
    from cursor_autohide import CursorAutoHide
    window.meme_cursor_autohide = CursorAutoHide(window)
    try:
        window.mainloop()
    except KeyboardInterrupt:
        desktop.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
