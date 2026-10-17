"""Local meme player. Python standard library only; Windows and Raspberry Pi OS."""
import argparse
import base64
import binascii
from contextlib import closing
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import sqlite3
import threading
import time
from urllib.parse import urlsplit
import webbrowser

FILE_LIMIT = 20 * 1024 * 1024
TOTAL_LIMIT = 256 * 1024 * 1024
BODY_LIMIT = 352 * 1024 * 1024
ASSET_LIMIT = 500
ASSET_ID = r'(?:user-[a-z0-9-]{1,64}|meme-\d{4})'
VISUAL_TYPES = {
    'image/png': 'image', 'image/jpeg': 'image', 'image/gif': 'gif',
    'image/webp': 'image', 'video/mp4': 'video', 'video/webm': 'video',
}
AUDIO_TYPES = {'audio/wav', 'audio/x-wav', 'audio/mpeg', 'audio/ogg',
               'audio/mp4', 'audio/aac', 'audio/webm'}


class PlayerError(ValueError):
    pass


def decode_media(value, audio=False):
    if not isinstance(value, dict):
        raise PlayerError('素材文件缺失')
    name, mime, data = (value.get(k) for k in ('name', 'type', 'data'))
    allowed = AUDIO_TYPES if audio else VISUAL_TYPES
    if (not isinstance(name, str) or not name or len(name) > 240 or
            not isinstance(mime, str) or mime not in allowed or
            not isinstance(data, str) or len(data) > ((FILE_LIMIT + 2) // 3) * 4):
        raise PlayerError('素材名称、格式或大小不正确')
    try:
        blob = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as error:
        raise PlayerError('素材文件编码不正确') from error
    if not blob or len(blob) > FILE_LIMIT:
        raise PlayerError('每个文件必须大于0且不超过20MB')
    return name, mime, blob


def decode_pack(pack):
    if (not isinstance(pack, dict) or pack.get('format') != 'meme-player-library' or
            pack.get('version') != 1 or not isinstance(pack.get('items'), list) or
            not 1 <= len(pack['items']) <= ASSET_LIMIT):
        raise PlayerError('这不是播放器导出的素材包')
    rows, ids, total = [], set(), 0
    for item in pack['items']:
        if not isinstance(item, dict):
            raise PlayerError('素材条目不正确')
        key, label = item.get('id'), item.get('label')
        duration, scene = item.get('duration_seconds'), item.get('scene_index')
        if (not isinstance(key, str) or not re.fullmatch(ASSET_ID, key) or
                key in ids or not isinstance(label, str) or not label.strip() or len(label) > 40 or
                type(duration) not in (int, float) or not math.isfinite(duration) or not .5 <= duration <= 120 or
                (scene is not None and (type(scene) is not int or not 0 <= scene <= 5))):
            raise PlayerError('素材编号、名称、时长或场景不正确')
        ids.add(key)
        vn, vt, vb = decode_media(item.get('visual'))
        if VISUAL_TYPES[vt] != item.get('visual_type'):
            raise PlayerError('素材类型不匹配')
        an, at, ab = decode_media(item['audio'], True) if item.get('audio') is not None else (None, None, None)
        total += len(vb) + len(ab or b'')
        if total > TOTAL_LIMIT:
            raise PlayerError('素材包超过256MB限制')
        rows.append((key, label.strip(), item['visual_type'], duration, scene,
                     vn, vt, vb, an, at, ab, time.time_ns()))
    return rows


class Player:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.website = self.root / 'website'
        self.base = json.loads((self.website / 'library.json').read_text(encoding='utf-8'))
        (self.root / 'data').mkdir(exist_ok=True)
        self.database = self.root / 'data' / 'assets.sqlite3'
        self.lock = threading.RLock()
        self.event = {'revision': 0, 'type': 'idle'}
        self.requests = {}
        with closing(self.connect()) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY, label TEXT, visual_type TEXT,
                duration_seconds REAL, scene_index INTEGER,
                visual_name TEXT, visual_mime TEXT, visual_blob BLOB,
                audio_name TEXT, audio_mime TEXT, audio_blob BLOB, updated_at INTEGER)''')

    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def assets(self):
        with closing(self.connect()) as db:
            rows = db.execute('''SELECT id,label,visual_type,duration_seconds,scene_index,
                visual_name,visual_mime,audio_name,audio_mime,updated_at FROM assets
                ORDER BY updated_at,id''').fetchall()
        return [dict(row) for row in rows]

    def import_pack(self, pack):
        rows = decode_pack(pack)
        self.save_rows(rows)
        return len(rows)

    def save_rows(self, rows):
        """Atomically store validated rows; accepts a stream for local ZIP migration."""
        with self.lock, closing(self.connect()) as db, db:
            db.executemany('INSERT OR REPLACE INTO assets VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', rows)
            count, size = db.execute('''SELECT count(*),coalesce(sum(length(visual_blob)+
                coalesce(length(audio_blob),0)),0) FROM assets''').fetchone()
            if count > ASSET_LIMIT or size > TOTAL_LIMIT:
                raise PlayerError('素材库最多500套、总文件不超过256MB')

    def export_pack(self):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT * FROM assets ORDER BY updated_at,id').fetchall()
        items = []
        for row in rows:
            item = {k: row[k] for k in ('id', 'label', 'visual_type', 'duration_seconds', 'scene_index')}
            for prefix in ('visual', 'audio'):
                blob = row[prefix + '_blob']
                item[prefix] = None if blob is None else {
                    'name': row[prefix + '_name'], 'type': row[prefix + '_mime'],
                    'data': base64.b64encode(blob).decode('ascii')}
            items.append(item)
        return {'format': 'meme-player-library', 'version': 1, 'items': items}

    def library(self):
        library = json.loads(json.dumps(self.base))
        positions = {item['id']: index for index, item in enumerate(library['items'])}
        for row in self.assets():
            path = '/user-media/' + row['id']
            item = {
                'id': row['id'], 'label': row['label'], 'visual_type': row['visual_type'],
                'visual': path + '/visual',
                'poster': None if row['visual_type'] == 'video' else path + '/visual',
                'audio': path + '/audio' if row['audio_mime'] else None,
                'duration_seconds': row['duration_seconds'], 'imported': True}
            if row['id'] in positions:
                library['items'][positions[row['id']]] = item
            else:
                positions[row['id']] = len(library['items'])
                library['items'].append(item)
            if row['scene_index'] is not None:
                library['scenarios'][row['scene_index']]['meme_id'] = row['id']
        return library

    def media(self, key, channel):
        if channel not in ('visual', 'audio'):
            return None
        with closing(self.connect()) as db:
            return db.execute('SELECT ' + channel + '_mime,' + channel +
                              '_blob FROM assets WHERE id=?', (key,)).fetchone()

    def respond(self, body):
        if not isinstance(body, dict):
            raise PlayerError('结果需要是JSON对象')
        key, text = body.get('meme_id'), body.get('reply_text')
        if not isinstance(key, str) or key not in {r['id'] for r in self.library()['items']}:
            raise PlayerError('meme_id 不在素材库中')
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise PlayerError('reply_text 需要是1到2000字的文本')
        request_id = body.get('request_id', 'local-' + str(time.time_ns()))
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request_id):
            raise PlayerError('request_id格式不正确')
        with self.lock:
            if request_id in self.requests:
                previous = self.requests[request_id]
                if previous['meme_id'] != key or previous['reply_text'] != text.strip():
                    raise PlayerError('同一request_id不能对应不同回复')
                return previous.copy()
            self.event = {'revision': self.event['revision'] + 1, 'type': 'response',
                          'request_id': request_id, 'meme_id': key, 'reply_text': text.strip()}
            self.requests[request_id] = self.event.copy()
            if len(self.requests) > 200:
                del self.requests[next(iter(self.requests))]
            return self.event.copy()

    def stop(self):
        with self.lock:
            self.event = {'revision': self.event['revision'] + 1, 'type': 'stop'}
            return self.event.copy()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, player, **kwargs):
        self.player = player
        super().__init__(*args, directory=str(player.website), **kwargs)

    def local_request(self):
        port = self.server.server_address[1]
        return self.headers.get('Host') in (f'127.0.0.1:{port}', f'localhost:{port}')

    def is_frontend(self):
        path = urlsplit(self.path).path
        return path in ('/', '/index.html') or path.endswith(('.css', '.js'))

    def end_headers(self):
        if self.is_frontend():
            self.send_header('Cache-Control', 'no-store')
        super().end_headers()

    def json_response(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode('utf-8'),
                        'application/json; charset=utf-8', status)

    def send_bytes(self, data, mime, status=200, media=False):
        total = len(data)
        start, end = 0, total - 1
        if media and self.headers.get('Range'):
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers['Range'])
            if match and any(match.groups()):
                first, last = match.groups()
                start = int(first) if first else max(0, total - int(last))
                end = min(int(last), total - 1) if first and last else total - 1
                if start <= end and start < total:
                    status = 206
                else:
                    match = None
            else:
                match = None
            if match is None:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{total}')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(end - start + 1))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if media:
            self.send_header('Accept-Ranges', 'bytes')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{total}')
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data[start:end + 1])

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self):
        if not self.local_request():
            self.json_response({'error': '只接受本机访问'}, 403)
            return
        path = urlsplit(self.path).path
        try:
            if path == '/api/library':
                self.json_response(self.player.library())
            elif path == '/api/state':
                with self.player.lock:
                    state = self.player.event.copy()
                self.json_response(state)
            elif path == '/api/assets':
                self.json_response(self.player.assets())
            elif path == '/api/assets/export':
                self.json_response(self.player.export_pack())
            elif path == '/api/health':
                self.json_response({'status': 'ready', 'mode': 'local-demo', 'storage': 'data/assets.sqlite3'})
            elif path.startswith('/user-media/'):
                match = re.fullmatch(r'/user-media/(' + ASSET_ID + r')/(visual|audio)', path)
                result = self.player.media(*match.groups()) if match else None
                if result is None or result[1] is None:
                    self.json_response({'error': '素材不存在'}, 404)
                else:
                    self.send_bytes(result[1], result[0], media=True)
            elif path.startswith('/api/'):
                self.json_response({'error': '接口不存在'}, 404)
            else:
                if self.is_frontend():
                    for header in ('If-Modified-Since', 'If-None-Match'):
                        if header in self.headers:
                            del self.headers[header]
                super().do_GET()
        except (OSError, sqlite3.Error):
            self.json_response({'error': '本地素材读取失败，请检查data目录和磁盘空间'}, 500)

    def do_POST(self):
        if not self.local_request():
            self.json_response({'error': '只接受本机访问'}, 403)
            return
        expected_origin = 'http://' + self.headers['Host']
        if self.headers.get('Origin') not in (None, expected_origin):
            self.json_response({'error': '请求来源不正确'}, 403)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= BODY_LIMIT:
                raise PlayerError('请求为空或超过素材包大小限制')
            if self.headers.get_content_type() != 'application/json':
                raise PlayerError('请使用JSON格式')
            body = json.loads(self.rfile.read(length).decode('utf-8'))
            path = urlsplit(self.path).path
            if path == '/api/assets/import':
                self.json_response({'count': self.player.import_pack(body)})
            elif path == '/api/response':
                self.json_response(self.player.respond(body))
            elif path == '/api/stop':
                self.json_response(self.player.stop())
            else:
                self.json_response({'error': '接口不存在'}, 404)
        except (PlayerError, ValueError, UnicodeError, RecursionError) as error:
            self.json_response({'error': str(error) if isinstance(error, PlayerError) else 'JSON数据不正确'}, 400)
        except (OSError, sqlite3.Error):
            self.json_response({'error': '本地保存失败，请检查data目录和磁盘空间'}, 500)


def main():
    parser = argparse.ArgumentParser(description='猫咪小剧场：本地网站')
    parser.add_argument('--port', type=int, default=8783)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--screen', action='store_true', help='Open the compact touch-screen display')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('端口必须在1到65535之间')
    root = Path(__file__).resolve().parent
    try:
        player = Player(root)
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, player=player))
    except (OSError, sqlite3.Error, ValueError) as error:
        print('启动失败：' + str(error), flush=True)
        print('请检查程序是否完整、目录能否写入，或用 --port 8784 更换端口。', flush=True)
        return 1
    url = f'http://127.0.0.1:{args.port}/' + ('?screen=1' if args.screen else '')
    print('猫咪小剧场本地播放器：' + url, flush=True)
    print('素材保存在：' + str(player.database), flush=True)
    print('当前为演示模式。按 Ctrl+C 停止服务。', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
