from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .service import ChatService


def make_handler(service: ChatService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # HTTP status only; don't put users' dialogue text in access logs.
            return

        def send_json(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json({"status": "ok"})
            else:
                self.send_json({"error": "接口不存在"}, 404)

        def do_POST(self):
            if self.path not in {"/chat", "/reset"}:
                self.send_json({"error": "接口不存在"}, 404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json({"error": "请使用 Content-Type: application/json"}, 415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    self.send_json({"error": "请求体必须为 1–65536 字节"}, 413)
                    return
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("请求体必须为 JSON 对象")
                session_id = body.get("session_id", "default")
                if not isinstance(session_id, str) or not session_id.strip() or len(session_id) > 128:
                    raise ValueError("session_id 必须为 1–128 字符的字符串")
                if self.path == "/chat":
                    text = body.get("text")
                    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
                        raise ValueError("text 必须为 1–16000 字符的非空字符串")
            except (ValueError, UnicodeDecodeError, TypeError) as error:
                self.send_json({"error": str(error)}, 400)
                return
            try:
                if self.path == "/reset":
                    service.sessions.reset(session_id)
                    self.send_json({"status": "reset"})
                else:
                    result = service.reply(text, session_id=session_id)
                    self.send_json({"reply": result["reply"], "meme_id": result["meme_id"]})
            except Exception:
                # Don't leak a traceback, model paths or partial model output.
                self.send_json({"error": "生成失败；该轮未写入会话，可以重试"}, 500)
    return Handler


def serve(engine, *, host="127.0.0.1", port=8790, idle_timeout=10.0):
    service = ChatService(engine, idle_timeout=idle_timeout, cleanup_interval=0.25)
    server = None
    try:
        server = ThreadingHTTPServer((host, port), make_handler(service))
        print(f"EmoStance meme API: http://{host}:{server.server_port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        service.close()
