"""Exercise the public JSON API over a real loopback HTTP connection."""

from __future__ import annotations

from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest

from emostance_meme.http_server import make_handler
from emostance_meme.service import ChatService


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.failure: Exception | None = None

    def reply(self, messages: list[dict[str, str]]) -> dict[str, object]:
        self.calls.append([dict(message) for message in messages])
        if self.failure is not None:
            raise self.failure
        return {
            "reply": "回复：" + messages[-1]["content"],
            "meme_id": "meme-0062",
            "distance": 0.125,
        }


class HTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = FakeEngine()
        # A fixed monotonic clock keeps tests independent of execution speed.
        self.service = ChatService(self.engine, clock=lambda: 0.0)
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), make_handler(self.service)
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)
        self.service.close()

    def request(
        self,
        method: str,
        path: str,
        body: object = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, object]]:
        payload = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            self.assertEqual(response.getheader("Content-Type"), "application/json; charset=utf-8")
            data = response.read()
            self.assertEqual(int(response.getheader("Content-Length")), len(data))
            return response.status, json.loads(data)
        finally:
            connection.close()

    def test_chat_returns_exact_reply_and_meme_id(self) -> None:
        status, body = self.request("POST", "/chat", {"text": "给你一个么么哒"})
        self.assertEqual(status, 200)
        self.assertEqual(body, {"reply": "回复：给你一个么么哒", "meme_id": "meme-0062"})
        self.assertEqual(self.engine.calls, [[{"role": "user", "content": "给你一个么么哒"}]])

    def test_session_histories_are_isolated_and_preserve_turn_order(self) -> None:
        for session_id, text in (("a", "第一句"), ("b", "另一个用户"), ("a", "第二句")):
            status, _ = self.request("POST", "/chat", {"session_id": session_id, "text": text})
            self.assertEqual(status, 200)
        self.assertEqual(self.engine.calls[1], [{"role": "user", "content": "另一个用户"}])
        self.assertEqual(
            self.engine.calls[2],
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": "回复：第一句"},
                {"role": "user", "content": "第二句"},
            ],
        )

    def test_reset_clears_only_selected_session(self) -> None:
        self.request("POST", "/chat", {"session_id": "a", "text": "旧对话"})
        self.request("POST", "/chat", {"session_id": "b", "text": "另一个会话"})
        self.assertEqual(self.request("POST", "/reset", {"session_id": "a"}), (200, {"status": "reset"}))
        self.assertEqual(self.service.sessions.history("a"), [])
        self.assertEqual(len(self.service.sessions.history("b")), 2)
        self.request("POST", "/chat", {"session_id": "a", "text": "新对话"})
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "新对话"}])

    def test_default_session_can_be_reset(self) -> None:
        self.request("POST", "/chat", {"text": "默认会话"})
        self.assertEqual(self.request("POST", "/reset", {}), (200, {"status": "reset"}))
        self.assertEqual(self.service.sessions.history("default"), [])

    def test_health_does_not_invoke_engine(self) -> None:
        self.assertEqual(self.request("GET", "/health"), (200, {"status": "ok"}))
        self.assertEqual(self.engine.calls, [])

    def test_unknown_interfaces_return_404(self) -> None:
        for method, path in (("GET", "/chat"), ("GET", "/missing"), ("POST", "/health"), ("POST", "/missing")):
            with self.subTest(method=method, path=path):
                status, body = self.request(method, path, {})
                self.assertEqual(status, 404)
                self.assertIn("error", body)
        self.assertEqual(self.engine.calls, [])

    def test_invalid_text_returns_400_without_generation(self) -> None:
        for body in ({}, {"text": None}, {"text": 123}, {"text": []}, {"text": ""}, {"text": " \n "}, {"text": "字" * 16001}):
            with self.subTest(body_type=type(body.get("text")).__name__):
                status, response = self.request("POST", "/chat", body)
                self.assertEqual(status, 400)
                self.assertIn("error", response)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.service.sessions.active_session_count, 0)

    def test_invalid_session_id_returns_400_for_chat_and_reset(self) -> None:
        for path in ("/chat", "/reset"):
            for session_id in (None, 1, "", "   ", "a" * 129):
                with self.subTest(path=path, session_id_type=type(session_id).__name__):
                    status, body = self.request("POST", path, {"session_id": session_id, "text": "你好"})
                    self.assertEqual(status, 400)
                    self.assertIn("error", body)
        self.assertEqual(self.engine.calls, [])

    def test_invalid_content_type_returns_415(self) -> None:
        for content_type in ("text/plain", "application/x-www-form-urlencoded", "application/octet-stream"):
            with self.subTest(content_type=content_type):
                status, body = self.request("POST", "/chat", {"text": "你好"}, headers={"Content-Type": content_type})
                self.assertEqual(status, 415)
                self.assertIn("error", body)
        self.assertEqual(self.engine.calls, [])

    def test_json_content_type_accepts_charset_parameter(self) -> None:
        status, _ = self.request("POST", "/chat", {"text": "你好"}, headers={"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(status, 200)

    def test_invalid_json_and_non_object_body_return_400(self) -> None:
        for payload in (b"{broken", b"\xff", b"[]", b"null", b'"hello"', b"123"):
            with self.subTest(payload=payload):
                status, body = self.request("POST", "/chat", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", body)
        self.assertEqual(self.engine.calls, [])

    def test_empty_and_oversized_body_return_413(self) -> None:
        for payload in (b"", b" " * 65537):
            with self.subTest(length=len(payload)):
                status, body = self.request("POST", "/chat", payload)
                self.assertEqual(status, 413)
                self.assertIn("error", body)
        self.assertEqual(self.engine.calls, [])

    def test_non_numeric_content_length_returns_400(self) -> None:
        status, body = self.request("POST", "/chat", b"{}", headers={"Content-Length": "invalid"})
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_engine_failures_return_500_and_do_not_leave_partial_history(self) -> None:
        self.request("POST", "/chat", {"text": "已完成的问题"})
        original = self.service.sessions.history("default")
        for exception_type in (RuntimeError, ValueError, TypeError):
            with self.subTest(exception_type=exception_type.__name__):
                self.engine.failure = exception_type("private model details")
                status, body = self.request("POST", "/chat", {"text": "失败的问题"})
                self.assertEqual(status, 500)
                self.assertIn("error", body)
                self.assertNotIn("private model details", json.dumps(body))
                self.assertEqual(self.service.sessions.history("default"), original)
        self.engine.failure = None
        status, _ = self.request("POST", "/chat", {"text": "重试的问题"})
        self.assertEqual(status, 200)
        self.assertEqual(
            [message["content"] for message in self.engine.calls[-1]],
            ["已完成的问题", "回复：已完成的问题", "重试的问题"],
        )


if __name__ == "__main__":
    unittest.main()
