"""Behavioral checks for deadline, queuing, reset and atomic history commits."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

from emostance_meme.service import ChatService
from emostance_meme.session import SessionManager


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self) -> float:
        with self._lock:
            self.calls += 1
            return self.now

    def set(self, value: float) -> None:
        with self._lock:
            self.now = value


class RecordingEngine:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def reply(self, messages: list[dict[str, str]]) -> dict[str, object]:
        self.calls.append([dict(message) for message in messages])
        return {
            "reply": "回答：" + messages[-1]["content"],
            "meme_id": "meme-0008",
            "distance": 0.25,
        }


class SessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.engine = RecordingEngine()
        self.service = ChatService(self.engine, clock=self.clock)
        self.addCleanup(self.service.close)

    def test_multiple_turns_and_separate_sessions(self) -> None:
        first = self.service.reply("你好", "a")
        self.clock.set(2)
        self.service.reply("你还记得我吗", "a")
        self.service.reply("另一位用户", "b")
        self.assertEqual(first, {"reply": "回答：你好", "meme_id": "meme-0008", "distance": 0.25})
        self.assertEqual(
            self.engine.calls[1],
            [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "回答：你好"},
                {"role": "user", "content": "你还记得我吗"},
            ],
        )
        self.assertEqual(self.engine.calls[2], [{"role": "user", "content": "另一位用户"}])

    def test_9_999_seconds_keeps_history(self) -> None:
        self.service.reply("第一句")
        self.clock.set(9.999)
        self.service.reply("第二句")
        self.assertEqual(len(self.engine.calls[-1]), 3)

    def test_exactly_10_seconds_resets_history(self) -> None:
        self.service.reply("第一句")
        self.clock.set(10)
        self.service.reply("新的对话")
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "新的对话"}])

    def test_idle_clock_starts_at_user_arrival_not_assistant_completion(self) -> None:
        def slow_reply(messages: list[dict[str, str]]) -> dict[str, str]:
            self.clock.set(8)
            return {"reply": "迟一些的回答", "meme_id": "meme-0002"}

        self.service.sessions.respond("default", "第一句", slow_reply)
        self.clock.set(10)
        self.service.reply("第二句")
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "第二句"}])

    def test_generation_past_deadline_returns_result_without_restoring_history(self) -> None:
        def slow_reply(messages: list[dict[str, str]]) -> dict[str, str]:
            self.clock.set(11)
            self.assertEqual(messages, [{"role": "user", "content": "第一句"}])
            return {"reply": "慢回答", "meme_id": "meme-0002"}

        result = self.service.sessions.respond("default", "第一句", slow_reply)
        self.assertEqual(result["reply"], "慢回答")
        self.assertEqual(self.service.sessions.history("default"), [])

    def test_failed_generation_leaves_no_partial_history_and_allows_retry(self) -> None:
        self.service.reply("已完成")

        def failing_reply(messages: list[dict[str, str]]) -> dict[str, str]:
            raise RuntimeError("generation failed")

        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.service.sessions.respond("default", "失败的输入", failing_reply)
        self.service.reply("重试")
        self.assertEqual(
            [m["content"] for m in self.engine.calls[-1]],
            ["已完成", "回答：已完成", "重试"],
        )

    def test_invalid_engine_result_is_not_committed(self) -> None:
        for bad in ({"reply": "缺少表情"}, {"reply": "", "meme_id": "meme-1"}, {"reply": "ok", "meme_id": "meme-1", "distance": float("nan")}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.service.sessions.respond("default", "问题", lambda _: bad)
        self.assertEqual(self.service.sessions.history("default"), [])

    def test_history_copies_protect_committed_state(self) -> None:
        self.service.reply("第一句")
        history = self.service.sessions.history("default")
        history[0]["content"] = "篡改"
        self.service.reply("第二句")
        self.assertEqual(self.engine.calls[-1][0]["content"], "第一句")

    def test_explicit_reset(self) -> None:
        self.service.reply("旧对话")
        self.assertTrue(self.service.reset())
        self.assertFalse(self.service.reset())
        self.service.reply("新对话")
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "新对话"}])

    def test_idle_cleanup_releases_only_idle_sessions(self) -> None:
        self.service.reply("旧", "old")
        self.clock.set(5)
        self.service.reply("新", "new")
        self.clock.set(10)
        self.assertEqual(self.service.sessions.expire_idle(), 1)
        self.assertEqual(self.service.sessions.active_session_count, 1)
        self.assertEqual(self.service.sessions.history("old"), [])
        self.assertEqual(len(self.service.sessions.history("new")), 2)

    def test_background_cleanup_expires_without_new_request(self) -> None:
        manager = SessionManager(clock=self.clock, cleanup_interval=0.005)
        self.addCleanup(manager.close)
        manager.respond("a", "问题", self.engine.reply)
        self.clock.set(10)
        deadline = time.monotonic() + 1
        while manager.active_session_count and time.monotonic() < deadline:
            threading.Event().wait(0.005)
        self.assertEqual(manager.active_session_count, 0)

    def test_same_session_fifo_uses_arrival_time_despite_slow_generation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        second_arrived = threading.Event()
        snapshots: list[list[dict[str, str]]] = []

        def clock() -> float:
            received_at = self.clock()
            if threading.current_thread().name.endswith("_1"):
                second_arrived.set()
            return received_at

        manager = SessionManager(clock=clock)
        self.addCleanup(manager.close)

        def generate(messages: list[dict[str, str]]) -> dict[str, str]:
            snapshots.append([dict(message) for message in messages])
            if messages[-1]["content"] == "第一句":
                started.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release generation")
            return {"reply": "回答：" + messages[-1]["content"], "meme_id": "meme-0002"}

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(manager.respond, "a", "第一句", generate)
            self.assertTrue(started.wait(1))
            self.clock.set(5)
            second = pool.submit(manager.respond, "a", "第二句", generate)
            self.assertTrue(second_arrived.wait(1))
            # Generation starts after the second arrival's own timeout, but
            # the accepted request still has the first request's context.
            self.clock.set(16)
            release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        self.assertEqual([m["content"] for m in snapshots[1]], ["第一句", "回答：第一句", "第二句"])
        self.assertEqual(manager.history("a"), [])

    def test_reset_during_generation_keeps_old_context_out_of_new_conversation(self) -> None:
        started = threading.Event()
        release = threading.Event()
        old_context: list[dict[str, str]] = []

        def generate(messages: list[dict[str, str]]) -> dict[str, str]:
            old_context.extend(messages)
            started.set()
            if not release.wait(2):
                raise TimeoutError("test did not release generation")
            return {"reply": "旧的慢回答", "meme_id": "meme-0002"}

        self.service.reply("前面的历史")
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.service.sessions.respond, "default", "慢问题", generate)
            self.assertTrue(started.wait(1))
            self.assertTrue(self.service.reset())
            self.assertEqual(self.service.sessions.history("default"), [])
            release.set()
            self.assertEqual(future.result(timeout=2)["reply"], "旧的慢回答")
        self.assertEqual(len(old_context), 3)
        self.service.reply("新的问题")
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "新的问题"}])

    def test_new_arrival_after_timeout_waits_for_old_request_but_has_fresh_history(self) -> None:
        started = threading.Event()
        release = threading.Event()
        second_arrived = threading.Event()
        calls: list[list[dict[str, str]]] = []

        def clock() -> float:
            received_at = self.clock()
            if threading.current_thread().name.endswith("_1"):
                second_arrived.set()
            return received_at

        manager = SessionManager(clock=clock)
        self.addCleanup(manager.close)

        def generate(messages: list[dict[str, str]]) -> dict[str, str]:
            calls.append([dict(message) for message in messages])
            if messages[-1]["content"] == "旧问题":
                started.set()
                if not release.wait(2):
                    raise TimeoutError("test did not release generation")
            return {"reply": "回答", "meme_id": "meme-0002"}

        with ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(manager.respond, "a", "旧问题", generate)
            self.assertTrue(started.wait(1))
            self.clock.set(10)
            new = pool.submit(manager.respond, "a", "新问题", generate)
            self.assertTrue(second_arrived.wait(1))
            release.set()
            old.result(timeout=2)
            new.result(timeout=2)
        self.assertEqual(calls[1], [{"role": "user", "content": "新问题"}])
        self.assertEqual(manager.history("a")[0]["content"], "新问题")

    def test_different_sessions_can_generate_concurrently(self) -> None:
        barrier = threading.Barrier(2)

        def generate(messages: list[dict[str, str]]) -> dict[str, str]:
            barrier.wait(timeout=1)
            return {"reply": messages[-1]["content"], "meme_id": "meme-0002"}

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.service.sessions.respond, session, "问题", generate) for session in ("a", "b")]
            self.assertEqual([future.result(timeout=2)["reply"] for future in futures], ["问题", "问题"])

    def test_empty_input_and_invalid_settings_fail_without_creating_history(self) -> None:
        with self.assertRaises(ValueError):
            self.service.reply("   ")
        with self.assertRaises(ValueError):
            self.service.reply("问题", "")
        self.assertEqual(self.service.sessions.active_session_count, 0)
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                SessionManager(idle_timeout=timeout)


if __name__ == "__main__":
    unittest.main()
