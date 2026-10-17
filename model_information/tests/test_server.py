"""Real loopback TCP checks for model admission, sessions, and wire results."""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import unittest
from unittest.mock import patch

from communication.common import FRAME_LIMIT, Peer, read_message
from model_information.pi_client import AuthenticationError, InferenceClient, RemoteError
from model_information.server import InferenceServer
from model_information.sessions import DialogueManager
from model_information.errors import PromptTooLong


TOKEN = "local-only-server-test-token"


class FakeClock:
    def __init__(self):
        self.value = 0.
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def set(self, value):
        with self.lock:
            self.value = float(value)


class RecordingEngine:
    def __init__(self, block_text=None):
        self.calls = []
        self.block_text = block_text
        self.started = threading.Event()
        self.release = threading.Event()

    def reply(self, messages):
        self.calls.append(copy.deepcopy(messages))
        text = messages[-1]["content"]
        if text == self.block_text:
            self.started.set()
            if not self.release.wait(3):
                raise TimeoutError("test did not release its fake model")
        if text == "失败":
            raise RuntimeError("private-engine-error-sentinel")
        if text == "过长":
            raise PromptTooLong("private-token-budget-details")
        return {"reply": "回答：" + text, "meme_id": "meme-0062", "internal": "not public"}


async def until(predicate, timeout=1):
    async def poll():
        while not predicate():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), timeout)


class ServerIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.clock = FakeClock()
        self.engine = RecordingEngine()
        self.server = None
        self.clients = []
        self.peers = []
        self.logs = []
        self.log_patch = patch("model_information.server.log", side_effect=self.logs.append)
        self.log_patch.start()

    async def asyncTearDown(self):
        self.engine.release.set()
        for client in self.clients:
            await client.close()
        for peer in self.peers:
            await peer.close()
        if self.server is not None:
            await self.server.close()
        self.log_patch.stop()

    async def launch(self, *, block_text=None, **manager_options):
        self.engine.block_text = block_text
        manager = DialogueManager(self.engine, clock=self.clock, cleanup_interval=None, **manager_options)
        self.server = await InferenceServer(self.engine, TOKEN, manager=manager).start("127.0.0.1", 0)
        self.port = self.server.addresses[0][1]
        return self.server

    async def raw(self, *, name="raspberrypi", session_id=None, token=TOKEN):
        reader, writer = await asyncio.open_connection("127.0.0.1", self.port, limit=FRAME_LIMIT)
        peer = Peer(writer)
        self.peers.append(peer)
        hello = {"type": "hello", "name": name, "token": token}
        if session_id is not None:
            hello["session_id"] = session_id
        await peer.send(hello)
        response = await read_message(reader, 1)
        return reader, peer, response

    async def ask_raw(self, reader, peer, text, request_id="one"):
        await peer.send({"type": "request", "id": request_id, "text": text})
        return await read_message(reader, 1)

    def client(self, *, name="raspberrypi", session_id=None, token=TOKEN):
        client = InferenceClient("127.0.0.1", self.port, name, token=token, session_id=session_id)
        self.clients.append(client)
        return client

    async def test_legacy_hello_receives_json_text_and_matching_direct_fields(self):
        await self.launch()
        reader, peer, hello = await self.raw()
        self.assertEqual(hello["type"], "hello_ok")
        self.assertEqual(hello["session_id"], hello["client_id"])
        text = "你好\n第二行 😺"
        message = await self.ask_raw(reader, peer, text, "legacy-id")
        expected = {"reply": "回答：" + text, "meme_id": "meme-0062"}
        self.assertEqual(set(message), {"type", "id", "text", "reply", "reply_text", "meme_id"})
        self.assertEqual(message["type"], "result")
        self.assertEqual(message["id"], "legacy-id")
        self.assertEqual(json.loads(message["text"]), expected)
        self.assertEqual({key: message[key] for key in expected}, expected)
        self.assertEqual(message["reply_text"], expected["reply"])

    async def test_actual_inference_client_receives_reply_and_meme_with_multiturn_history(self):
        await self.launch()
        client = self.client()
        self.assertIs(await client.connect(timeout=1), client)
        self.assertEqual(await client.request("你好 😺", timeout=1), {"reply": "回答：你好 😺", "meme_id": "meme-0062"})
        self.clock.set(3)
        self.assertEqual((await client.request("接着说", timeout=1))["reply"], "回答：接着说")
        self.assertEqual([m["content"] for m in self.engine.calls[1]], ["你好 😺", "回答：你好 😺", "接着说"])
        self.assertEqual(client.pending_count, 0)

    async def test_ping_keeps_transport_alive_without_extending_exact_10s_dialogue_deadline(self):
        await self.launch()
        reader, peer, _ = await self.raw()
        await self.ask_raw(reader, peer, "首句", "first")
        self.clock.set(9)
        await peer.send({"type": "ping"})
        self.assertEqual(await read_message(reader, 1), {"type": "pong"})
        self.clock.set(10)
        result = await self.ask_raw(reader, peer, "新对话", "at-ten")
        self.assertEqual(result["type"], "result")
        self.assertEqual(self.engine.calls[1], [{"role": "user", "content": "新对话"}])

    async def test_same_display_name_devices_get_separate_dialogue_histories(self):
        await self.launch()
        r1, p1, h1 = await self.raw(name="same-name")
        r2, p2, h2 = await self.raw(name="same-name")
        self.assertNotEqual(h1["session_id"], h2["session_id"])
        await self.ask_raw(r1, p1, "甲首句", "a-first")
        await self.ask_raw(r2, p2, "乙首句", "b-first")
        await self.ask_raw(r1, p1, "甲后续", "a-second")
        self.assertEqual(self.engine.calls[1], [{"role": "user", "content": "乙首句"}])
        self.assertEqual([m["content"] for m in self.engine.calls[2]], ["甲首句", "回答：甲首句", "甲后续"])

    async def test_slow_model_and_network_admission_preserve_old_fifo_but_start_new_segment(self):
        await self.launch(block_text="第一句")
        reader, peer, hello = await self.raw()
        await peer.send({"type": "request", "id": "first", "text": "第一句"})
        await until(self.engine.started.is_set)
        self.clock.set(5)
        await peer.send({"type": "request", "id": "second", "text": "第二句"})
        await until(lambda: self.server.manager.pending_count == 2)
        # Ping must be answered while model inference is blocked.
        await peer.send({"type": "ping"})
        self.assertEqual(await read_message(reader, 1), {"type": "pong"})
        self.clock.set(16)
        await peer.send({"type": "request", "id": "third", "text": "第三句是新段"})
        await until(lambda: self.server.manager.pending_count == 3)
        self.engine.release.set()
        replies = [await read_message(reader, 1) for _ in range(3)]
        self.assertEqual({r["id"] for r in replies}, {"first", "second", "third"})
        self.assertTrue(all(r["type"] == "result" for r in replies))
        self.assertEqual([len(history) for history in self.engine.calls], [1, 3, 1])
        self.assertEqual([m["content"] for m in self.engine.calls[1]], ["第一句", "回答：第一句", "第二句"])
        self.assertEqual([m["content"] for m in self.server.manager.history(hello["session_id"])], ["第三句是新段", "回答：第三句是新段"])

    async def test_stable_session_reconnect_retains_before_10s_and_resets_at_boundary(self):
        await self.launch()
        session_id = "fixed-random-device-session"
        r1, p1, h1 = await self.raw(session_id=session_id)
        await self.ask_raw(r1, p1, "连接一", "first")
        await p1.close()
        await until(lambda: not self.server.clients)
        self.clock.set(9)
        r2, p2, h2 = await self.raw(session_id=session_id)
        self.assertNotEqual(h1["client_id"], h2["client_id"])
        await self.ask_raw(r2, p2, "连接二", "second")
        self.assertEqual([m["content"] for m in self.engine.calls[1]], ["连接一", "回答：连接一", "连接二"])
        await p2.close()
        await until(lambda: not self.server.clients)
        self.clock.set(19)  # Exactly ten seconds after the second user arrival.
        r3, p3, h3 = await self.raw(session_id=session_id)
        self.assertEqual(h3["session_id"], session_id)
        await self.ask_raw(r3, p3, "连接三新段", "third")
        self.assertEqual(self.engine.calls[2], [{"role": "user", "content": "连接三新段"}])

    async def test_legacy_disconnect_does_not_resume_by_same_device_name(self):
        await self.launch()
        r1, p1, h1 = await self.raw(name="same-device")
        await self.ask_raw(r1, p1, "旧连接", "first")
        await p1.close()
        await until(lambda: not self.server.clients)
        self.assertEqual(self.server.manager.history(h1["session_id"]), [])
        self.clock.set(1)
        r2, p2, h2 = await self.raw(name="same-device")
        self.assertNotEqual(h1["session_id"], h2["session_id"])
        await self.ask_raw(r2, p2, "新连接", "second")
        self.assertEqual(self.engine.calls[1], [{"role": "user", "content": "新连接"}])

    async def test_auth_failure_on_legacy_wire_never_reaches_engine(self):
        await self.launch()
        reader, _, hello = await self.raw(token="incorrect-local-only-token")
        self.assertEqual(hello["type"], "error")
        self.assertEqual(hello["code"], "auth_failed")
        self.assertEqual(await asyncio.wait_for(reader.read(), 1), b"")
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.server.manager.pending_count, 0)

    async def test_actual_client_auth_failure_is_fatal_and_does_not_invoke_engine(self):
        await self.launch()
        client = self.client(token="incorrect-local-only-token")
        with self.assertRaises(AuthenticationError):
            await client.connect(timeout=1)
        with self.assertRaises(AuthenticationError):
            await client.request("不可发送", timeout=1)
        self.assertEqual(self.engine.calls, [])

    async def test_failure_returns_correlated_error_and_keeps_only_complete_history(self):
        await self.launch()
        client = self.client()
        await client.request("已成功", timeout=1)
        with self.assertRaises(RemoteError) as error:
            await client.request("失败", timeout=1)
        self.assertEqual(error.exception.code, "handler_error")
        self.assertNotIn("private-engine-error-sentinel", str(error.exception))
        await client.request("后续问题", timeout=1)
        self.assertEqual([m["content"] for m in self.engine.calls[-1]], ["已成功", "回答：已成功", "后续问题"])
        self.assertEqual(self.server.manager.pending_count, 0)

    async def test_prompt_budget_error_asks_for_shorter_input_without_partial_history(self):
        await self.launch()
        client = self.client()
        with self.assertRaises(RemoteError) as error:
            await client.request("过长", timeout=1)
        self.assertEqual(error.exception.code, "input_too_long")
        self.assertIn("较短", str(error.exception))
        self.assertNotIn("private-token-budget-details", str(error.exception))
        await client.request("短句", timeout=1)
        self.assertEqual(self.engine.calls[-1], [{"role": "user", "content": "短句"}])

    async def test_busy_reply_does_not_queue_rejected_question(self):
        await self.launch(block_text="慢问题", max_pending_per_session=1)
        reader, peer, _ = await self.raw()
        await peer.send({"type": "request", "id": "first", "text": "慢问题"})
        await until(self.engine.started.is_set)
        rejected = await self.ask_raw(reader, peer, "被拒绝", "second")
        self.assertEqual(rejected["type"], "error")
        self.assertEqual(rejected["id"], "second")
        self.assertEqual(rejected["code"], "busy")
        self.assertEqual(self.server.manager.pending_count, 1)
        self.engine.release.set()
        self.assertEqual((await read_message(reader, 1))["id"], "first")
        await self.ask_raw(reader, peer, "完成后继续", "third")
        self.assertEqual([m["content"] for m in self.engine.calls[-1]], ["慢问题", "回答：慢问题", "完成后继续"])

    async def test_duplicate_pending_id_is_rejected_without_running_twice(self):
        await self.launch(block_text="原请求")
        reader, peer, _ = await self.raw()
        await peer.send({"type": "request", "id": "same-id", "text": "原请求"})
        await until(self.engine.started.is_set)
        duplicate = await self.ask_raw(reader, peer, "重复请求", "same-id")
        self.assertEqual(duplicate["code"], "duplicate_request")
        self.engine.release.set()
        reply = await read_message(reader, 1)
        self.assertEqual(reply["type"], "result")
        self.assertEqual(reply["id"], "same-id")
        self.assertEqual(len(self.engine.calls), 1)

    async def test_invalid_network_request_does_not_change_user_deadline(self):
        await self.launch()
        reader, peer, _ = await self.raw()
        await self.ask_raw(reader, peer, "首句", "first")
        self.clock.set(9)
        for message in ({"type": "request", "id": "invalid", "text": " "},
                        {"type": "request", "id": 7, "text": "无效编号"},
                        {"type": "request", "id": "invalid", "text": "字" * 16001},
                        {"type": "unknown"}):
            await peer.send(message)
            error = await read_message(reader, 1)
            self.assertEqual(error["type"], "error")
            self.assertIn(error["code"], {"bad_request", "bad_type"})
        self.assertEqual(len(self.engine.calls), 1)
        self.clock.set(10)
        await self.ask_raw(reader, peer, "新段", "new")
        self.assertEqual(self.engine.calls[1], [{"role": "user", "content": "新段"}])


if __name__ == "__main__":
    unittest.main()
