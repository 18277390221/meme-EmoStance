"""Protocol and Pi client checks against local TCP stubs with fake tokens."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from communication.common import FRAME_LIMIT, Peer, read_message
from model_information.pi_client import AuthenticationError, InferenceClient, RemoteError
from model_information.protocol import (
    ValidationError, build_result, decode_result, validate_reply, validate_text,
)


TOKEN = "local-only-client-test-token"


class ProtocolTests(unittest.TestCase):
    def test_round_trip_keeps_legacy_text_and_public_fields(self):
        result = {"reply": "你好！\n第二行 😺", "meme_id": "meme-0008"}
        message = build_result("one", {**result, "distance": 0.2})
        self.assertEqual(decode_result(message), result)
        self.assertEqual(json.loads(message["text"]), result)
        self.assertEqual(set(message), {"type", "id", "text", "reply", "meme_id", "reply_text"})
        self.assertEqual(message["reply_text"], result["reply"])

    def test_legacy_and_direct_reply_fields_are_supported(self):
        payload = {"reply": "你好", "meme_id": "meme-0002"}
        self.assertEqual(decode_result({"type": "result", "id": "a", "text": json.dumps(payload)}), payload)
        self.assertEqual(decode_result({"type": "result", "id": "a", **payload}), payload)

    def test_invalid_reply_and_ids_are_rejected(self):
        for value in (None, {}, {"reply": " ", "meme_id": "meme-0002"},
                      {"reply": "x", "meme_id": "meme-02"},
                      {"reply": "x", "meme_id": "meme-１２３４"},
                      {"reply": "x" * 16001, "meme_id": "meme-0002"}):
            with self.subTest(value=str(value)[:40]), self.assertRaises(ValueError):
                validate_reply(value)
        with self.assertRaises(ValueError):
            build_result("", {"reply": "x", "meme_id": "meme-0002"})
        self.assertEqual(validate_text("x" * 16000), "x" * 16000)

    def test_malformed_or_conflicting_envelopes_are_rejected(self):
        for message in (
            {"type": "push", "id": "a"},
            {"type": "result", "id": "a", "text": "not JSON"},
            {"type": "result", "id": "a", "text": "[]"},
            {"type": "result", "id": "a", "text": 5},
            {"type": "result", "id": "a"},
            {**build_result("a", {"reply": "first", "meme_id": "meme-0002"}), "reply": "second"},
            {**build_result("a", {"reply": "first", "meme_id": "meme-0002"}), "reply_text": "second"},
        ):
            with self.subTest(message=message), self.assertRaises(ValidationError):
                decode_result(message)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.writers = set()
        self.connection_tasks = set()
        self.hellos = []
        self.requests = []
        self.pings = 0
        self.connected_event = asyncio.Event()
        self.clients = []
        self.server = None

    async def asyncTearDown(self):
        for client in self.clients:
            await client.close()
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for writer in self.writers:
            writer.close()
        tasks = list(self.connection_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def launch(self, on_request=None, *, reject_auth=False, legacy_hello=False):
        async def connected(reader, writer):
            task = asyncio.current_task()
            self.connection_tasks.add(task)
            self.writers.add(writer)
            peer = Peer(writer)
            try:
                hello = await read_message(reader)
                self.hellos.append(hello)
                if reject_auth or hello.get("token") != TOKEN:
                    await peer.send({"type": "error", "code": "auth_failed", "message": "rejected"})
                    return
                reply = {"type": "hello_ok", "name": "dummy", "client_id": f"connection-{len(self.hellos)}"}
                if not legacy_hello:
                    reply["session_id"] = hello["session_id"]
                await peer.send(reply)
                self.connected_event.set()
                while True:
                    message = await read_message(reader)
                    if message["type"] == "ping":
                        self.pings += 1
                        await peer.send({"type": "pong"})
                    elif message["type"] == "request":
                        self.requests.append(message)
                        if on_request is None:
                            await peer.send(build_result(message["id"], {"reply": message["text"], "meme_id": "meme-0008"}))
                        else:
                            await on_request(message, peer)
            except (EOFError, ConnectionError, OSError):
                pass
            finally:
                await peer.close()
                self.writers.discard(writer)
                self.connection_tasks.discard(task)

        self.server = await asyncio.start_server(connected, "127.0.0.1", 0, limit=FRAME_LIMIT)
        port = self.server.sockets[0].getsockname()[1]
        client = InferenceClient("127.0.0.1", port, token=TOKEN)
        self.clients.append(client)
        return client

    async def test_chinese_newline_emoji_request_and_stable_session_hello(self):
        client = await self.launch()
        self.assertIs(await client.connect(), client)
        result = await client.request("你好\n第二行 😺")
        self.assertEqual(result, {"reply": "你好\n第二行 😺", "meme_id": "meme-0008"})
        self.assertEqual(self.hellos[0]["session_id"], client.session_id)
        self.assertTrue(client.connected)
        self.assertEqual(client.pending_count, 0)

    async def test_old_server_hello_without_session_id_is_compatible(self):
        client = await self.launch(legacy_hello=True)
        self.assertEqual((await client.request("旧握手"))["reply"], "旧握手")

    async def test_request_errors_and_unknown_ids_do_not_mix_replies(self):
        async def reply(message, peer):
            await peer.send(build_result("unrelated-id", {"reply": "wrong", "meme_id": "meme-0002"}))
            if message["text"] == "bad":
                await peer.send({"type": "error", "id": message["id"], "code": "handler_error", "message": "test failure"})
            else:
                await peer.send(build_result(message["id"], {"reply": "good", "meme_id": "meme-0062"}))
        client = await self.launch(reply)
        results = await asyncio.gather(client.request("bad"), client.request("good"), return_exceptions=True)
        self.assertIsInstance(results[0], RemoteError)
        self.assertEqual(results[0].code, "handler_error")
        self.assertEqual(results[1], {"reply": "good", "meme_id": "meme-0062"})

    async def test_eight_pending_requests_limit(self):
        all_received = asyncio.Event()
        peers = []

        async def hold(message, peer):
            peers.append((message, peer))
            if len(peers) == 8:
                all_received.set()
        client = await self.launch(hold)
        await client.start()
        jobs = [asyncio.create_task(client.request(str(index))) for index in range(8)]
        try:
            await asyncio.wait_for(all_received.wait(), 2)
            with self.assertRaises(RuntimeError):
                await client.request("ninth")
            self.assertEqual(len(self.requests), 8)
            for message, peer in peers:
                await peer.send(build_result(message["id"], {"reply": "ok", "meme_id": "meme-0002"}))
            self.assertEqual(len(await asyncio.gather(*jobs)), 8)
        finally:
            for task in jobs:
                task.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

    async def test_disconnect_fails_pending_without_resend_and_reconnect_keeps_session(self):
        async def reply(message, peer):
            if message["text"] == "disconnect-once":
                peer.writer.close()
            else:
                await peer.send(build_result(message["id"], {"reply": "after reconnect", "meme_id": "meme-0002"}))
        client = await self.launch(reply)
        with self.assertRaises(ConnectionError):
            await client.request("disconnect-once")
        result = await client.request("new request")
        self.assertEqual(result["reply"], "after reconnect")
        self.assertEqual([message["text"] for message in self.requests], ["disconnect-once", "new request"])
        self.assertEqual(len(self.hellos), 2)
        self.assertEqual(self.hellos[0]["session_id"], self.hellos[1]["session_id"])

    async def test_authentication_failure_is_fatal_and_not_retried(self):
        client = await self.launch(reject_auth=True)
        with self.assertRaises(AuthenticationError):
            await client.start()
        with self.assertRaises(AuthenticationError):
            await client.request("never sent")
        self.assertEqual(len(self.hellos), 1)
        self.assertEqual(self.requests, [])

    async def test_heartbeat_and_push_continue_while_request_is_pending(self):
        request_received = asyncio.Event()

        async def hold(message, peer):
            await peer.send({"type": "push", "text": "主动推送"})
            request_received.set()
        client = await self.launch(hold)
        with patch("model_information.pi_client.HEARTBEAT_INTERVAL", 0.02):
            job = asyncio.create_task(client.request("held"))
            try:
                await asyncio.wait_for(request_received.wait(), 2)
                self.assertEqual((await client.pushes.get())["text"], "主动推送")
                deadline = asyncio.get_running_loop().time() + 1
                while self.pings < 2 and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.01)
                self.assertGreaterEqual(self.pings, 2)
                self.assertFalse(job.done())
                await client.close()
                with self.assertRaises(ConnectionError):
                    await job
            finally:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)

    async def test_timeout_does_not_resend_or_corrupt_next_reply(self):
        async def reply(message, peer):
            if message["text"] != "timeout":
                await peer.send(build_result(message["id"], {"reply": "next", "meme_id": "meme-0002"}))
        client = await self.launch(reply)
        with self.assertRaises(asyncio.TimeoutError):
            await client.request("timeout", timeout=0.02)
        self.assertEqual(client.pending_count, 0)
        self.assertEqual((await client.request("next"))["reply"], "next")
        self.assertEqual([message["text"] for message in self.requests], ["timeout", "next"])


if __name__ == "__main__":
    unittest.main()
