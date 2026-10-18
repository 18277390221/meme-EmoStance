"""Cross-component checks: shipped Pi socket client against the real server."""
from __future__ import annotations

import asyncio
import copy
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from meme_software.player.desktop.server_link import ServerLink
from model_information_qwen_v3.server import InferenceServer
from model_information_qwen_v3.sessions import DialogueManager


TOKEN = "local-player-handshake-test-only"


class Engine:
    def __init__(self):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()

    def reply(self, messages):
        self.calls.append(copy.deepcopy(messages))
        text = messages[-1]["content"]
        if text == "等待断线测试":
            self.started.set()
            if not self.release.wait(8):
                raise TimeoutError("test model was not released")
        return {"reply": "收到：" + text, "meme_id": "meme-0062"}


class PlayerHandshakeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = Engine()
        self.now = 100.0
        self.manager = DialogueManager(self.engine, clock=lambda: self.now, cleanup_interval=None)
        self.server = InferenceServer(self.engine, TOKEN, manager=self.manager)
        self.logs = patch("model_information_qwen_v3.server.log")
        self.logs.start()
        journal = tempfile.TemporaryDirectory()
        self.addCleanup(journal.cleanup)
        journal_patch = patch("model_information_qwen_v3.dialogue_log.DEFAULT_LOG_PATH", Path(journal.name) / "conversation.log")
        journal_patch.start()
        self.addCleanup(journal_patch.stop)
        await self.server.start("127.0.0.1", 0)
        self.ready = []
        self.link = ServerLink(on_ready=self.ready.append)
        self.events = []
        with patch("meme_software.player.desktop.server_link.load_token", return_value=TOKEN):
            self.link.start("127.0.0.1", self.server.addresses[0][1], "unused-test-token-file")
        await self.event("connected")

    async def asyncTearDown(self):
        self.engine.release.set()
        await asyncio.to_thread(self.link.close, 0)
        await self.server.close()
        self.logs.stop()

    async def until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("timed out waiting for the cross-component state")
            await asyncio.sleep(.01)

    async def event(self, kind, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.events.extend(self.link.drain())
            for index, event in enumerate(self.events):
                if event["type"] == kind:
                    return self.events.pop(index)
            await asyncio.sleep(.01)
        self.fail(f"missing {kind!r}; events={self.events!r}")

    async def finish_playback(self, key):
        self.assertTrue(self.link.playback_done(key))

        def acknowledged():
            with self.link.lock:
                return key not in self.link.playback_receipts

        await self.until(acknowledged)

    async def test_full_turn_requires_final_ack_result_and_playback_done(self):
        self.assertTrue(self.link.model_ready)
        self.assertTrue(self.ready[-1])
        self.assertTrue(self.link.speech_start())
        key = self.link.submit("你好")
        self.assertIsInstance(key, str)
        accepted = await self.event("accepted")
        self.assertEqual(accepted["id"], key)
        reply = await self.event("reply")
        self.assertEqual(reply["id"], key)
        self.assertEqual(reply["reply_text"], "收到：你好")
        self.assertIsNone(self.link.submit("不能插队"))
        await self.finish_playback(key)
        following = self.link.submit("接着聊")
        self.assertIsInstance(following, str)
        self.assertEqual((await self.event("reply"))["id"], following)
        await self.finish_playback(following)
        self.assertEqual(len(self.engine.calls), 2)
        self.assertEqual(len(self.engine.calls[1]), 3)
        self.assertEqual(len(self.manager.history(self.link.session_id)), 4)

    async def test_disconnect_after_accepted_recovers_original_result_once(self):
        key = self.link.submit("等待断线测试")
        self.assertEqual((await self.event("accepted"))["id"], key)
        await self.until(self.engine.started.is_set)
        epoch = self.link.server_instance_id
        self.link.socket.shutdown(socket.SHUT_RDWR)
        await self.event("disconnected")
        await self.until(lambda: not self.link.model_ready)
        self.assertTrue(self.link.recovering)
        self.assertIsNone(self.link.submit("断线时不能说"))
        self.engine.release.set()
        await self.event("connected")
        result = await self.event("reply")
        self.assertEqual(result["id"], key)
        self.assertRegex(result["conversation_id"], r"^[a-f0-9]{32}$")
        self.assertEqual(result["conversation_id"], self.link.conversation_id)
        self.assertEqual(self.link.server_instance_id, epoch)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(self.manager.history(self.link.session_id)), 2)
        # Querying a completed request can resend its frame, but the shipped
        # player must not produce another media event for the same request.
        self.assertTrue(self.link._queue_control({"type": "request_status", "id": key}))
        self.assertTrue(self.link._queue_control({"type": "speech_cancel"}))
        await asyncio.sleep(.25)
        self.events.extend(self.link.drain())
        self.assertFalse(any(e["type"] == "reply" for e in self.events))
        self.assertEqual(len(self.engine.calls), 1)
        await self.finish_playback(key)

    async def test_rolling_window_changes_background_after_five_slides_without_losing_context(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is needed for the directory media integration")
        await asyncio.to_thread(self.link.close, 0)
        await self.server.close()
        self.server = InferenceServer(self.engine, TOKEN)
        self.manager = self.server.manager
        self.manager.clock = lambda: self.now
        self.server.request_cache.clock = self.manager.clock
        self.assertEqual(self.manager.max_turns, 4)
        await self.server.start("127.0.0.1", 0)
        self.link = ServerLink(on_ready=self.ready.append)
        self.events = []
        with patch("meme_software.player.desktop.server_link.load_token", return_value=TOKEN):
            self.link.start("127.0.0.1", self.server.addresses[0][1], "unused-test-token-file")
        await self.event("connected")
        desktop = Path(__file__).resolve().parents[2] / "meme_software/player/desktop"
        with patch.object(sys, "path", [str(desktop), str(desktop.parent), *sys.path]):
            from directory_library import DirectoryPlayer
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "sucai_meme/meme_gif"
            background = original.parent / "meme_gif_background"
            original.mkdir(parents=True)
            background.mkdir()
            player_root = root / "player"
            (player_root / "website").mkdir(parents=True)
            (player_root / "website/library.json").write_text('{"items":[],"scenarios":[]}')
            for meme_id in ("meme-0001", "meme-0062"):
                Image.new("RGB", (2, 2), "black").save(original / (meme_id + "-cat.gif"))
                for number in range(1, 11):
                    Image.new("RGB", (2, 2), (number * 20, 0, 0)).save(
                        background / (meme_id + "-cat-" + str(number) + ".gif"))
            choices = iter((1, 3, 10, 7, 2))
            player = DirectoryPlayer(player_root, original, choice=lambda values: next(choices))
            original_reply = self.engine.reply

            def reply(messages):
                result = original_reply(messages)
                result["meme_id"] = "meme-0001" if len(self.engine.calls) % 2 else "meme-0062"
                return result

            self.engine.reply = reply
            events, bodies = [], []
            for turn in range(17):
                key = self.link.submit("继续聊" + str(turn))
                result = await self.event("reply")
                revision = max(0, turn + 1 - 4) // 5
                self.assertEqual(result["background_revision"], revision)
                body = {"meme_id": result["meme_id"], "reply_text": result["reply_text"],
                        "conversation_id": result["conversation_id"],
                        "background_revision": result["background_revision"],
                        "request_id": "server-" + key}
                bodies.append(body)
                event = player.respond(body)
                events.append(event)
                self.assertEqual(event["background_revision"], revision)
                row = player.read_native_asset(event["meme_id"], background_id=event["background_id"])
                expected_background = (3, 10, 7)[revision]
                self.assertTrue(row["visual_name"].endswith(f"-{expected_background}.gif"))
                self.assertEqual(background, Path(row["source_path"]).parent)
                self.assertLessEqual(len(self.manager.history(self.link.session_id)), 8)
                await self.finish_playback(key)
            self.assertEqual(1, len({event["conversation_id"] for event in events}))
            self.assertEqual({3}, {event["background_id"] for event in events[:8]})
            self.assertEqual({10}, {event["background_id"] for event in events[8:13]})
            self.assertEqual({7}, {event["background_id"] for event in events[13:]})
            self.assertEqual([len(call) for call in self.engine.calls],
                             [1,3,5,7] + [9] * 13)
            self.assertEqual(self.engine.calls[5][0], {"role":"user", "content":"继续聊1"})
            self.assertEqual(self.manager.history(self.link.session_id)[0],
                             {"role":"user", "content":"继续聊13"})
            # Cached media and network receipts keep the original background and
            # revision without changing the current selection or regenerating.
            self.assertEqual(player.respond(bodies[0]), events[0])
            self.assertEqual(player.background_id, 7)
            self.assertTrue(self.link._queue_control({"type":"request_status", "id":bodies[0]["request_id"][7:]}))
            await asyncio.sleep(.15)
            self.assertFalse(any(event["type"] == "reply" for event in self.link.drain()))
            self.assertEqual(len(self.engine.calls), 17)
            self.now += 10
            key = self.link.submit("空闲后的新对话")
            result = await self.event("reply")
            self.assertEqual(result["background_revision"], 0)
            following = player.respond({"meme_id": result["meme_id"], "reply_text": result["reply_text"],
                                       "conversation_id": result["conversation_id"],
                                       "background_revision": result["background_revision"],
                                       "request_id": "server-" + key})
            self.assertNotEqual(events[-1]["conversation_id"], following["conversation_id"])
            self.assertEqual(2, following["background_id"])
            frozen = player.read_native_asset(events[0]["meme_id"], background_id=events[0]["background_id"])
            self.assertTrue(frozen["visual_name"].endswith("-3.gif"))
            await self.finish_playback(key)


if __name__ == "__main__":
    unittest.main()
