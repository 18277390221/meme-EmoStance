"""Repeated-reply rollover, including queued and detached conversations."""
from __future__ import annotations

import asyncio
import unittest

from model_information.sessions import DialogueManager
from model_information.tests import test_sessions as sessions


class ScriptedReplyEngine(sessions.RecordingEngine):
    """Keep admission/failure behavior while controlling successful reply text."""

    def __init__(self, replies, **kwargs):
        super().__init__(**kwargs)
        self.replies = replies

    def reply(self, messages):
        result = super().reply(messages)
        result["reply"] = self.replies.get(messages[-1]["content"], result["reply"])
        if "meme_id" in result:
            result["meme_id"] = f"meme-{len(self.calls):04d}"
        return result


class RepeatedReplyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.managers = []
        self.engines = []

    async def asyncTearDown(self):
        for engine in self.engines:
            engine.release.set()
        for manager in self.managers:
            await manager.close()

    def manager(self, engine, **kwargs):
        clock = sessions.FakeClock()
        self.engines.append(engine)
        manager = DialogueManager(engine, clock=clock, cleanup_interval=None, **kwargs)
        self.managers.append(manager)
        manager.start()
        return manager, engine, clock

    async def test_two_identical_replies_reset_next_turn_even_without_turn_limit(self):
        engine = ScriptedReplyEngine({"第一句": "同一句回复", "第二句": "同一句回复"})
        manager, _, _ = self.manager(engine, max_turns=None)
        first = manager.accept("a", "第一句")
        first_result = await first
        second = manager.accept("a", "第二句")
        second_result = await second
        self.assertEqual(first_result["reply"], second_result["reply"])
        self.assertNotEqual(first_result["meme_id"], second_result["meme_id"])
        self.assertEqual(first.conversation_id, second.conversation_id)
        self.assertEqual(len(manager.history("a")), 4)
        third = manager.accept("a", "新问题")
        await third
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "新问题"}])
        self.assertNotEqual(second.conversation_id, third.conversation_id)
        self.assertEqual([m["content"] for m in manager.history("a")], ["新问题", "回答：新问题"])

    async def test_only_adjacent_exact_reply_text_matches_trigger_reset(self):
        engine = ScriptedReplyEngine({"一": "相同", "二": "不同", "三": "相同", "四": "相同 ", "五": "相同。", "六": "相同。"})
        manager, _, _ = self.manager(engine, max_turns=None)
        futures = []
        for text in ("一", "二", "三", "四", "五", "六"):
            future = manager.accept("a", text)
            await future
            futures.append(future)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 7, 9, 11])
        self.assertEqual(len({future.conversation_id for future in futures}), 1)
        seventh = manager.accept("a", "七")
        await seventh
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "七"}])
        self.assertNotEqual(futures[-1].conversation_id, seventh.conversation_id)

    async def test_repeated_reply_rollover_migrates_already_queued_requests_in_fifo_order(self):
        engine = ScriptedReplyEngine({"一": "复读", "二": "复读"}, block_text="二")
        manager, _, _ = self.manager(engine, max_turns=6)
        first = manager.accept("a", "一")
        await first
        second = manager.accept("a", "二")
        await sessions.wait_event(engine.started)
        third = manager.accept("a", "三")
        fourth = manager.accept("a", "四")
        engine.release.set()
        await asyncio.gather(second, third, fourth)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 1, 3])
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["三", "回答：三", "四"])
        self.assertEqual(first.conversation_id, second.conversation_id)
        self.assertNotEqual(second.conversation_id, third.conversation_id)
        self.assertEqual(third.conversation_id, fourth.conversation_id)

    async def test_failed_and_invalid_attempts_do_not_enter_repeat_comparison(self):
        engine = ScriptedReplyEngine({"一": "复读", "二": "复读"})
        manager, _, _ = self.manager(engine)
        first = manager.accept("a", "一")
        await first
        with self.assertRaisesRegex(RuntimeError, "fake engine failure"):
            await manager.accept("a", "失败")
        with self.assertRaises(ValueError):
            await manager.accept("a", "无效结果")
        self.assertEqual(len(manager.history("a")), 2)
        second = manager.accept("a", "二")
        await second
        self.assertEqual(first.conversation_id, second.conversation_id)
        with self.assertRaisesRegex(RuntimeError, "fake engine failure"):
            await manager.accept("a", "失败")
        self.assertEqual(manager.history("a"), [])
        third = manager.accept("a", "新一")
        await third
        self.assertNotEqual(second.conversation_id, third.conversation_id)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 3, 3, 1, 1])

    async def test_repeated_reply_reset_is_independent_for_each_device(self):
        engine = ScriptedReplyEngine({"甲一": "复读", "甲二": "复读", "乙一": "复读"})
        manager, _, _ = self.manager(engine)
        await manager.accept("a", "甲一")
        other_first = manager.accept("b", "乙一")
        await other_first
        await manager.accept("a", "甲二")
        await manager.accept("a", "甲三")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "甲三"}])
        other_second = manager.accept("b", "乙二")
        await other_second
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["乙一", "复读", "乙二"])
        self.assertEqual(other_first.conversation_id, other_second.conversation_id)

    async def test_repeat_rollover_of_detached_queue_cannot_revive_idle_or_manual_reset(self):
        for reset_kind in ("idle", "manual"):
            with self.subTest(reset=reset_kind):
                engine = ScriptedReplyEngine({"旧一": "复读", "旧二": "复读"}, block_text="旧二")
                manager, _, clock = self.manager(engine)
                first = manager.accept("a", "旧一")
                await first
                second = manager.accept("a", "旧二")
                await sessions.wait_event(engine.started)
                third = manager.accept("a", "旧三")
                if reset_kind == "idle":
                    clock.set(10)
                else:
                    self.assertTrue(manager.reset("a"))
                fresh = manager.accept("a", "新问题")
                engine.release.set()
                await asyncio.gather(second, third, fresh)
                self.assertEqual([len(call) for call in engine.calls], [1, 3, 1, 1])
                self.assertEqual(first.conversation_id, second.conversation_id)
                self.assertEqual(len({second.conversation_id, third.conversation_id, fresh.conversation_id}), 3)
                self.assertEqual([m["content"] for m in manager.history("a")], ["新问题", "回答：新问题"])

    async def test_repeat_starts_fresh_conversation_even_when_context_window_is_full(self):
        engine = ScriptedReplyEngine({"第4轮": "复读", "第5轮": "复读"})
        manager, _, _ = self.manager(engine)
        previous = None
        for number in range(1, 6):
            previous = manager.accept("a", f"第{number}轮")
            await previous
        sixth = manager.accept("a", "第6轮")
        await sixth
        seventh = manager.accept("a", "第7轮")
        await seventh
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 7, 9, 1, 3])
        self.assertNotEqual(previous.conversation_id, sixth.conversation_id)
        self.assertEqual(sixth.conversation_id, seventh.conversation_id)

    async def test_single_pair_window_retains_independent_two_reply_repeat_detection(self):
        engine = ScriptedReplyEngine({"一": "复读", "二": "复读"})
        manager, _, _ = self.manager(engine, max_turns=1)
        first = manager.accept("a", "一")
        await first
        second = manager.accept("a", "二")
        await second
        self.assertEqual(len(manager.history("a")), 2)
        self.assertEqual(first.conversation_id, second.conversation_id)
        third = manager.accept("a", "三")
        await third
        self.assertNotEqual(second.conversation_id, third.conversation_id)
        self.assertEqual(third.background_revision, 0)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 1])

    async def test_repeated_reply_reset_restarts_background_revision_after_sliding(self):
        engine = ScriptedReplyEngine({"第9轮": "复读", "第10轮": "复读"})
        manager, _, _ = self.manager(engine)
        for number in range(1, 11):
            old = manager.accept("a", f"第{number}轮")
            await old
        self.assertEqual(old.background_revision, 1)
        fresh = manager.accept("a", "新问题")
        await fresh
        self.assertNotEqual(old.conversation_id, fresh.conversation_id)
        self.assertEqual(fresh.background_revision, 0)
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "新问题"}])
