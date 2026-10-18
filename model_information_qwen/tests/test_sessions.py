"""Session behavior at network admission, including queued slow-model calls."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import FrozenInstanceError
import gc
import threading
import unittest

from model_information_qwen.sessions import BusyError, DialogueManager
from model_information_qwen.turns import TurnResult


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
    def __init__(self, block_text=None, clock=None):
        self.calls = []
        self.block_text = block_text
        self.clock = clock
        self.started = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def reply(self, messages):
        with self.lock:
            self.calls.append(copy.deepcopy(messages))
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            text = messages[-1]["content"]
            if text == self.block_text:
                self.started.set()
                if not self.release.wait(2):
                    raise TimeoutError("test did not release its blocking model")
            if text == "失败":
                raise RuntimeError("fake engine failure")
            if text == "无效结果":
                return {"reply": "缺表情"}
            if text == "耗时8秒" and self.clock:
                self.clock.set(8)
            return {"reply": "回答：" + text, "meme_id": "meme-0008", "internal": "not public"}
        finally:
            with self.lock:
                self.active -= 1
            if messages[-1]["content"] == self.block_text:
                self.exited.set()


async def wait_event(event):
    async def poll():
        while not event.is_set():
            await asyncio.sleep(.001)
    await asyncio.wait_for(poll(), 1)


class DialogueManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.managers = []
        self.engines = []

    async def asyncTearDown(self):
        for engine in self.engines:
            engine.release.set()
        for manager in self.managers:
            await manager.close()

    def manager(self, engine=None, clock=None, **kwargs):
        engine = engine or RecordingEngine()
        clock = clock or FakeClock()
        self.engines.append(engine)
        manager = DialogueManager(engine, clock=clock, cleanup_interval=kwargs.pop("cleanup_interval", None), **kwargs)
        self.managers.append(manager)
        self.assertIs(manager.start(), manager)
        return manager, engine, clock

    async def test_start_is_idempotent_and_accept_requires_start(self):
        engine = RecordingEngine()
        manager = DialogueManager(engine)
        with self.assertRaisesRegex(RuntimeError, "start"):
            manager.accept("a", "问题")
        self.managers.append(manager)
        self.assertIs(manager.start(), manager)
        self.assertIs(manager.start(), manager)
        self.assertEqual(await manager.accept("a", "问题"), {"reply": "回答：问题", "meme_id": "meme-0008"})
        await manager.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            manager.accept("a", "关闭后")

    async def test_three_turns_keep_complete_ababa_context(self):
        manager, engine, clock = self.manager()
        for instant, text in [(0, "第一句"), (3, "第二句"), (6, "第三句")]:
            clock.set(instant)
            await manager.accept("connection-a", text)
        self.assertEqual(engine.calls[2], [
            {"role": "user", "content": "第一句"}, {"role": "assistant", "content": "回答：第一句"},
            {"role": "user", "content": "第二句"}, {"role": "assistant", "content": "回答：第二句"},
            {"role": "user", "content": "第三句"},
        ])
        self.assertEqual(len(manager.history("connection-a")), 6)
        self.assertEqual(manager.pending_count, 0)

    async def test_explicit_unlimited_keeps_twelve_successful_turns_in_one_conversation(self):
        manager, engine, clock = self.manager(max_turns=None)
        self.assertIsNone(manager.max_turns)
        expected = []
        conversation_ids = []
        for number in range(1, 13):
            clock.set(number)
            text = f"第{number}轮"
            expected.append({"role": "user", "content": text})
            future = manager.accept("a", text)
            result = await future
            conversation_ids.append(future.conversation_id)
            self.assertEqual(set(result), {"reply", "meme_id"})
            self.assertEqual(engine.calls[-1], expected)
            expected.append({"role": "assistant", "content": "回答：" + text})
            self.assertEqual(manager.history("a"), expected)
        self.assertEqual([len(call) for call in engine.calls], list(range(1, 24, 2)))
        self.assertEqual(len(set(conversation_ids)), 1)
        self.assertRegex(conversation_ids[0], r"^[0-9a-f]{32}$")

    async def test_conversation_ids_follow_idle_expiry_and_manual_reset(self):
        manager, engine, clock = self.manager()
        first = manager.accept("a", "第一句")
        await first
        clock.set(9.999)
        second = manager.accept("a", "第二句")
        await second
        self.assertEqual(first.conversation_id, second.conversation_id)
        clock.set(20)
        fresh = manager.accept("a", "空闲后")
        await fresh
        self.assertNotEqual(second.conversation_id, fresh.conversation_id)
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "空闲后"}])
        manager.reset("a")
        reset = manager.accept("a", "重置后")
        await reset
        self.assertNotEqual(fresh.conversation_id, reset.conversation_id)
        self.assertEqual(manager.history("a"), [
            {"role": "user", "content": "重置后"},
            {"role": "assistant", "content": "回答：重置后"},
        ])

    async def test_queued_one_pair_window_keeps_actual_conversation_id(self):
        engine = RecordingEngine(block_text="第一句")
        manager, _, _ = self.manager(engine, max_turns=1)
        first = manager.accept("a", "第一句")
        await wait_event(engine.started)
        second = manager.accept("a", "第二句")
        third = manager.accept("a", "第三句")
        engine.release.set()
        await asyncio.gather(first, second, third)
        self.assertEqual(len({future.conversation_id for future in (first, second, third)}), 1)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 3])
        self.assertEqual([message["content"] for message in engine.calls[-1]],
                         ["第二句", "回答：第二句", "第三句"])
        self.assertEqual(len(manager.history("a")), 2)

    async def test_late_old_reply_keeps_captured_conversation_id_after_reset(self):
        engine = RecordingEngine(block_text="旧慢问题")
        manager, _, _ = self.manager(engine)
        first = manager.accept("a", "旧历史")
        await first
        old = manager.accept("a", "旧慢问题")
        await wait_event(engine.started)
        manager.reset("a")
        new = manager.accept("a", "新问题")
        engine.release.set()
        await asyncio.gather(old, new)
        self.assertEqual(first.conversation_id, old.conversation_id)
        self.assertNotEqual(old.conversation_id, new.conversation_id)
        self.assertEqual(manager.history("a"), [
            {"role": "user", "content": "新问题"},
            {"role": "assistant", "content": "回答：新问题"},
        ])

    async def test_default_queued_requests_slide_without_changing_conversation(self):
        engine = RecordingEngine(block_text="第5轮")
        manager, _, _ = self.manager(engine)
        for number in range(1, 5):
            await manager.accept("a", f"第{number}轮")
        fifth = manager.accept("a", "第5轮")
        await wait_event(engine.started)
        queued = [manager.accept("a", f"第{number}轮") for number in range(6, 13)]
        self.assertEqual(manager.pending_count, 8)
        engine.release.set()
        await asyncio.gather(fifth, *queued)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 7] + [9]*8)
        self.assertEqual(engine.calls[4][0], {"role": "user", "content": "第1轮"})
        self.assertEqual(engine.calls[10][0], {"role": "user", "content": "第7轮"})
        self.assertEqual(len({future.conversation_id for future in [fifth, *queued]}), 1)
        self.assertEqual([future.background_revision for future in [fifth, *queued]], [0,0,0,0,1,1,1,1])
        self.assertEqual(len(manager.history("a")), 8)
        self.assertEqual(manager.pending_count, 0)

    async def test_default_nineteen_turns_slide_four_pairs_and_advance_background_at_nine_fourteen_nineteen(self):
        manager, engine, _ = self.manager()
        self.assertEqual(manager.max_turns, 4)
        futures = []
        for number in range(1, 20):
            future = manager.accept("a", f"第{number}轮")
            await future
            futures.append(future)
            self.assertLessEqual(len(manager.history("a")), 8)
            expected_numbers = range(max(1, number-3), number+1)
            self.assertEqual([message["content"] for message in manager.history("a")],
                [value for index in expected_numbers for value in (f"第{index}轮", f"回答：第{index}轮")])
        self.assertEqual([len(call) for call in engine.calls], [1,3,5,7] + [9]*15)
        self.assertEqual(len({future.conversation_id for future in futures}), 1)
        self.assertEqual([future.background_revision for future in futures], [0]*8 + [1]*5 + [2]*5 + [3])
        self.assertEqual(engine.calls[4][0], {"role": "user", "content": "第1轮"})
        self.assertEqual(engine.calls[5][0], {"role": "user", "content": "第2轮"})

    async def test_explicit_five_pair_window_preserves_its_own_size_and_background_boundary(self):
        manager, engine, _ = self.manager(max_turns=5)
        futures = []
        for number in range(1, 11):
            future = manager.accept("a", f"第{number}轮")
            await future
            futures.append(future)
        self.assertEqual([len(call) for call in engine.calls], [1,3,5,7,9] + [11]*5)
        self.assertEqual(len(manager.history("a")), 10)
        self.assertEqual(manager.history("a")[0], {"role":"user", "content":"第6轮"})
        self.assertEqual([future.background_revision for future in futures], [0]*9+[1])
        self.assertEqual(len({future.conversation_id for future in futures}), 1)

    async def test_failure_at_background_boundary_does_not_slide_or_change_revision(self):
        manager, _, _ = self.manager()
        prior = []
        for number in range(1,9):
            future = manager.accept("a", f"第{number}轮")
            await future
            prior.append(future)
        history = manager.history("a")
        for text, error in (("失败", RuntimeError), ("无效结果", ValueError)):
            with self.assertRaises(error):
                await manager.accept("a", text)
            self.assertEqual(manager.history("a"), history)
            self.assertEqual(manager._sessions["a"].current.window_slides, 4)
        ninth = manager.accept("a", "第9轮")
        await ninth
        self.assertEqual(prior[-1].background_revision, 0)
        self.assertEqual(ninth.background_revision, 1)
        self.assertEqual(prior[-1].conversation_id, ninth.conversation_id)
        with self.assertRaises(AttributeError):
            ninth.background_revision = 100

    async def test_idle_manual_and_repeat_resets_clear_window_and_background_metadata(self):
        for reset_kind in ("idle", "manual", "repeat"):
            with self.subTest(reset=reset_kind):
                engine = RecordingEngine()
                original = engine.reply
                def reply(messages):
                    result = original(messages)
                    if messages[-1]["content"] in ("重复一", "重复二"):
                        result["reply"] = "相同回复"
                    return result
                engine.reply = reply
                manager, _, clock = self.manager(engine)
                for number in range(1,11):
                    previous = manager.accept("a", f"第{number}轮")
                    await previous
                self.assertEqual(previous.background_revision, 1)
                if reset_kind == "idle":
                    clock.set(10)
                elif reset_kind == "manual":
                    manager.reset("a")
                else:
                    for text in ("重复一", "重复二"):
                        previous = manager.accept("a", text)
                        await previous
                fresh = manager.accept("a", "新问题")
                await fresh
                self.assertNotEqual(previous.conversation_id, fresh.conversation_id)
                self.assertEqual(fresh.background_revision, 0)
                self.assertEqual(manager._sessions["a"].current.window_slides, 0)
                self.assertEqual(engine.calls[-1], [{"role":"user", "content":"新问题"}])

    async def test_already_queued_requests_cross_window_boundary_in_same_conversation(self):
        engine = RecordingEngine(block_text="第二轮")
        manager, _, _ = self.manager(engine, max_turns=2)
        await manager.accept("a", "第一轮")
        second = manager.accept("a", "第二轮")
        await wait_event(engine.started)
        third = manager.accept("a", "第三轮")
        fourth = manager.accept("a", "第四轮")
        engine.release.set()
        await asyncio.gather(second, third, fourth)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 5])
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["第二轮", "回答：第二轮", "第三轮", "回答：第三轮", "第四轮"])
        self.assertEqual(len({future.conversation_id for future in (second, third, fourth)}), 1)
        self.assertEqual([m["content"] for m in manager.history("a")],
                         ["第三轮", "回答：第三轮", "第四轮", "回答：第四轮"])
        await manager.accept("a", "第五轮")
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["第三轮", "回答：第三轮", "第四轮", "回答：第四轮", "第五轮"])
        self.assertEqual(manager.pending_count, 0)

    async def test_failed_and_invalid_queued_replies_do_not_advance_window(self):
        manager, engine, _ = self.manager(max_turns=2)
        await manager.accept("a", "第一轮")
        failed = manager.accept("a", "失败")
        invalid = manager.accept("a", "无效结果")
        second = manager.accept("a", "第二轮")
        third = manager.accept("a", "第三轮")
        with self.assertRaisesRegex(RuntimeError, "fake engine failure"):
            await failed
        with self.assertRaises(ValueError):
            await invalid
        await asyncio.gather(second, third)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 3, 3, 5])
        self.assertEqual([m["content"] for m in engine.calls[-2]], ["第一轮", "回答：第一轮", "第二轮"])
        self.assertEqual([m["content"] for m in manager.history("a")], ["第二轮", "回答：第二轮", "第三轮", "回答：第三轮"])
        self.assertEqual(manager._sessions["a"].current.window_slides, 1)

    async def test_failed_attempt_at_full_window_does_not_remove_any_pair(self):
        manager, engine, _ = self.manager(max_turns=2)
        await manager.accept("a", "旧一")
        await manager.accept("a", "旧二")
        with self.assertRaisesRegex(RuntimeError, "fake engine failure"):
            await manager.accept("a", "失败")
        self.assertEqual([m["content"] for m in manager.history("a")], ["旧一", "回答：旧一", "旧二", "回答：旧二"])
        self.assertEqual(manager._sessions["a"].current.window_slides, 0)
        await manager.accept("a", "新一")
        await manager.accept("a", "新二")
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 5, 5])
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["旧二", "回答：旧二", "新一", "回答：新一", "新二"])

    async def test_window_slides_are_independent_for_each_session(self):
        manager, engine, _ = self.manager(max_turns=2)
        await manager.accept("a", "甲一")
        await manager.accept("b", "乙一")
        await manager.accept("a", "甲二")
        await manager.accept("a", "甲三")
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["甲一", "回答：甲一", "甲二", "回答：甲二", "甲三"])
        await manager.accept("b", "乙二")
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["乙一", "回答：乙一", "乙二"])
        self.assertEqual([m["content"] for m in manager.history("a")], ["甲二", "回答：甲二", "甲三", "回答：甲三"])
        self.assertEqual(len(manager.history("b")), 4)

    async def test_detached_queued_window_does_not_revive_history_after_idle_reset(self):
        engine = RecordingEngine(block_text="旧二")
        manager, _, clock = self.manager(engine, max_turns=2)
        await manager.accept("a", "旧一")
        second = manager.accept("a", "旧二")
        await wait_event(engine.started)
        third = manager.accept("a", "旧三")
        clock.set(10)
        fresh = manager.accept("a", "空闲后新问题")
        engine.release.set()
        await asyncio.gather(second, third, fresh)
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 5, 1])
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "空闲后新问题"}])
        self.assertEqual([m["content"] for m in manager.history("a")],
                         ["空闲后新问题", "回答：空闲后新问题"])

    async def test_different_connection_ids_keep_histories_isolated(self):
        manager, engine, _ = self.manager()
        await manager.accept("uuid-a", "设备一")
        await manager.accept("uuid-b", "同名的另一设备")
        await manager.accept("uuid-a", "继续一")
        self.assertEqual(engine.calls[1], [{"role": "user", "content": "同名的另一设备"}])
        self.assertEqual([message["content"] for message in engine.calls[2]], ["设备一", "回答：设备一", "继续一"])
        self.assertEqual(len(manager.history("uuid-b")), 2)

    async def test_9_999_keeps_context_but_exact_10_resets(self):
        manager, engine, clock = self.manager()
        await manager.accept("a", "第一句")
        clock.set(9.999)
        await manager.accept("a", "第二句")
        self.assertEqual(len(engine.calls[-1]), 3)
        clock.set(20)
        await manager.accept("boundary", "边界首句")
        clock.set(30)  # Integer timestamps exercise exactly ten seconds.
        await manager.accept("boundary", "新段")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "新段"}])

    async def test_deadline_uses_user_arrival_not_generation_completion_or_history_read(self):
        clock = FakeClock()
        manager, engine, _ = self.manager(RecordingEngine(clock=clock), clock)
        await manager.accept("a", "耗时8秒")
        clock.set(9)
        self.assertEqual(len(manager.history("a")), 2)
        # Reads/heartbeat handling do not count as user questions.
        clock.set(10)
        await manager.accept("a", "第二句")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "第二句"}])

    async def test_generation_beyond_10s_returns_result_without_reviving_history(self):
        engine = RecordingEngine(block_text="慢问题")
        manager, _, clock = self.manager(engine)
        result = manager.accept("a", "慢问题")
        await wait_event(engine.started)
        clock.set(11)
        self.assertEqual(manager.history("a"), [])
        engine.release.set()
        self.assertEqual((await result)["reply"], "回答：慢问题")
        self.assertEqual(manager.history("a"), [])

    async def test_queued_same_device_jobs_capture_arrival_before_model_wait(self):
        engine = RecordingEngine(block_text="第一句")
        manager, _, clock = self.manager(engine)
        first = manager.accept("a", "第一句")
        await wait_event(engine.started)
        clock.set(5)
        second = manager.accept("a", "第二句")
        self.assertEqual(manager.pending_count, 2)
        clock.set(16)
        engine.release.set()
        await asyncio.gather(first, second)
        self.assertEqual([m["content"] for m in engine.calls[1]], ["第一句", "回答：第一句", "第二句"])
        self.assertEqual(manager.history("a"), [])

    async def test_arrival_exactly_at_timeout_gets_fresh_context_behind_old_job(self):
        engine = RecordingEngine(block_text="旧问题")
        manager, _, clock = self.manager(engine)
        old = manager.accept("a", "旧问题")
        await wait_event(engine.started)
        clock.set(10)
        new = manager.accept("a", "新问题")
        engine.release.set()
        await asyncio.gather(old, new)
        self.assertEqual(engine.calls[1], [{"role": "user", "content": "新问题"}])
        self.assertEqual([m["content"] for m in manager.history("a")], ["新问题", "回答：新问题"])

    async def test_cross_device_gpu_wait_does_not_delay_last_user_arrival(self):
        engine = RecordingEngine(block_text="占用GPU")
        manager, _, clock = self.manager(engine)
        blocking = manager.accept("a", "占用GPU")
        await wait_event(engine.started)
        clock.set(4)
        first_b = manager.accept("b", "乙一")
        await asyncio.sleep(0)  # Its worker is waiting on another device's GPU.
        clock.set(9)
        second_b = manager.accept("b", "乙二")
        clock.set(20)
        engine.release.set()
        await asyncio.gather(blocking, first_b, second_b)
        self.assertEqual([m["content"] for m in engine.calls[2]], ["乙一", "回答：乙一", "乙二"])
        self.assertEqual(manager.history("b"), [])
        self.assertEqual(engine.max_active, 1)

    async def test_reset_detaches_pending_old_context_and_new_question_stays_fresh(self):
        engine = RecordingEngine(block_text="旧慢问题")
        manager, _, clock = self.manager(engine)
        await manager.accept("a", "旧历史")
        old = manager.accept("a", "旧慢问题")
        await wait_event(engine.started)
        self.assertTrue(manager.reset("a"))
        self.assertEqual(manager.history("a"), [])
        clock.set(1)
        new = manager.accept("a", "新问题")
        engine.release.set()
        await asyncio.gather(old, new)
        self.assertEqual([m["content"] for m in engine.calls[1]], ["旧历史", "回答：旧历史", "旧慢问题"])
        self.assertEqual(engine.calls[2], [{"role": "user", "content": "新问题"}])
        self.assertEqual([m["content"] for m in manager.history("a")], ["新问题", "回答：新问题"])
        self.assertFalse(manager.reset("unknown"))

    async def test_failed_job_leaves_no_partial_history_and_queue_continues(self):
        manager, engine, _ = self.manager()
        await manager.accept("a", "此前成功")
        failed = manager.accept("a", "失败")
        following = manager.accept("a", "后续")
        with self.assertRaisesRegex(RuntimeError, "fake engine failure"):
            await failed
        self.assertEqual((await following)["reply"], "回答：后续")
        self.assertEqual([m["content"] for m in engine.calls[-1]], ["此前成功", "回答：此前成功", "后续"])
        self.assertEqual(manager.pending_count, 0)

    async def test_invalid_result_is_not_committed(self):
        manager, engine, _ = self.manager()
        with self.assertRaises(ValueError):
            await manager.accept("a", "无效结果")
        await manager.accept("a", "重试")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "重试"}])

    async def test_wire_invalid_meme_or_oversized_reply_is_rejected_before_history_commit(self):
        class InvalidWireEngine(RecordingEngine):
            def reply(self, messages):
                result = super().reply(messages)
                if messages[-1]["content"] == "非法编号":
                    result["meme_id"] = "not-a-meme"
                if messages[-1]["content"] == "超长回复":
                    result["reply"] = "字" * 16001
                return result
        manager, engine, _ = self.manager(InvalidWireEngine())
        for text in ("非法编号", "超长回复"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                await manager.accept("a", text)
            self.assertEqual(manager.history("a"), [])
        await manager.accept("a", "正常")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "正常"}])

    async def test_per_device_busy_rejection_does_not_extend_deadline_or_mutate_history(self):
        engine = RecordingEngine(block_text="慢问题")
        manager, _, clock = self.manager(engine, max_pending_per_session=1)
        await manager.accept("a", "原历史")
        clock.set(1)
        pending = manager.accept("a", "慢问题")
        await wait_event(engine.started)
        clock.set(9.5)
        before = manager.history("a")
        with self.assertRaises(BusyError):
            manager.accept("a", "拒绝的新问题")
        self.assertEqual(manager.history("a"), before)
        self.assertEqual(manager.pending_count, 1)
        clock.set(11)
        engine.release.set()
        await pending
        self.assertEqual(manager.history("a"), [])
        self.assertEqual(len(engine.calls), 2)

    async def test_global_pending_limit_rejects_new_device_without_creating_context(self):
        engine = RecordingEngine(block_text="占用GPU")
        manager, _, _ = self.manager(engine, max_pending_total=2)
        one = manager.accept("a", "占用GPU")
        await wait_event(engine.started)
        two = manager.accept("b", "等待GPU")
        with self.assertRaises(BusyError):
            manager.accept("c", "被拒绝")
        self.assertEqual(manager.pending_count, 2)
        self.assertEqual(manager.history("c"), [])
        engine.release.set()
        await asyncio.gather(one, two)
        self.assertEqual(manager.pending_count, 0)
        self.assertEqual(len(engine.calls), 2)

    async def test_future_cancellation_does_not_retract_accepted_context(self):
        engine = RecordingEngine(block_text="第一句")
        manager, _, clock = self.manager(engine)
        first = manager.accept("a", "第一句")
        await wait_event(engine.started)
        first.cancel()
        clock.set(1)
        second = manager.accept("a", "第二句")
        engine.release.set()
        await second
        self.assertTrue(first.cancelled())
        self.assertEqual([m["content"] for m in engine.calls[1]], ["第一句", "回答：第一句", "第二句"])

    async def test_disconnected_unawaited_failure_does_not_emit_future_warning(self):
        manager, _, _ = self.manager()
        loop = asyncio.get_running_loop()
        events = []
        prior = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, event: events.append(event))
        try:
            ignored = manager.accept("a", "失败")
            while manager.pending_count:
                await asyncio.sleep(.001)
            await asyncio.sleep(0)
            del ignored
            gc.collect()
            await asyncio.sleep(0)
            self.assertFalse([e for e in events if "exception was never retrieved" in e.get("message", "")])
        finally:
            loop.set_exception_handler(prior)

    async def test_close_cancels_queued_and_running_futures_without_waiting_for_thread(self):
        engine = RecordingEngine(block_text="慢问题")
        manager, _, _ = self.manager(engine)
        first = manager.accept("a", "慢问题")
        await wait_event(engine.started)
        second = manager.accept("a", "不应运行")
        await asyncio.wait_for(manager.close(), timeout=.25)
        self.assertFalse(engine.release.is_set())
        self.assertTrue(first.cancelled() and second.cancelled())
        self.assertEqual(manager.pending_count, 0)
        self.assertEqual(manager.history("a"), [])
        engine.release.set()
        await wait_event(engine.exited)
        await asyncio.sleep(0)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(manager.history("a"), [])
        await manager.close()

    async def test_background_cleanup_expires_without_next_question(self):
        manager, _, clock = self.manager(cleanup_interval=.005)
        await manager.accept("a", "旧对话")
        clock.set(10)
        async def wait_release():
            while manager._sessions:
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait_release(), 1)
        self.assertEqual(manager.history("a"), [])

    async def test_history_and_engine_mutation_cannot_change_committed_state(self):
        class MutatingEngine(RecordingEngine):
            def reply(self, messages):
                result = super().reply(messages)
                messages[0]["content"] = "篡改引擎输入"
                return result
        manager, _, _ = self.manager(MutatingEngine())
        await manager.accept("a", "真实原句")
        snapshot = manager.history("a")
        snapshot[0]["content"] = "篡改读回副本"
        self.assertEqual(manager.history("a")[0]["content"], "真实原句")

    async def test_turn_result_is_immutable_and_exposes_only_two_public_fields(self):
        turn = TurnResult("归一化用户句", "最终回复", "meme-0008")
        with self.assertRaises(FrozenInstanceError):
            turn.user_text = "不能改写"
        public = turn.as_reply()
        self.assertEqual(public, {"reply": "最终回复", "meme_id": "meme-0008"})
        public["reply"] = "修改读回字典"
        self.assertEqual(turn.reply, "最终回复")

    async def test_turn_aware_engine_commits_normalized_text_and_next_turn_sees_complete_history(self):
        class PreparedEngine(RecordingEngine):
            def reply(self, messages):
                raise AssertionError("reply_turn must be preferred without a second reply call")

            def reply_turn(self, messages):
                self.calls.append(copy.deepcopy(messages))
                raw = messages[-1]["content"]
                corrected = {"STT错句一": "正确问题一", "STT错句二": "正确问题二"}[raw]
                # A normalizer's private prompt/answer must not become turns.
                return TurnResult(corrected, "回答：" + corrected, "meme-0008")

        manager, engine, clock = self.manager(PreparedEngine())
        first = await manager.accept("a", "STT错句一")
        clock.set(5)
        second = await manager.accept("a", "STT错句二")
        self.assertEqual(first, {"reply": "回答：正确问题一", "meme_id": "meme-0008"})
        self.assertEqual(second, {"reply": "回答：正确问题二", "meme_id": "meme-0008"})
        self.assertEqual(engine.calls[1], [
            {"role": "user", "content": "正确问题一"},
            {"role": "assistant", "content": "回答：正确问题一"},
            {"role": "user", "content": "STT错句二"},
        ])
        self.assertEqual(manager.history("a"), [
            {"role": "user", "content": "正确问题一"},
            {"role": "assistant", "content": "回答：正确问题一"},
            {"role": "user", "content": "正确问题二"},
            {"role": "assistant", "content": "回答：正确问题二"},
        ])
        self.assertEqual(len(engine.calls), 2)

    async def test_normalization_spanning_deadline_keeps_queued_history_without_reviving_current(self):
        class SlowPreparedEngine(RecordingEngine):
            def reply_turn(self, messages):
                self.calls.append(copy.deepcopy(messages))
                raw = messages[-1]["content"]
                if raw == "STT错句一":
                    self.started.set()
                    if not self.release.wait(2):
                        raise TimeoutError("test did not release preparation")
                corrected = {"STT错句一": "正确一", "STT错句二": "正确二", "新段错句": "新段正确句"}[raw]
                return TurnResult(corrected, "答：" + corrected, "meme-0008")

        manager, engine, clock = self.manager(SlowPreparedEngine())
        first = manager.accept("a", "STT错句一")
        await wait_event(engine.started)
        clock.set(5)
        second = manager.accept("a", "STT错句二")
        clock.set(16)
        third = manager.accept("a", "新段错句")
        self.assertEqual(manager.pending_count, 3)
        engine.release.set()
        results = await asyncio.gather(first, second, third)
        self.assertTrue(all(set(result) == {"reply", "meme_id"} for result in results))
        self.assertEqual([len(call) for call in engine.calls], [1, 3, 1])
        self.assertEqual([m["content"] for m in engine.calls[1]], ["正确一", "答：正确一", "STT错句二"])
        self.assertEqual(manager.history("a"), [
            {"role": "user", "content": "新段正确句"},
            {"role": "assistant", "content": "答：新段正确句"},
        ])

    async def test_slow_prepared_turn_alone_cannot_extend_arrival_deadline(self):
        class SlowPreparedEngine(RecordingEngine):
            def reply_turn(self, messages):
                self.started.set()
                if not self.release.wait(2):
                    raise TimeoutError("test did not release preparation")
                return TurnResult("更正后的原问题", "最终答案", "meme-0008")

        manager, engine, clock = self.manager(SlowPreparedEngine())
        pending = manager.accept("a", "识别原文")
        await wait_event(engine.started)
        clock.set(11)
        engine.release.set()
        self.assertEqual(await pending, {"reply": "最终答案", "meme_id": "meme-0008"})
        self.assertEqual(manager.history("a"), [])

    async def test_invalid_turn_result_never_commits_prepared_user_or_half_turn(self):
        bad_results = [
            {"user_text": "更正句", "reply": "回答", "meme_id": "meme-0008"},
            None,
            TurnResult(" ", "回答", "meme-0008"),
            TurnResult("字" * 16001, "回答", "meme-0008"),
            TurnResult("更正句", "", "meme-0008"),
            TurnResult("更正句", "回答", "meme-invalid"),
        ]

        class InvalidPreparedEngine(RecordingEngine):
            def reply_turn(self, messages):
                self.calls.append(copy.deepcopy(messages))
                if messages[-1]["content"] == "最后成功":
                    return TurnResult("最后正确句", "有效回答", "meme-0008")
                return bad_results.pop(0)

        manager, engine, _ = self.manager(InvalidPreparedEngine())
        for index in range(len(bad_results)):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                await manager.accept("a", f"非法结果输入{index}")
            self.assertEqual(manager.history("a"), [])
            self.assertEqual(manager.pending_count, 0)
        await manager.accept("a", "最后成功")
        self.assertEqual(engine.calls[-1], [{"role": "user", "content": "最后成功"}])
        self.assertEqual(len(manager.history("a")), 2)

    async def test_prepared_engine_mutation_cannot_replace_returned_effective_user_text(self):
        class MutatingPreparedEngine(RecordingEngine):
            def reply_turn(self, messages):
                self.calls.append(copy.deepcopy(messages))
                raw = messages[-1]["content"]
                normalized = "正确：" + raw
                result = TurnResult(normalized, "回答：" + normalized, "meme-0008")
                messages[-1]["content"] = "引擎篡改了输入"
                messages.insert(0, {"role": "system", "content": "内部纠错指令"})
                return result

        manager, engine, _ = self.manager(MutatingPreparedEngine())
        await manager.accept("a", "原句一")
        await manager.accept("a", "原句二")
        self.assertEqual(engine.calls[1], [
            {"role": "user", "content": "正确：原句一"},
            {"role": "assistant", "content": "回答：正确：原句一"},
            {"role": "user", "content": "原句二"},
        ])
        self.assertEqual([m["content"] for m in manager.history("a")], ["正确：原句一", "回答：正确：原句一", "正确：原句二", "回答：正确：原句二"])

    async def test_turn_only_engine_needs_no_legacy_reply_method(self):
        class TurnOnlyEngine:
            release = threading.Event()

            def reply_turn(self, messages):
                return TurnResult(messages[-1]["content"], "唯一最终回复", "meme-0008")

        manager, _, _ = self.manager(TurnOnlyEngine())
        self.assertEqual(await manager.accept("a", "原问题"), {"reply": "唯一最终回复", "meme_id": "meme-0008"})
        self.assertEqual([m["content"] for m in manager.history("a")], ["原问题", "唯一最终回复"])

    async def test_borrowed_multiworker_executor_still_serializes_engine_and_stays_owned_by_caller(self):
        engine = RecordingEngine(block_text="占用GPU")
        executor = ThreadPoolExecutor(max_workers=2)
        try:
            manager, _, _ = self.manager(engine, executor=executor)
            first = manager.accept("a", "占用GPU")
            await wait_event(engine.started)
            second = manager.accept("b", "另设备")
            await asyncio.sleep(.01)
            self.assertEqual(len(engine.calls), 1)
            engine.release.set()
            await asyncio.gather(first, second)
            self.assertEqual(engine.max_active, 1)
            await manager.close()
            self.assertEqual(await asyncio.wrap_future(executor.submit(lambda: 7)), 7)
        finally:
            engine.release.set()
            executor.shutdown(wait=False, cancel_futures=True)

    async def test_invalid_admission_does_not_create_pending_work(self):
        manager, engine, _ = self.manager()
        for session_id, text in [("", "问题"), (" ", "问题"), (None, "问题"), ("a", ""), ("a", "  "), ("a", None), ("a", 7)]:
            with self.subTest(session_id=session_id, text=text), self.assertRaises(ValueError):
                manager.accept(session_id, text)
        self.assertEqual(manager.pending_count, 0)
        self.assertEqual(engine.calls, [])

    async def test_invalid_settings_are_rejected(self):
        for keyword, value in [("idle_timeout", 0), ("idle_timeout", float("nan")), ("idle_timeout", True),
                               ("cleanup_interval", -1), ("max_pending_per_session", 0), ("max_pending_total", 1.5), ("max_pending_total", True),
                               ("max_turns", 0), ("max_turns", -1), ("max_turns", 1.5), ("max_turns", True), ("max_turns", False)]:
            with self.subTest(keyword=keyword, value=value), self.assertRaises(ValueError):
                DialogueManager(RecordingEngine(), **{keyword: value})


class IdleActivityTests(unittest.IsolatedAsyncioTestCase):
    """Activity leases supplement, rather than alter, legacy arrival timing."""

    asyncSetUp = DialogueManagerTests.asyncSetUp
    asyncTearDown = DialogueManagerTests.asyncTearDown
    manager = DialogueManagerTests.manager

    async def test_first_speech_hold_survives_empty_session_cleanup(self):
        manager, _, clock = self.manager()
        manager.hold_idle("a", 30)
        self.assertEqual(manager.history("a"), [])
        self.assertIn("a", manager._sessions)
        clock.set(20)
        await manager.accept("a", "完整第一句")
        manager.hold_idle("a")
        clock.set(1000)
        self.assertEqual(len(manager.history("a")), 2)
        self.assertTrue(manager.resume_idle("a"))
        clock.set(1009)
        self.assertEqual(len(manager.history("a")), 2)
        clock.set(1010)
        self.assertEqual(manager.history("a"), [])

    async def test_late_lease_cleanup_counts_from_deadline_not_cleanup_time(self):
        manager, _, clock = self.manager()
        await manager.accept("a", "首句")
        clock.set(9)
        manager.hold_idle("a", 30)
        clock.set(50)  # Lease ended at 39; waiting was already idle for 11s.
        self.assertFalse(manager.resume_idle("a"))
        self.assertEqual(manager.history("a"), [])

    async def test_manual_resume_is_idempotent_and_reset_clears_hold(self):
        manager, _, clock = self.manager()
        await manager.accept("a", "首句")
        manager.hold_idle("a")
        clock.set(100)
        self.assertTrue(manager.resume_idle("a"))
        clock.set(109)
        self.assertFalse(manager.resume_idle("a"))
        clock.set(110)
        self.assertEqual(manager.history("a"), [])
        manager.hold_idle("a")
        self.assertTrue(manager.reset("a"))
        self.assertNotIn("a", manager._sessions)

    async def test_hold_after_existing_idle_deadline_starts_new_conversation(self):
        manager, engine, clock = self.manager()
        await manager.accept("a", "旧首句")
        clock.set(10)
        manager.hold_idle("a", 30)
        clock.set(25)
        await manager.accept("a", "新首句")
        self.assertEqual(engine.calls[-1], [{"role":"user", "content":"新首句"}])

    async def test_invalid_leases_do_not_create_or_extend_session(self):
        manager, _, clock = self.manager()
        for lease in (True, False, 0, -1, float("inf"), float("nan"), "30"):
            with self.subTest(lease=lease), self.assertRaises(ValueError):
                manager.hold_idle("a", lease)
        self.assertNotIn("a", manager._sessions)
        self.assertFalse(manager.resume_idle("missing"))
        await manager.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            manager.hold_idle("a", 30)


if __name__ == "__main__":
    unittest.main()
