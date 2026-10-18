"""Single-response generation and final-reply emotion retrieval tests."""
from __future__ import annotations

from copy import deepcopy
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from model_information_qwen.chat_engine import EmotionalChatEngine, load_chat_settings
from model_information_qwen.sessions import DialogueManager
from model_information_qwen.turns import TurnResult

ORIGINAL = "我今天不开心，鸭力很大。"
CORRECTED = "我今天不开心，压力很大。"
FINAL_REPLY = "今天一直绷着，确实不好受。先缓一缓，meme在这里陪你。"


def editor_result(text=CORRECTED, status="corrected", clarification=""):
    return json.dumps({"status": status, "text": text, "clarification": clarification}, ensure_ascii=False)


class FakeGenerator:
    def __init__(self, outputs, events):
        self.outputs = list(outputs)
        self.calls = []
        self.events = events

    def complete(self, system, messages, **options):
        self.calls.append((system, deepcopy(messages), deepcopy(options)))
        self.events.append(("generate", len(self.calls)))
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakePlanner:
    def __init__(self, events):
        self.calls = []
        self.events = events
        self.target_vector = np.array([.25, .75])
        self.realized_vectors = {}

    def predict(self, history, next_role):
        self.calls.append((deepcopy(history), next_role))
        self.events.append(("planner", next_role))
        if next_role == "B":
            return {"target_vector": self.target_vector.copy()}
        reply = history[-1]["text"]
        return {"source_realized_vector": np.asarray(self.realized_vectors.get(reply, [909., 1.])).copy()}


class FakeIndex:
    def __init__(self, events):
        self.calls = []
        self.events = events

    def nearest(self, vector):
        self.calls.append(np.asarray(vector).copy())
        self.events.append(("nearest", float(vector[0])))
        return {"meme_id": "meme-0062"}


def make_engine(outputs=None, **overrides):
    events = []
    planner, index = FakePlanner(events), FakeIndex(events)
    raw = SimpleNamespace(config={"archive": "fake"}, planner=planner, index=index,
                          generator=object(), _lock=threading.Lock())
    raw.reply = lambda *_: (_ for _ in ()).throw(AssertionError("raw reply must not be called"))
    settings = load_chat_settings()
    # Keep explicit coverage of optional repair independently of production's
    # fast default; fast-path tests below pass the actual deployed setting.
    settings["asr_enabled"] = True
    settings.update(overrides)
    generator = FakeGenerator(outputs if outputs is not None else [editor_result(), FINAL_REPLY], events)
    engine = EmotionalChatEngine(raw, settings, generator=generator)
    return engine, generator, planner, index, events


class ChatEngineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("model_information_qwen.chat_engine.log")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_role_prompt_and_stance_are_sent_only_to_one_reply_generation(self):
        engine, generator, planner, _, _ = make_engine()
        turn = engine.reply_turn([{"role": "user", "content": ORIGINAL}])
        self.assertIsInstance(turn, TurnResult)
        self.assertIn("meme", engine.settings["role_prompt"])
        self.assertNotIn("num_candidates", engine.settings)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(generator.calls[0][0], engine.settings["asr_prompt"])
        self.assertFalse(generator.calls[0][2]["do_sample"])
        self.assertNotIn("stance_vector", generator.calls[0][2])
        system, messages, options = generator.calls[1]
        self.assertEqual(system, engine.settings["role_prompt"])
        self.assertEqual(messages, [{"role": "user", "content": CORRECTED}])
        np.testing.assert_array_equal(options["stance_vector"], planner.target_vector)
        self.assertEqual(options["max_new_tokens"], engine.settings["max_reply_tokens"])
        self.assertTrue(options["do_sample"])

    def test_corrected_text_enters_planner_and_turn_result_without_mutating_input(self):
        engine, generator, planner, _, _ = make_engine()
        messages = [{"role": "user", "content": "昨天发生了些事。"},
                    {"role": "assistant", "content": "先缓一缓吧。"},
                    {"role": "user", "content": ORIGINAL}]
        before = deepcopy(messages)
        turn = engine.reply_turn(messages)
        self.assertEqual(turn.user_text, CORRECTED)
        self.assertEqual(messages, before)
        expected = [{"role": "A", "text": messages[0]["content"]},
                    {"role": "B", "text": messages[1]["content"]}, {"role": "A", "text": CORRECTED}]
        self.assertEqual(planner.calls[0], (expected, "B"))
        self.assertEqual(planner.calls[1], (expected + [{"role": "B", "text": FINAL_REPLY}], "A"))
        self.assertEqual(json.loads(generator.calls[0][1][0]["content"])["text"], ORIGINAL)

    def test_single_final_reply_vector_drives_one_lookup_not_target_vector(self):
        engine, generator, planner, index, events = make_engine()
        planner.realized_vectors[FINAL_REPLY] = [707., 1.]
        turn = engine.reply_turn([{"role": "user", "content": ORIGINAL}])
        self.assertEqual(turn.reply, FINAL_REPLY)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(index.calls), 1)
        np.testing.assert_array_equal(index.calls[0], [707., 1.])
        self.assertEqual(events, [("generate", 1), ("planner", "B"),
                                  ("generate", 2), ("planner", "A"), ("nearest", 707.)])

    def test_legacy_candidate_setting_cannot_reenable_multiple_generation(self):
        engine, generator, planner, index, _ = make_engine(num_candidates=4)
        self.assertEqual(engine.reply([{"role": "user", "content": ORIGINAL}])["reply"], FINAL_REPLY)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(len(index.calls), 1)

    def test_planner_and_index_never_receive_role_or_editor_instructions(self):
        engine, _, planner, index, _ = make_engine()
        engine.reply_turn([{"role": "user", "content": ORIGINAL}])
        for history, _ in planner.calls:
            for message in history:
                self.assertEqual(set(message), {"role", "text"})
                self.assertIn(message["role"], ("A", "B"))
                self.assertNotIn(engine.settings["role_prompt"], message["text"])
                self.assertNotIn(engine.settings["asr_prompt"], message["text"])
                self.assertNotEqual(message["text"], ORIGINAL)
        self.assertEqual(index.calls[0].shape, planner.target_vector.shape)

    def test_uncertain_returns_clarification_and_queries_same_meme_index(self):
        original = "帮我去葱一下。"
        question = "你说的‘去葱’具体想做什么呀？"
        engine, generator, planner, index, _ = make_engine([editor_result(original, "uncertain", question)])
        turn = engine.reply_turn([{"role": "user", "content": original}])
        self.assertEqual((turn.user_text, turn.reply, turn.meme_id), (original, question, "meme-0062"))
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(planner.calls, [([{"role": "A", "text": original}, {"role": "B", "text": question}], "A")])
        self.assertEqual(len(index.calls), 1)
        np.testing.assert_array_equal(index.calls[0], [909., 1.])

    def test_asr_failure_falls_back_without_losing_reply(self):
        for failure in (RuntimeError("fake ASR failure"), "not JSON"):
            with self.subTest(failure=failure):
                engine, generator, planner, _, _ = make_engine([failure, FINAL_REPLY])
                turn = engine.reply_turn([{"role": "user", "content": ORIGINAL}])
                self.assertEqual(turn.user_text, ORIGINAL)
                self.assertEqual(turn.reply, FINAL_REPLY)
                self.assertEqual(len(generator.calls), 2)
                self.assertEqual(planner.calls[0][0], [{"role": "A", "text": ORIGINAL}])
                self.assertEqual(generator.calls[1][1], [{"role": "user", "content": ORIGINAL}])

    def test_disabled_asr_skips_editor_and_generates_exactly_one_reply(self):
        engine, generator, planner, _, _ = make_engine([FINAL_REPLY], asr_enabled=False)
        turn = engine.reply_turn([{"role": "user", "content": ORIGINAL}])
        self.assertEqual(turn.user_text, ORIGINAL)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(planner.calls[0][0][-1]["text"], ORIGINAL)
        self.assertEqual(generator.calls[0][0], engine.settings["role_prompt"])

    def test_fast_default_keeps_name_rules_raw_typo_and_one_model_generation(self):
        settings = load_chat_settings()
        self.assertFalse(settings["asr_enabled"])
        original = "米米猫，我今天不开心，鸭力很大。"
        effective = "meme，我今天不开心，鸭力很大。"
        engine, generator, planner, index, _ = make_engine([FINAL_REPLY], asr_enabled=settings["asr_enabled"])
        result = engine.reply_turn([{"role": "user", "content": original}])
        self.assertEqual(result.user_text, effective)
        self.assertEqual(result.reply, FINAL_REPLY)
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(generator.calls[0][1], [{"role": "user", "content": effective}])
        self.assertEqual(planner.calls[0], ([{"role": "A", "text": effective}], "B"))
        self.assertEqual(len(planner.calls), 2)
        self.assertEqual(len(index.calls), 1)

    def test_fast_clarification_is_one_reply_generation_and_uses_its_own_meme_vector(self):
        question = "你能换个说法吗？"
        original = "帮我去葱一下。"
        engine, generator, planner, index, _ = make_engine([question], asr_enabled=False)
        planner.realized_vectors[question] = [404., 1.]
        result = engine.reply_turn([{"role": "user", "content": original}])
        self.assertEqual(result.user_text, original)
        self.assertEqual(result.as_reply(), {"reply": question, "meme_id": "meme-0062"})
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(planner.calls[-1], ([{"role": "A", "text": original}, {"role": "B", "text": question}], "A"))
        np.testing.assert_array_equal(index.calls[0], [404., 1.])

    def test_deterministic_configuration_also_generates_only_one_reply(self):
        engine, generator, _, index, _ = make_engine(do_sample=False)
        result = engine.reply([{"role": "user", "content": ORIGINAL}])
        self.assertEqual(result["reply"], FINAL_REPLY)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(index.calls), 1)
        self.assertFalse(generator.calls[1][2]["do_sample"])

    def test_public_reply_contains_only_reply_and_meme_id(self):
        engine, _, _, _, _ = make_engine()
        result = engine.reply([{"role": "user", "content": ORIGINAL}])
        self.assertEqual(result, {"reply": FINAL_REPLY, "meme_id": "meme-0062"})
        self.assertEqual(set(result), {"reply", "meme_id"})

    def test_question_or_style_violation_does_not_trigger_another_generation(self):
        for generated in ("你愿意说说吗？", "你还是要坚强哦。", "我以前也经历过这些。"):
            with self.subTest(generated=generated):
                engine, generator, planner, index, _ = make_engine([editor_result(), generated])
                planner.realized_vectors[generated] = [555., 1.]
                result = engine.reply([{"role": "user", "content": ORIGINAL}])
                self.assertEqual(result["reply"], generated)
                self.assertEqual(len(generator.calls), 2)
                self.assertEqual(len(planner.calls), 2)
                self.assertEqual(len(index.calls), 1)
                np.testing.assert_array_equal(index.calls[0], [555., 1.])

    def test_reply_generation_failure_is_not_retried_or_retrieved(self):
        engine, generator, planner, index, _ = make_engine([editor_result(), RuntimeError("generation failed")])
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            engine.reply_turn([{"role": "user", "content": ORIGINAL}])
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(planner.calls), 1)
        self.assertEqual(index.calls, [])

    def test_system_or_wrong_order_history_is_rejected_before_generation(self):
        invalid = ([], [{"role": "system", "content": "bad"}],
                   [{"role": "assistant", "content": "bad"}],
                   [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        for messages in invalid:
            with self.subTest(messages=messages):
                engine, generator, _, _, _ = make_engine()
                with self.assertRaises(ValueError):
                    engine.reply_turn(messages)
                self.assertEqual(generator.calls, [])

    def test_player_reply_limit_is_enforced(self):
        engine, _, _, _, _ = make_engine([editor_result(), "字" * 2001])
        with self.assertRaisesRegex(RuntimeError, "2000"):
            engine.reply_turn([{"role": "user", "content": ORIGINAL}])


class CorrectedHistoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fast_history_commits_name_only_and_passes_it_to_next_reply(self):
        original = "米米猫，我今天不开心，鸭力很大。"
        effective = "meme，我今天不开心，鸭力很大。"
        second_reply = "meme陪着你。"
        engine, generator, _, _, _ = make_engine([FINAL_REPLY, second_reply], asr_enabled=False)
        manager = DialogueManager(engine, clock=lambda: 0., cleanup_interval=None)
        manager.start()
        try:
            with patch("model_information_qwen.chat_engine.log"):
                await manager.accept("fast", original)
                await manager.accept("fast", "谢谢")
            expected = [{"role": "user", "content": effective}, {"role": "assistant", "content": FINAL_REPLY},
                        {"role": "user", "content": "谢谢"}, {"role": "assistant", "content": second_reply}]
            self.assertEqual(manager.history("fast"), expected)
            self.assertEqual(generator.calls[1][1], expected[:3])
            self.assertEqual(len(generator.calls), 2)
        finally:
            await manager.close()

    async def test_only_completed_effective_user_and_final_reply_enter_history(self):
        second_reply = "不客气，meme陪着你。"
        outputs = [editor_result(), FINAL_REPLY, editor_result("谢谢", "unchanged"), second_reply]
        engine, generator, planner, index, _ = make_engine(outputs)
        manager = DialogueManager(engine, clock=lambda: 0., cleanup_interval=None)
        manager.start()
        try:
            with patch("model_information_qwen.chat_engine.log"):
                first = await manager.accept("one", ORIGINAL)
                second = await manager.accept("one", "谢谢")
            self.assertEqual(first, {"reply": FINAL_REPLY, "meme_id": "meme-0062"})
            self.assertEqual(second, {"reply": second_reply, "meme_id": "meme-0062"})
            expected = [{"role": "user", "content": CORRECTED}, {"role": "assistant", "content": FINAL_REPLY},
                        {"role": "user", "content": "谢谢"}, {"role": "assistant", "content": second_reply}]
            self.assertEqual(manager.history("one"), expected)
            second_editor = json.loads(generator.calls[2][1][0]["content"])
            self.assertEqual(second_editor["history"], expected[:2])
            self.assertEqual(generator.calls[3][1], expected[:3])
            self.assertEqual(planner.calls[2][0], [{"role": "A", "text": CORRECTED},
                                                 {"role": "B", "text": FINAL_REPLY}, {"role": "A", "text": "谢谢"}])
            self.assertEqual(len(generator.calls), 4)
            self.assertEqual(len(index.calls), 2)
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
