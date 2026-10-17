"""Direct-response instructions and essential ASR clarifications."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from model_information.normalization import parse_repair
from model_information.tests.test_chat_engine import FINAL_REPLY, editor_result, make_engine


class ReplyStyleTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("model_information.chat_engine.log")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_emotion_without_essential_object_gets_one_direct_reply(self):
        original = "他又那样了，我真烦。"
        unnecessary = "他具体做了什么呀？"
        parsed = parse_repair(editor_result(original, "uncertain", unnecessary), original)
        self.assertEqual((parsed.text, parsed.status, parsed.reason, parsed.clarification),
                         (original, "unchanged", "nonessential_ambiguity", ""))
        engine, generator, planner, index, _ = make_engine([editor_result(original, "uncertain", unnecessary), FINAL_REPLY])
        turn = engine.reply_turn([{"role": "user", "content": original}])
        self.assertEqual(turn.user_text, original)
        self.assertEqual(turn.reply, FINAL_REPLY)
        self.assertNotEqual(turn.reply, unnecessary)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(planner.calls[0], ([{"role": "A", "text": original}], "B"))
        self.assertEqual(len(index.calls), 1)

    def test_essential_asr_object_clarification_is_returned_and_retrieved(self):
        original = "我很烦，帮我取消那个。"
        question = "你想取消哪一项？"
        parsed = parse_repair(editor_result(original, "uncertain", question), original)
        self.assertEqual((parsed.text, parsed.status, parsed.clarification), (original, "uncertain", question))
        engine, generator, planner, index, _ = make_engine([editor_result(original, "uncertain", question)])
        planner.realized_vectors[question] = [919., 1.]
        turn = engine.reply_turn([{"role": "user", "content": original}])
        self.assertEqual(turn.as_reply(), {"reply": question, "meme_id": "meme-0062"})
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(planner.calls, [([{"role": "A", "text": original}, {"role": "B", "text": question}], "A")])
        np.testing.assert_array_equal(index.calls[0], [919., 1.])

    def test_explicit_emotion_advice_request_needs_no_cause_clarification(self):
        original = "我很难过怎么办。"
        unnecessary = "你为什么难过呀？"
        result = parse_repair(editor_result(original, "uncertain", unnecessary), original)
        self.assertEqual((result.text, result.status, result.reason, result.clarification),
                         (original, "unchanged", "nonessential_ambiguity", ""))

if __name__ == "__main__":
    unittest.main()
