"""Model-free rollout auditing with unlimited and repeated-reply contexts."""
import unittest

from model_information_qwen.server import DEFAULT_MAX_TURNS
from model_information_qwen.tools.verify_dialogue_rollout import expected_rollout_layout, validate_rollout_layout


class RolloutLayoutTests(unittest.TestCase):
    def test_unlimited_ten_unique_replies_keep_all_successful_history(self):
        turns = [{"reply": f"回答{index}", "input_messages": 2*index+1, "conversation_id": "same", "background_revision":0}
                 for index in range(10)]
        layout = validate_rollout_layout(turns, None)
        self.assertEqual(layout["input_messages"], list(range(1, 20, 2)))
        self.assertEqual(layout["conversation_groups"], [0]*10)

    def test_identical_replies_reset_the_following_turn_without_a_count_cap(self):
        replies = ["相同", "相同", "相同", "不同", "不同", "下一段"]
        lengths = [1, 3, 1, 3, 5, 1]
        ids = ["a", "a", "b", "b", "b", "c"]
        turns = [{"reply": reply, "input_messages": length, "conversation_id": identifier, "background_revision":0}
                 for reply, length, identifier in zip(replies, lengths, ids)]
        layout = validate_rollout_layout(turns)
        self.assertEqual(layout["reset_reasons"], [["initial"], [], ["repeated_reply"], [], [], ["repeated_reply"]])

    def test_explicit_six_pair_window_slides_without_resetting_conversation(self):
        turns = [{"reply": str(index), "input_messages": length,
                  "conversation_id": "a", "background_revision":0}
                 for index, length in enumerate([1, 3, 5, 7, 9, 11, 13, 13])]
        layout = validate_rollout_layout(turns, 6)
        self.assertEqual(layout["reset_reasons"][6], [])
        self.assertEqual(layout["window_slides"], [0]*6+[1,2])

    def test_repeat_protection_still_resets_a_full_window(self):
        layout = expected_rollout_layout([{"reply": value} for value in ["同", "同", "新"]], 2)
        self.assertEqual(layout["input_messages"], [1, 3, 1])
        self.assertEqual(layout["reset_reasons"][2], ["repeated_reply"])

    def test_default_window_advances_background_on_successful_turns_nine_fourteen_nineteen(self):
        self.assertEqual(DEFAULT_MAX_TURNS, 4)
        turns = [{"reply":str(index), "input_messages":2*min(index,4)+1,
                  "conversation_id":"a", "background_revision":max(0,index+1-4)//5}
                 for index in range(20)]
        layout = validate_rollout_layout(turns, DEFAULT_MAX_TURNS)
        self.assertEqual(layout["background_revisions"], [0]*8+[1]*5+[2]*5+[3]*2)
        self.assertEqual(layout["conversation_groups"], [0]*20)

    def test_explicit_five_pair_window_advances_background_on_turns_ten_fifteen_twenty(self):
        turns = [{"reply":str(index), "input_messages":2*min(index,5)+1,
                  "conversation_id":"a", "background_revision":max(0,index+1-5)//5}
                 for index in range(20)]
        layout = validate_rollout_layout(turns, 5)
        self.assertEqual(layout["background_revisions"], [0]*9+[1]*5+[2]*5+[3])
        self.assertEqual(layout["conversation_groups"], [0]*20)

    def test_one_pair_window_preserves_repeat_detection_after_dropping_old_pair(self):
        layout = expected_rollout_layout([{"reply":value} for value in ["相同","相同","新"]],1)
        self.assertEqual(layout["input_messages"], [1,3,1])
        self.assertEqual(layout["window_slides"], [0,1,0])

    def test_unexpected_six_turn_reset_and_missing_repeat_reset_are_rejected(self):
        with self.assertRaisesRegex(AssertionError, "history lengths"):
            validate_rollout_layout([{"reply": str(index), "input_messages": length, "conversation_id": "a", "background_revision":0}
                for index, length in enumerate([1, 3, 5, 7, 9, 11, 1])])
        with self.assertRaisesRegex(AssertionError, "new conversation"):
            validate_rollout_layout([
                {"reply": "同", "input_messages": 1, "conversation_id": "a", "background_revision":0},
                {"reply": "同", "input_messages": 3, "conversation_id": "a", "background_revision":0},
                {"reply": "新", "input_messages": 1, "conversation_id": "a", "background_revision":0},
            ])


if __name__ == "__main__":
    unittest.main()
