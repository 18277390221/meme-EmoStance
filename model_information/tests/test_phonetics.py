"""Real pypinyin + failure-injection tests; no model or GPU is loaded."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from model_information.phonetics import phonetic_edit_reason


class PhoneticEditTests(unittest.TestCase):
    def test_same_pronunciation_repairs_pass_ignoring_tone(self):
        cases = (("我今天不开心，鸭力很大。", "我今天不开心，压力很大。"),
                 ("这6张发漂一共128元。", "这6张发票一共128元。"),
                 ("鸭力", "压力"), ("发漂", "发票"))
        for original, candidate in cases:
            with self.subTest(original=original, candidate=candidate):
                self.assertEqual(phonetic_edit_reason(original, candidate), "")

    def test_model_semantic_rewrite_pressure_to_strength_is_rejected(self):
        self.assertEqual(phonetic_edit_reason("我今天不开心，鸭力很大。", "我今天不开心，力气很大。"), "phonetic_mismatch")
        self.assertTrue(phonetic_edit_reason("鸭力", "力气"))

    def test_chinese_insertions_deletions_and_reordering_cannot_fill_missing_words(self):
        for original, candidate in (("开心", "不开心"), ("鸭力", "我压力"),
                                    ("我不开心", "我开心"), ("压力", "力压"), ("鸭", "压力")):
            with self.subTest(original=original, candidate=candidate):
                self.assertTrue(phonetic_edit_reason(original, candidate))

    def test_common_single_feature_near_sounds_are_permitted(self):
        cases = (("南", "蓝"), ("蓝", "南"),
                 ("张", "脏"), ("脏", "张"),
                 ("陈", "岑"), ("岑", "陈"),
                 ("诗", "丝"), ("丝", "诗"),
                 ("真", "争"), ("争", "真"),
                 ("心", "星"), ("星", "心"))
        for original, candidate in cases:
            with self.subTest(original=original, candidate=candidate):
                self.assertEqual(phonetic_edit_reason(original, candidate), "")

    def test_unlisted_or_combined_sound_changes_are_not_accepted(self):
        # n/l plus in/ing together, ang/an, f/h, r/l, and unrelated vowels.
        for original, candidate in (("宁", "林"), ("张", "占"),
                                    ("发", "哈"), ("日", "力"), ("压", "力")):
            with self.subTest(original=original, candidate=candidate):
                self.assertTrue(phonetic_edit_reason(original, candidate))

    def test_punctuation_and_whitespace_do_not_require_dependency(self):
        with patch.dict(sys.modules, {"pypinyin": None}):
            self.assertEqual(phonetic_edit_reason("我，今天 不开心！", "我今天不开心。"), "")
            self.assertEqual(phonetic_edit_reason("hello 23", "hello 24"), "")
            self.assertEqual(phonetic_edit_reason("王莉今天不来。", "王莉今天不来。"), "")

    def test_missing_dependency_fails_closed_for_actual_chinese_edits(self):
        with patch.dict(sys.modules, {"pypinyin": None}):
            self.assertEqual(phonetic_edit_reason("鸭力", "压力"), "phonetic_dependency_unavailable")
            self.assertEqual(phonetic_edit_reason("开心", "不开心"), "phonetic_alignment_changed")

    def test_broken_dependency_lookup_fails_closed(self):
        with patch("pypinyin.lazy_pinyin", side_effect=RuntimeError("test lookup failure")):
            self.assertEqual(phonetic_edit_reason("鸭力", "压力"), "phonetic_lookup_failed")

    def test_missing_or_non_latin_pinyin_fails_closed(self):
        for output in (None, [], ["ya"], ["鸭", "li"], [None, "li"]):
            with self.subTest(output=output), patch("pypinyin.lazy_pinyin", return_value=output):
                self.assertEqual(phonetic_edit_reason("鸭力", "压力"), "phonetic_lookup_failed")

    def test_unchanged_characters_do_not_need_convertible_pinyin(self):
        with patch("pypinyin.lazy_pinyin", side_effect=[["unknown", "ya"], ["unknown", "ya"]]):
            self.assertEqual(phonetic_edit_reason("𰻞鸭", "𰻞压"), "")

    def test_invalid_arguments_return_reasons_without_throwing(self):
        for original, candidate in ((None, "压力"), ("鸭力", None), (23, "二十三")):
            with self.subTest(original=original, candidate=candidate):
                self.assertEqual(phonetic_edit_reason(original, candidate), "phonetic_invalid_text")


if __name__ == "__main__":
    unittest.main()
