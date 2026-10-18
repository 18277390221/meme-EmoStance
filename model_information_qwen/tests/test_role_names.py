"""Deterministic name normalization, including protected counterexamples."""

from __future__ import annotations

from copy import deepcopy
import unittest

from model_information_qwen.role_names import ROLE_ALIASES, ROLE_NAME, canonicalize_role_name


class RoleNameTests(unittest.TestCase):
    def test_known_chinese_voice_spellings_are_normalized_when_addressed(self):
        self.assertEqual(ROLE_NAME, "meme")
        self.assertIsInstance(ROLE_ALIASES, tuple)
        for alias in ROLE_ALIASES:
            with self.subTest(alias=alias):
                self.assertEqual(canonicalize_role_name(alias + "，你在吗？"), "meme，你在吗？")

    def test_emotion_negation_and_unrelated_text_are_preserved_verbatim(self):
        cases = (("妹妹猫，我今天不开心。", "meme，我今天不开心。"),
                 ("美美猫，别说话了。", "meme，别说话了。"),
                 ("咪咪猫，我不是不想聊天，明天23号再聊。", "meme，我不是不想聊天，明天23号再聊。"))
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), expected)

    def test_latin_case_and_word_internal_spacing_are_canonical(self):
        for alias in ("MEME", "Meme", "me me", "m e m e", "m\te\tm\te", "MEME猫", "Meme 猫", "me me猫", "m e m e 猫", "m\te\tm\te\t猫", "ｍｅｍｅ猫"):
            with self.subTest(alias=alias):
                expected = alias + "，你好。" if alias.startswith("ｍ") else "meme，你好。"
                self.assertEqual(canonicalize_role_name(alias + "，你好。"), expected)

    def test_missing_stt_punctuation_is_supported(self):
        cases = (("咪咪猫我今天有点难过", "meme我今天有点难过"),
                 ("Meme 猫在吗", "meme在吗"),
                 ("me me猫帮我一下", "meme帮我一下"),
                 ("MEME我今天不开心", "meme我今天不开心"),
                 ("m e m e别说话了", "meme别说话了"),
                 ("米米猫能陪我吗", "meme能陪我吗"))
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), expected)

    def test_isolated_name_and_greeting_can_call_the_assistant(self):
        for original, expected in (("米米猫", "meme"), ("  咪咪猫！", "  meme！"), ("MEME", "meme"),
                                   ("你好，米米猫", "你好，meme"), ("谢谢你，美美猫。", "谢谢你，meme。")):
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), expected)

    def test_pet_names_and_pet_naming_are_not_repaired(self):
        unchanged = ("我家咪咪猫今天不吃饭。", "我想给新养的小猫取名米米猫。",
                     "她家的妹妹猫很可爱。", "我的米米猫，你今天吃饭了吗？",
                     "我家的猫叫米米猫。", "米米猫是我的猫。", "米米猫，品种是英短。", "我家迷迷猫今天不吃饭。",
                     "我家meme猫今天不吃饭。", "我家MEME今天不吃饭。", "MEME是我的猫。", "我的MEME，你吃饭了吗？")
        for original in unchanged:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), original)

    def test_human_names_user_identity_and_explicit_renaming_are_protected(self):
        unchanged = ("美美猫是我同学的网名。", "我叫美美猫。", "她叫米米猫。",
                     "我想给你改名叫美美猫。", "以后你就叫咪咪猫。", "我的网名是米米猫。", "梅梅猫是我同学的网名。",
                     "我叫MEME。", "MEME是我同学的网名。", "以后你就叫MEME。")
        for original in unchanged:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), original)

    def test_quotes_code_and_unclosed_quotes_are_literal(self):
        unchanged = ('请原样输出“MEME 猫”。', '请打印"米米猫"。', "变量名是`meme_cat`，显示文字是`米米猫`。",
                     '```python\nname = "咪咪猫"\n```', '他说：“咪咪猫，你在吗？”', '“米米猫，你在吗？',
                     '请输出“MEME”。', '```python\nname = "MEME"\n```', '`m e m e`，你在吗？')
        for original in unchanged:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), original)

    def test_each_occurrence_is_classified_separately(self):
        cases = (("咪咪猫，我家米米猫今天不吃饭。", "meme，我家米米猫今天不吃饭。"),
                 ('咪咪猫，帮我把“米米猫”写下来。', 'meme，帮我把“米米猫”写下来。'),
                 ('MEME 猫，帮我打印`Meme 猫`。', 'meme，帮我打印`Meme 猫`。'),
                 ('MEME，帮我把“meme猫”写下来。', 'meme，帮我把“meme猫”写下来。'))
        for original, expected in cases:
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), expected)

    def test_pet_name_question_changes_the_meaning_of_an_isolated_answer(self):
        history = [{"role": "user", "content": "我养了一只猫。"},
                   {"role": "assistant", "content": "你家猫叫什么名字呀？"}]
        before = deepcopy(history)
        self.assertEqual(canonicalize_role_name("米米猫", history), "米米猫")
        self.assertEqual(canonicalize_role_name("MEME 猫。", history), "MEME 猫。")
        self.assertEqual(canonicalize_role_name("M E M E", history), "M E M E")
        self.assertEqual(history, before)

    def test_person_name_answers_are_protected_but_assistant_name_guess_can_normalize(self):
        for question in ("你叫什么名字呀？", "你希望我怎么称呼你？", "你同学叫什么？"):
            history = [{"role": "assistant", "content": question}]
            with self.subTest(question=question):
                self.assertEqual(canonicalize_role_name("美美猫", history), "美美猫")
        self.assertEqual(canonicalize_role_name("米米猫", [{"role": "assistant", "content": "你知道我叫什么吗？"}]), "meme")

    def test_explicit_assistant_identity_allows_a_reference_but_not_a_pet(self):
        for name in ("meme", "meme猫", "MEME", "M E M E 猫"):
            history = [{"role": "user", "content": "你叫什么？"},
                       {"role": "assistant", "content": f"我叫{name}，可以陪你聊聊。"}]
            with self.subTest(name=name):
                self.assertEqual(canonicalize_role_name("我想和米米猫聊聊。", history), "我想和meme聊聊。")
                self.assertEqual(canonicalize_role_name("我家米米猫今天不吃饭。", history), "我家米米猫今天不吃饭。")
        self.assertEqual(canonicalize_role_name("我想和米米猫聊聊。"), "我想和米米猫聊聊。")

    def test_explicit_you_name_reference_is_allowed_without_rewriting_other_entities(self):
        for original, expected in (("你叫米米猫对吗？", "你叫meme对吗？"),
                                   ("你的名字是美美猫吧？", "你的名字是meme吧？"),
                                   ("我可以叫你咪咪猫吗？", "我可以叫你meme吗？"),
                                   ("你叫MEME对吗？", "你叫meme对吗？")):
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), expected)

    def test_dataset_style_history_is_supported_without_system_context(self):
        history = [{"role": "A", "text": "你是谁？"}, {"role": "B", "text": "我是meme。"}]
        self.assertEqual(canonicalize_role_name("我想和美美猫聊聊。", history), "我想和meme聊聊。")
        system = [{"role": "system", "content": "我叫meme。"}]
        self.assertEqual(canonicalize_role_name("我想和美美猫聊聊。", system), "我想和美美猫聊聊。")

    def test_arbitrary_cats_or_embedded_latin_identifiers_are_not_names(self):
        for original in ("那只猫很可爱。", "喵喵猫你好。", "小咪你好。", "猫猫陪我。", "xmeme猫你好", "meme猫_id你好",
                         "xMEME你好", "MEME_id你好", "MEME猫_id你好", "M E M E 猫_id你好", "MEME123你好"):
            with self.subTest(original=original):
                self.assertEqual(canonicalize_role_name(original), original)

    def test_reported_speech_without_quotes_is_not_treated_as_a_call(self):
        original = "她说，咪咪猫，你在吗？"
        self.assertEqual(canonicalize_role_name(original), original)

    def test_canonicalization_is_idempotent_and_empty_input_is_valid(self):
        original = "妹妹猫，我今天不开心。"
        normalized = canonicalize_role_name(original)
        self.assertEqual(canonicalize_role_name(normalized), normalized)
        self.assertEqual(canonicalize_role_name(""), "")
        self.assertEqual(canonicalize_role_name("普通文本", history="not a history"), "普通文本")


if __name__ == "__main__":
    unittest.main()
