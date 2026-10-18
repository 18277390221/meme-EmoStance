import unittest

from meme_huanxing import classify_command, is_sleep_command, is_wake_command
from meme_huanxing.commands import ADDRESS_ALIASES


class StandbyCommandTests(unittest.TestCase):
    def test_eight_requested_cat_and_me_phrases_have_distinct_intents(self):
        phrases = (
            ('小猫，睡觉吧','sleep'), ('小猫，睡觉。','sleep'),
            ('小猫，睡觉了。','sleep'), ('小me，睡觉吧。','sleep'),
            ('小猫，起床吧','wake'), ('小猫，起床。','wake'),
            ('小猫，起床了。','wake'), ('小me，起床吧。','wake'),
        )
        for text, expected in phrases:
            with self.subTest(text=text):
                self.assertEqual(expected, classify_command(text))
                self.assertEqual(expected == 'sleep', is_sleep_command(text))
                self.assertEqual(expected == 'wake', is_wake_command(text))

    def test_short_sleep_and_wake_commands_require_a_bounded_address(self):
        for name in (*ADDRESS_ALIASES, '小ME', '小 mE', '小ＭＥ'):
            for body, expected in (('睡觉','sleep'), ('起床','wake'), ('睡觉吧','sleep'), ('起床吧','wake')):
                for text in (f'{name}，{body}。', f'{name}，{body}', f' {name} , {body} . '):
                    with self.subTest(text=text):
                        self.assertEqual(expected, classify_command(text))

    def test_short_commands_still_reject_unaddressed_quoted_negative_conditional_and_questions(self):
        for body in ('睡觉','起床','睡觉吧','起床吧'):
            for name in ('小me','小猫'):
                for text in (body, body+'。', f'“{name}，{body}。”', f'{name}，{body}？',
                             f'{name}，不要{body}。', f'{name}，别{body}。', f'如果{name}，{body}。',
                             f'{name}，如果困了就{body}。', f'他说{name}，{body}。', f'{name}，{body}是什么意思'):
                    with self.subTest(text=text):
                        self.assertIsNone(classify_command(text))

    def test_three_exact_requested_commands_with_local_asr_variations(self):
        names = (*ADDRESS_ALIASES, '小ME', '小 Me', '小ＭＥ')
        for name in names:
            for body, expected in (('起床了','wake'), ('起床吧','wake'), ('别睡了','wake'), ('睡觉吧','sleep')):
                for text in (f'{name}，{body}。', f' {name} , {body} . ', f'{name}\t{body}！'):
                    with self.subTest(text=text):
                        self.assertEqual(expected, classify_command(text))

    def test_requested_commands_remain_conversation_when_quoted_conditional_or_questioned(self):
        for body in ('起床了', '起床吧', '别睡了', '睡觉吧'):
            for text in (f'“小me，{body}。”', f'小me，{body}？',
                         f'如果小me，{body}。', f'小me，如果我说{body}就照做。',
                         f'他说小me，{body}。'):
                with self.subTest(text=text):
                    self.assertIsNone(classify_command(text))

    def test_requested_sleep_and_installed_legacy_command(self):
        for text in (
            "小me，睡觉去吧", "去睡觉吧meme", "小me，去睡觉吧", "小me睡觉吧",
            "小me该睡觉了", "小me该去睡觉了", "小me去睡觉", "睡觉去吧，小me",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify_command(text), "sleep")
                self.assertTrue(is_sleep_command(text))
                self.assertFalse(is_wake_command(text))

    def test_requested_wake_commands_and_nearby_phrasing(self):
        for text in (
            "小me，别睡了，起床了", "小me，别睡了", "小me，起床了", "小me醒醒吧",
            "小me不要睡了", "小me别睡觉了", "小me醒过来吧", "小me起床啦",
            "小me别睡了快起床", "小me醒醒别睡了起床了", "别睡了，小me",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify_command(text), "wake")
                self.assertTrue(is_wake_command(text))
                self.assertFalse(is_sleep_command(text))

    def test_bounded_address_aliases(self):
        for name in ADDRESS_ALIASES:
            with self.subTest(name=name):
                self.assertEqual(classify_command(name + "，去睡觉吧"), "sleep")
                self.assertEqual(classify_command(name + "，别睡了"), "wake")

    def test_nfkc_case_punctuation_and_whitespace(self):
        for text in (
            " 小ＭＥ ， 睡觉 去吧 。 ", "小Me\t去睡觉吧", "ＭＥＭＥ\n该睡觉了！",
            "小 me、睡觉去吧", "嘿，小ME，睡觉去吧！", "小me啊，请你去睡觉吧。",
        ):
            with self.subTest(text=text):
                self.assertEqual(classify_command(text), "sleep")
        self.assertEqual(classify_command("小ｍｅ，别睡了；快起床了！"), "wake")

    def test_polite_and_urgent_direct_commands(self):
        for text, expected in (
            ("小me，你现在去睡觉吧", "sleep"), ("小me，请你先睡觉吧", "sleep"),
            ("喂小me，快醒醒吧", "wake"), ("小me麻烦你起床了", "wake"),
            ("小me进入待机模式", "sleep"), ("小me去休息吧", "sleep"),
        ):
            with self.subTest(text=text):
                self.assertEqual(classify_command(text), expected)

    def test_sleep_negations_do_not_activate_sleep(self):
        for text in (
            "小me别去睡觉吧", "小me不要睡觉去吧", "小me不用去睡觉", "小me不是该睡觉了",
            "小me没有去睡觉", "小me我不想让你睡觉", "别让小me去睡觉吧",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_wake_negations_do_not_activate_wake(self):
        for text in (
            "小me不要起床", "小me别醒醒吧", "小me别起床了", "小me不用醒过来",
            "小me不是说别睡了", "小me别说别睡了", "小me不要说起床了",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_quoted_commands_are_conversation(self):
        for text in (
            '"小me，睡觉去吧"', "“小me，别睡了”", "小me‘睡觉吧’", "小me「起床了」",
            "小me说‘别睡了’", "小me《睡觉吧》", "小me '起床了'",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_questions_are_conversation(self):
        for text in (
            "小me睡觉去吧？", "小me起床了?", "小me该睡觉了吗", "小me能不能别睡了",
            "小me别睡了好吗", "小me什么时候起床", "小me你睡了吗", "小me要不要睡觉",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_conditions_explanations_and_mentions_are_conversation(self):
        for text in (
            "如果小me睡觉吧", "小me如果累了就去睡觉吧", "小me要是困了去睡觉吧",
            "等下小me别睡了", "小me等我回来再起床了", "小me我说起床了", "他说小me睡觉吧",
            "小me刚才妈妈说别睡了", "小me起床了是什么意思", "小me睡觉吧这句话怎么说",
            "小me睡觉吧我还有事情", "小me别睡了然后再去睡觉吧",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_missing_or_embedded_address_does_not_match(self):
        for text in (
            "去睡觉吧", "睡觉去吧", "起床了", "别睡了", "醒醒吧", "me睡觉吧", "猫睡觉吧",
            "小猫咪睡觉吧", "小猫咪起床了", "小猫猫睡觉吧", "小猫儿起床吧", "小猫",
            "你的meme睡觉吧", "小meme2睡觉吧", "小meow别睡了", "大meme起床了",
            "小mee睡觉吧", "小me", "小米手机睡觉吧", "小美同学起床了",
        ):
            with self.subTest(text=text):
                self.assertIsNone(classify_command(text))

    def test_invalid_values_are_not_commands(self):
        for value in (None, False, 0, 1, {}, [], b"meme", "", "  \n", "，。！"):
            with self.subTest(value=value):
                self.assertIsNone(classify_command(value))
                self.assertFalse(is_sleep_command(value))
                self.assertFalse(is_wake_command(value))

    def test_state_is_a_caller_responsibility(self):
        # The rule recognizer cannot cause an awake player to run a wake
        # animation: it exposes intent only, without any side effects/state.
        text = "小me，别睡了，起床了"
        self.assertEqual(classify_command(text), "wake")
        self.assertEqual(classify_command(text), "wake")


if __name__ == "__main__":
    unittest.main()
