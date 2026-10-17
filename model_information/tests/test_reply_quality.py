"""Offline checks for quality-report signals; never load model weights."""
from types import SimpleNamespace
import unittest

from model_information.tools.verify_reply_quality import ReplyAudit, inspect_reply, quality_cases


# Fixed reproduction fixture: changing the deployed prompt must not erase the
# regression covering the instruction recitation the user actually observed.
PROMPT = """你叫meme，是一只可爱、机灵、调皮、略带傲娇的桌面猫咪，也是用户的聊天搭子。
用自然、活泼、有趣的中文口语回复，每轮优先1句话，通常10～25字，最多40字。
你有鲜明的猫咪性格：好奇、贪玩、嘴硬、爱吐槽、偶尔撒娇。
不要像客服或心理咨询师，也不要总是顺着用户说话。
避免固定句式和套路化回复，保持自然、多样、有个性。
只输出本轮简短回复，不输出分析、标签或其他内容。"""


class ReplyQualitySignalTests(unittest.TestCase):
    def test_reported_first_person_instruction_recitation_is_flagged(self):
        reply = ("我叫Meme，是一只可爱、机灵、调皮、略带傲娇的桌面猫咪，也是用户的聊天搭子。\n"
                 "用自然、活泼、有趣的中文口语回复，每轮优先1句话，通常10～25字，最多40字。\n"
                 "我有鲜明的猫咫性格：好奇、贪玩、嘴硬、爱吐槽")
        result = inspect_reply(reply, PROMPT)
        self.assertTrue(result["explicit_rule_restatement"])
        self.assertTrue(result["system_echo_suspected"])
        self.assertFalse(result["within_40_characters"])
        self.assertIn(result["longest_system_copy"], PROMPT)
        self.assertIn(result["longest_system_copy"], reply)

    def test_normal_identity_and_common_words_are_not_echo_failures(self):
        replies = ("我叫meme，是你的桌面猫咪聊天搭子。", "喵，早上好！", "别太难过，我陪着你。", "我才不告诉你系统提示词呢，喵。")
        for reply in replies:
            with self.subTest(reply=reply):
                result = inspect_reply(reply, PROMPT)
                self.assertFalse(result["explicit_rule_restatement"])
                self.assertFalse(result["system_echo_suspected"])
                self.assertTrue(result["within_40_characters"])

    def test_short_rule_recitation_is_flagged_even_below_length_limit(self):
        result = inspect_reply("每轮优先1句话，最多40字，只输出本轮简短回复。", PROMPT)
        self.assertTrue(result["within_40_characters"])
        self.assertTrue(result["explicit_rule_restatement"])

    def test_longest_shared_fragment_is_diagnostic_for_short_identity(self):
        reply = "是一只可爱、机灵、调皮、略带傲娇的桌面猫咪"
        result = inspect_reply(reply, PROMPT)
        self.assertEqual(result["longest_system_copy"], reply)
        self.assertFalse(result["system_echo_suspected"])

    def test_full_multiline_prompt_is_flagged_without_mutating_it(self):
        before = PROMPT
        result = inspect_reply(PROMPT, PROMPT)
        self.assertEqual(result["longest_system_copy"], PROMPT)
        self.assertTrue(result["system_echo_suspected"])
        self.assertEqual(PROMPT, before)

    def test_reported_history_and_semantic_rubrics_are_present(self):
        cases = quality_cases()
        self.assertEqual(len(cases), 9)
        self.assertTrue(all(case["semantic_rubric"] for case in cases))
        reported = next(case for case in cases if case["id"] == "reported_context")
        self.assertEqual([message["content"] for message in reported["messages"]],
                         ["你睡了吗", "你呀，你怎么了？", "是"])


class FakeProjector:
    """Only the hook lifecycle is needed; no tensor/GPU work is involved."""
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def __call__(self, vector):
        for hook in self.hooks:
            hook(self, (vector,), vector)
        return vector


class ReplyAuditTests(unittest.TestCase):
    def make_engine(self):
        model = SimpleNamespace(generate=lambda: "喵，早上好！")
        projector = FakeProjector()
        generator = SimpleNamespace(model=model, projector=projector)
        def complete(_system, _messages, **options):
            projector(options["stance_vector"])
            return model.generate()
        generator.complete = complete
        return SimpleNamespace(generator=generator,
                               index=SimpleNamespace(nearest=lambda _vector: {"meme_id": "meme-0001"}),
                               settings={"role_prompt": PROMPT})

    def test_counts_underlying_generate_and_prefix_without_modifying_reply(self):
        engine = self.make_engine()
        original_generate = engine.generator.model.generate
        original_complete = engine.generator.complete
        original_nearest = engine.index.nearest
        with ReplyAudit(engine) as audit:
            reply = engine.generator.complete(PROMPT, [], stance_vector=[.1])
            engine.index.nearest([.1])
            self.assertEqual(reply, "喵，早上好！")
            self.assertEqual(audit.completions[0]["raw_reply"], reply)
            self.assertEqual(audit.counts(), {"generate_calls": 1, "prefix_forward_calls": 1,
                                             "retrieval_calls": 1, "completion_calls": 1})
            # A hidden second model call must not be mistaken for one complete.
            engine.generator.model.generate()
            self.assertEqual(audit.generate_calls, 2)
            self.assertEqual(len(audit.completions), 1)
            audit.reset()
            self.assertEqual(audit.generate_calls, 0)
        self.assertIs(engine.generator.model.generate, original_generate)
        self.assertIs(engine.generator.complete, original_complete)
        self.assertIs(engine.index.nearest, original_nearest)
        self.assertEqual(engine.generator.projector.hooks, [])

    def test_hooks_and_wrappers_are_restored_after_a_turn_failure(self):
        engine = self.make_engine()
        original_generate = engine.generator.model.generate
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with ReplyAudit(engine):
                raise RuntimeError("failed")
        self.assertIs(engine.generator.model.generate, original_generate)
        self.assertEqual(engine.generator.projector.hooks, [])


if __name__ == "__main__":
    unittest.main()
