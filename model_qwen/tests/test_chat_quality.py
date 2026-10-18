"""Chat-quality controls without loading a backbone or touching a GPU."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emostance_qwen.chat_quality import (JUDGE_SYSTEM_PROMPT, compare_quality,
    evaluate_chat_quality, load_cases, score_reply, semantic_judge_prompt)


def example_case():
    return dict(id="pet", category="context_tracking",
        messages=[{"role": "user", "content": "我家猫叫团子。"},
                  {"role": "assistant", "content": "团子听着就可爱。"},
                  {"role": "user", "content": "我的猫叫什么？"}],
        semantic_rubric="记住猫叫团子，别猜其它名字。",
        checks=[dict(name="remember_name", type="contains_any", values=["团子"], hard=True)])


def judge_json(**overrides):
    value = dict(context=2, role=2, language=2, grounded=2,
                 critical_failure=False, reason="准确记住名字且自然简短。")
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


class FakeGenerator:
    def __init__(self, *, reply="你家猫叫团子呀。", judge=None, fail=False):
        self.model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        self.model.config = SimpleNamespace(use_cache=False)
        self.model.gradient_checkpointing = True
        self.model[0].gradient_checkpointing = True
        self.model[1].eval()  # Intentionally mixed training modes.
        self.projector = torch.nn.Linear(2, 2)
        self.projector.train()
        self.reply, self.judge, self.fail = reply, judge or judge_json(), fail
        self.calls = []

    def complete(self, system_prompt, messages, **kwargs):
        draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.calls.append(dict(system_prompt=system_prompt, messages=copy.deepcopy(messages),
                               draws=draws, **kwargs))
        self.model.eval()
        self.projector.eval()
        self.model.config.use_cache = True
        if system_prompt == JUDGE_SYSTEM_PROMPT:
            return self.judge
        if self.fail:
            raise RuntimeError("fake generation failed")
        return self.reply


class FakePlanner:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2).train()
        self.calls = []

    def predict(self, history, next_role):
        self.calls.append((copy.deepcopy(history), next_role))
        return {"target_vector": np.asarray([.25, -.5], dtype=np.float32)}


class ChatQualityTests(unittest.TestCase):
    def setUp(self):
        self.cuda_patch = patch("torch.cuda.is_available", return_value=False)
        self.cuda_patch.start()
        self.addCleanup(self.cuda_patch.stop)
        self.case = example_case()
        self.config = dict(system_prompt="你是meme桌面猫，只用中文短句。", chat_quality_seeds=[17, 23])

    def test_context_fact_probe_rejects_wrong_fact_even_if_fluent(self):
        good = score_reply(self.case, "你家猫叫团子呀。")
        bad = score_reply(self.case, "你家猫叫雪球呀。")
        self.assertTrue(good["hard_pass"])
        self.assertFalse(bad["hard_pass"])
        self.assertLess(bad["score"], good["score"])

    def test_empty_and_control_token_replies_fail_closed(self):
        for reply in ("", "   ", None, "<think>想一下</think>团子", "团子<|im_end|>"):
            with self.subTest(reply=reply):
                self.assertFalse(score_reply(self.case, reply)["hard_pass"])
        self.assertEqual(score_reply(self.case, "")["score"], 0)

    def test_semantic_judge_catches_keyword_stuffing_and_changes_main_score(self):
        reply = "团子，反正你问的事不重要。"
        mechanical = score_reply(self.case, reply)
        self.assertTrue(mechanical["hard_pass"])
        semantic = score_reply(self.case, reply, {"raw": judge_json(context=0, language=0,
                                                   critical_failure=True, reason="虽含名字但答非所问。")})
        self.assertFalse(semantic["hard_pass"])
        self.assertLess(semantic["score"], mechanical["score"])
        self.assertAlmostEqual(semantic["score"], .25 * semantic["rule_score"] + .75 * semantic["semantic_score"])

    def test_malformed_judgments_fail_closed(self):
        outputs = ["```json\n" + judge_json() + "\n```", "{}", "not json",
                   judge_json(context=True), judge_json(role=3), judge_json(reason=""),
                   judge_json()[:-1] + ',"context":2}']
        for raw in outputs:
            with self.subTest(raw=raw):
                result = score_reply(self.case, "猫叫团子。", {"raw": raw})
                self.assertFalse(result["hard_pass"])
                self.assertIsNone(result["semantic"])
        errored = score_reply(self.case, "猫叫团子。", {"raw": judge_json(), "error": "timeout"})
        self.assertFalse(errored["hard_pass"])
        self.assertIsNone(errored["semantic"])

    def test_one_reply_per_case_seed_and_same_prompt_rng_for_baseline(self):
        generator, planner = FakeGenerator(), FakePlanner()
        base = evaluate_chat_quality(generator, planner, [self.case], self.config, use_prefix=False)
        original = list(generator.calls)
        generator.calls.clear()
        candidate = evaluate_chat_quality(generator, planner, [self.case], self.config)
        self.assertEqual(len(original), 2)
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(len(planner.calls), 1)
        for before, after in zip(original, generator.calls):
            self.assertEqual(before["messages"], after["messages"])
            self.assertEqual(before["system_prompt"], after["system_prompt"])
            self.assertEqual(before["draws"], after["draws"])
            self.assertIsNone(before["stance_vector"])
            self.assertEqual(after["stance_vector"].tolist(), [.25, -.5])
            self.assertTrue(after["do_sample"])
        self.assertTrue(compare_quality(candidate, base)["passed"])

    def test_evaluation_restores_rng_and_mixed_training_cache_checkpointing(self):
        generator, planner = FakeGenerator(fail=True), FakePlanner()
        random.seed(9)
        np.random.seed(10)
        torch.manual_seed(11)
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.seed(9)
        np.random.seed(10)
        torch.manual_seed(11)
        modes = [module.training for module in generator.model.modules()]
        report = evaluate_chat_quality(generator, planner, [self.case], self.config)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(actual, expected)
        self.assertEqual([module.training for module in generator.model.modules()], modes)
        self.assertTrue(generator.projector.training)
        self.assertTrue(planner.model.training)
        self.assertFalse(generator.model.config.use_cache)
        self.assertTrue(generator.model.gradient_checkpointing)
        self.assertTrue(generator.model[0].gradient_checkpointing)
        self.assertGreater(report["hard_failures"], 0)
        self.assertTrue(all("fake generation failed" in record["error"] for record in report["cases"]))

    def test_semantic_judge_is_offline_greedy_no_prefix_and_bound_to_reply(self):
        config = {**self.config, "chat_quality_semantic_judge": True,
                  "chat_quality_judge_max_new_tokens": 256, "chat_quality_judge_max_prompt_tokens": 4096,
                  "chat_quality_max_prompt_tokens": 1024, "chat_quality_no_repeat_ngram_size": 3}
        generator = FakeGenerator()
        report = evaluate_chat_quality(generator, FakePlanner(), [self.case], config)
        self.assertEqual(len(generator.calls), 4)
        self.assertTrue(report["semantic_judge_enabled"])
        self.assertEqual(report["hard_failures"], 0)
        for record, call in zip(report["cases"], generator.calls[1::2]):
            expected_system, expected_messages = semantic_judge_prompt(self.case, record["reply"], config)
            self.assertEqual(call["system_prompt"], expected_system)
            self.assertEqual(call["messages"], expected_messages)
            self.assertIsNone(call["stance_vector"])
            self.assertFalse(call["do_sample"])
            self.assertEqual(call["max_new_tokens"], 256)
            self.assertEqual(call["max_prompt_tokens"], 4096)
            self.assertEqual(record["semantic_judge"]["raw"], judge_json())
            self.assertIn("history", json.loads(call["messages"][0]["content"]))
        self.assertEqual(report["generation_options"]["max_prompt_tokens"], 1024)
        self.assertEqual(report["generation_options"]["no_repeat_ngram_size"], 3)

    def test_cuda_rng_states_restored_without_accessing_a_gpu(self):
        fake_state = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_rng_state_all", return_value=fake_state) as read, \
             patch("torch.cuda.set_rng_state_all") as restore, \
             patch("torch.cuda.manual_seed_all"):
            report = evaluate_chat_quality(FakeGenerator(), FakePlanner(), [self.case], self.config)
        self.assertEqual(report["hard_failures"], 0)
        read.assert_called_once_with()
        restore.assert_called_once_with(fake_state)

    def test_baseline_judge_technical_failure_cannot_lower_acceptance_bar(self):
        config = {**self.config, "chat_quality_semantic_judge": True}
        base = evaluate_chat_quality(FakeGenerator(judge="broken"), FakePlanner(), [self.case], config, use_prefix=False)
        candidate = evaluate_chat_quality(FakeGenerator(), FakePlanner(), [self.case], config)
        comparison = compare_quality(candidate, base, config)
        self.assertFalse(comparison["passed"])
        self.assertTrue(any("technical failure" in reason for reason in comparison["reasons"]))

    def test_no_total_gain_can_hide_category_regression_or_missing_cases(self):
        generator, planner = FakeGenerator(), FakePlanner()
        base = evaluate_chat_quality(generator, planner, [self.case], self.config, use_prefix=False)
        candidate = evaluate_chat_quality(generator, planner, [self.case], self.config)
        candidate["category_scores"]["context_tracking"] -= .1
        self.assertFalse(compare_quality(candidate, base)["passed"])
        candidate["cases"].pop()
        self.assertTrue(any("coverage" in reason for reason in compare_quality(candidate, base)["reasons"]))

    def test_context_gain_cannot_hide_language_regression(self):
        config = {**self.config, "chat_quality_semantic_judge": True}
        base = evaluate_chat_quality(FakeGenerator(judge=judge_json(context=1)), FakePlanner(),
                                     [self.case], config, use_prefix=False)
        candidate = evaluate_chat_quality(FakeGenerator(judge=judge_json(language=1)), FakePlanner(),
                                          [self.case], config)
        self.assertEqual(candidate["score"], base["score"])
        comparison = compare_quality(candidate, base, config)
        self.assertFalse(comparison["passed"])
        self.assertEqual(comparison["semantic_dimension_deltas"]["language"], -1)

    def test_suites_are_independent_and_cover_all_eight_behaviors(self):
        root = Path(__file__).resolve().parents[1] / "evaluation"
        dev, holdout = load_cases(root / "chat_dev.jsonl"), load_cases(root / "chat_holdout.jsonl")
        self.assertEqual(len(dev), 16)
        self.assertEqual(len(holdout), 16)
        self.assertEqual(len({case["category"] for case in dev}), 8)
        self.assertEqual({case["category"] for case in dev}, {case["category"] for case in holdout})
        self.assertTrue({case["id"] for case in dev}.isdisjoint(case["id"] for case in holdout))
        self.assertTrue({json.dumps(case["messages"], ensure_ascii=False) for case in dev}.isdisjoint(
                       json.dumps(case["messages"], ensure_ascii=False) for case in holdout))
        self.assertTrue(all(case.get("semantic_rubric") for case in dev + holdout))

    def test_duplicate_suite_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cases.jsonl"
            path.write_text(json.dumps(self.case) + "\n" + json.dumps(self.case) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
