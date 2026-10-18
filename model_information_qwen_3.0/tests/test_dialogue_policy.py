"""Prompt regressions supplement, rather than bypass, the original release."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from model_information_qwen_v3 import dialogue_policy as policy
from model_information_qwen_v3 import model


class DialoguePolicyReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/"model_qwen/runs/selected"
        self.app = self.root/"model_information_qwen_v3"
        self.config_path = self.root/"model_qwen/inference_config.json"
        self.policy_path = self.app/"candidate.json"
        self.release_path = self.app/"dialogue_policy_release.json"
        self.config = dict(run_dir=str(self.run), system_prompt="原始角色指令。", device="cpu", dtype="float32",
                           data_dir=str(self.root/"data"), base_model_path=str(self.root/"base"),
                           max_prompt_tokens=4096, max_new_tokens=128, do_sample=True,
                           temperature=.7, top_p=.8, no_repeat_ngram_size=4,
                           serving_max_prompt_tokens=4096, top_k=20, enable_thinking=False)
        self.write(self.config_path, self.config)
        self.write(self.policy_path, dict(format_version=1, enabled=True, mode="append", policy="承接最新信息，避免无意义复读。"))
        for name in ("generator/best.pt", "planner/best.pt", "space.json"):
            self.write(self.run/name, b"unchanged trained model")
        for name in ("manifest.json", "generator/manifest.json", "generator/quality_dev.json",
                     "generator/quality_holdout.json", "deployment_selection.json"):
            self.write(self.run/name, dict(status="complete", quality_status="rejected"))
        for name, value in (("PROJECT_ROOT", self.root), ("APPLICATION_ROOT", self.app),
                            ("DEFAULT_POLICY_PATH", self.app/"dialogue_policy.json"),
                            ("DEFAULT_RELEASE_PATH", self.release_path)):
            stub = patch.object(policy, name, value)
            stub.start()
            self.addCleanup(stub.stop)
        self.inputs = []
        for phase, text in (("dev", "我已经承认熬夜了。"), ("holdout", "刚才已确定改为周三。")):
            cases_path, report_path = self.app/f"{phase}_cases.json", self.app/f"{phase}_report.json"
            case = dict(id=phase+"-confirmation", messages=[dict(role="user", content=text),
                dict(role="assistant", content="你确定吗？"), dict(role="user", content="是的。")])
            self.write(cases_path, dict(format_version=1, split=phase, cases=[case]))
            binding = policy.policy_binding(self.config, self.config_path, self.policy_path, cases_path)
            prompt = policy.effective_system_prompt(self.config, policy.load_policy(self.policy_path))
            outputs = [dict(id=case["id"], seed=seed, messages=copy.deepcopy(case["messages"]),
                            system_prompt=prompt, reply="好，已经确认了。", generate_calls=1,
                            full_history_retained=True, prompt_tokens=100,
                            prompt=prompt+"\n"+"\n".join(item["content"] for item in case["messages"]), binding=binding)
                       for seed in (1729, 2718)]
            report = dict(format_version=1, status="complete", binding=binding, seeds=[1729, 2718],
                          generation_options=policy.policy_generation_options(self.config),
                          regression=dict(status="unreviewed", scope=policy.RELEASE_SCOPE), cases=outputs)
            self.write(report_path, report)
            if phase == "holdout":
                self.write(Path(str(report_path)+".attempt.json"), dict(status="complete", binding=binding,
                    seeds=report["seeds"], report_sha256=policy._sha256(report_path)))
            self.inputs.append(dict(phase=phase, cases_path=str(cases_path), report_path=str(report_path),
                review=dict(status="passed", reviewer="codex_manual", notes="已逐例审阅完整历史和所有真实输出。")))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value)

    def signed_release(self):
        release = policy.build_policy_release(self.config, self.config_path, self.policy_path, self.inputs)
        self.write(self.release_path, release)
        return release

    def resolve(self):
        return policy.resolve_policy(self.config, self.config_path, policy_path=self.policy_path,
                                     release_path=self.release_path)

    def test_missing_or_explicitly_disabled_policy_keeps_original_prompt(self):
        state = policy.resolve_policy(self.config, self.config_path)
        self.assertFalse(state["enabled"])
        self.assertEqual(state["system_prompt"], self.config["system_prompt"])
        self.write(self.policy_path, dict(enabled=False, policy="尚未启用的规则。"))
        self.assertEqual(self.resolve()["status"], "disabled")

    def test_unreviewed_policy_cannot_be_activated(self):
        with self.assertRaisesRegex(ValueError, "no reviewed release report"):
            self.resolve()
        inputs = copy.deepcopy(self.inputs)
        inputs[0]["review"]["status"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "recorded semantic review"):
            policy.build_policy_release(self.config, self.config_path, self.policy_path, inputs)

    def test_two_reviewed_regressions_enable_append_without_relabeling_base_quality(self):
        release = self.signed_release()
        state = self.resolve()
        self.assertTrue(state["enabled"])
        self.assertEqual(state["status"], "regression_passed")
        self.assertEqual(state["base_quality_status"], "rejected")
        self.assertEqual(release["scope"], "dialogue_repetition_regression_only")
        self.assertEqual(state["system_prompt"], "原始角色指令。\n\n承接最新信息，避免无意义复读。")

    def test_replace_mode_uses_full_new_prompt_and_keeps_original_binding(self):
        self.write(self.policy_path, dict(mode="replace", policy="全新中文角色指令。"))
        document = policy.load_policy(self.policy_path)
        self.assertEqual(policy.effective_system_prompt(self.config, document), "全新中文角色指令。")
        binding = policy.policy_binding(self.config, self.config_path, self.policy_path, self.inputs[0]["cases_path"])
        self.assertEqual(binding["base_system_prompt_sha256"], policy._text_sha256(self.config["system_prompt"]))
        self.assertEqual(binding["effective_system_prompt_sha256"], policy._text_sha256("全新中文角色指令。"))

    def test_dev_only_or_nonindependent_holdout_cannot_release(self):
        with self.assertRaisesRegex(ValueError, "independent holdout"):
            policy.build_policy_release(self.config, self.config_path, self.policy_path, self.inputs[:1])
        duplicated = copy.deepcopy(self.inputs)
        duplicated[1]["phase"] = "dev"
        with self.assertRaisesRegex(ValueError, "one dev and one holdout"):
            policy.build_policy_release(self.config, self.config_path, self.policy_path, duplicated)

    def test_model_policy_config_and_original_reports_are_bound(self):
        self.signed_release()
        paths = (self.run/"generator/best.pt", self.run/"planner/best.pt", self.run/"space.json",
                 self.config_path, self.run/"generator/quality_dev.json", self.run/"generator/quality_holdout.json")
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original+b" ")
                with self.assertRaisesRegex(ValueError, "stale bindings"):
                    self.resolve()
                path.write_bytes(original)
        original = self.policy_path.read_bytes()
        self.policy_path.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError, "stale bindings"):
            self.resolve()

    def test_replacing_output_after_review_cannot_reuse_old_release(self):
        self.signed_release()
        path = Path(self.inputs[0]["report_path"])
        report = json.loads(path.read_text())
        report["cases"][0]["reply"] = "被替换的另一条回复。"
        self.write(path, report)
        with self.assertRaisesRegex(ValueError, "reviewed report has changed"):
            self.resolve()

    def test_different_decoding_or_missing_latest_user_cannot_pass_evidence_validation(self):
        path = Path(self.inputs[0]["report_path"])
        original = json.loads(path.read_text())
        changes = [lambda r:r["generation_options"].update(temperature=.9),
                   lambda r:r["cases"][0].update(generate_calls=2),
                   lambda r:r["cases"][0].update(prompt=r["cases"][0]["system_prompt"]),
                   lambda r:r["cases"][0].update(error={"type":"RuntimeError"})]
        for change in changes:
            with self.subTest(change=change):
                report = copy.deepcopy(original)
                change(report)
                self.write(path, report)
                with self.assertRaises(ValueError):
                    self.signed_release()
        self.write(path, original)

    def test_incomplete_or_changed_once_only_holdout_marker_blocks_activation(self):
        path = Path(self.inputs[1]["report_path"]+".attempt.json")
        marker = json.loads(path.read_text())
        marker["status"] = "running"
        self.write(path, marker)
        with self.assertRaisesRegex(ValueError, "one completed frozen assessment"):
            self.signed_release()

    def test_original_release_validates_before_supplement_and_backbone(self):
        events = []
        def original(*args, **kwargs):
            events.append("original")
        def supplement(*args, **kwargs):
            events.append("supplement")
            raise ValueError("Supplement not reviewed")
        with patch.object(model, "_configuration", return_value=(self.config_path, self.config)), \
             patch.object(model, "_ready", side_effect=original), \
             patch.object(policy, "resolve_policy", side_effect=supplement), \
             patch.object(model, "_backbone", side_effect=AssertionError("must not load weights")):
            with self.assertRaisesRegex(ValueError, "Supplement not reviewed"):
                model.load_engine(device="cpu")
        self.assertEqual(events, ["original", "supplement"])

    def test_chat_applies_verified_prompt_without_changing_original_config(self):
        self.signed_release()
        state = self.resolve()
        raw = SimpleNamespace(config=dict(self.config), dialogue_policy_state=state)
        settings = dict(max_prompt_tokens=4096, max_reply_tokens=128, do_sample=True, temperature=.7,
                        top_p=.8, no_repeat_ngram_size=4, role_prompt=self.config["system_prompt"], asr_enabled=False)
        with patch.object(model, "_configuration", return_value=(self.config_path, self.config)), \
             patch.object(model, "load_engine", return_value=raw), \
             patch("model_information_qwen_v3.chat_engine.load_chat_settings", return_value=settings), \
             patch("model_information_qwen_v3.chat_engine.EmotionalChatEngine", side_effect=lambda r,s:SimpleNamespace(settings=s)):
            chat = model.load_chat_engine()
        self.assertEqual(chat.settings["role_prompt"], state["system_prompt"])
        self.assertEqual(raw.config["system_prompt"], "原始角色指令。")
        self.assertEqual(chat.dialogue_policy_state["base_quality_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
