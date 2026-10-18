"""A trained prefix stays unavailable until chat evidence passes both gates."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from emostance_qwen import paper_pipeline as pipeline


class QualityReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/"runs/full"
        self.model_dir = patch.object(pipeline, "MODEL_DIR", self.root)
        self.model_dir.start()
        self.addCleanup(self.model_dir.stop)
        cpu_only = patch("torch.cuda.is_available", return_value=False)
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        for name in ("planner/best.pt", "generator/best.pt", "space.json"):
            self.write(self.run/name, b"frozen local artifact")
        self.write(self.run/"planner/manifest.json", {"status":"complete"})
        self.config = dict(run_dir=str(self.run), system_prompt="你是meme，请用简短中文回应。",
                           chat_quality_seeds=[1729, 2718], chat_quality_semantic_judge=False)
        for phase, question in (("dev", "请确认收到消息。"), ("holdout", "请确认收到通知。")):
            path = self.root/f"{phase}.jsonl"
            case = dict(id=phase+"-confirm", category="context", messages=[dict(role="user", content=question)],
                        checks=[dict(name="confirmation", type="contains_all", values=["收到"], hard=True)])
            path.write_text(json.dumps(case, ensure_ascii=False)+"\n", encoding="utf-8")
            self.config[f"chat_quality_{phase}_file"] = str(path)
        self.generator = types.SimpleNamespace(complete=lambda *args, **kwargs: "收到。")
        self.planner = types.SimpleNamespace(predict=lambda *args: {"target_vector":np.asarray([1., 0.])})
        self.reports = {}
        self.manifest = dict(status="complete", quality_status="passed",
                             selection_criterion=pipeline.QUALITY_SELECTION_CRITERION,
                             checkpoint_sha256=pipeline.file_sha256(self.run/"generator/best.pt"))
        for phase in ("dev", "holdout"):
            report = pipeline.build_quality_report(self.config, self.generator, self.planner, phase=phase,
                                                   checkpoint_path=self.run/"generator/best.pt")
            self.reports[phase] = report
            relative = f"generator/quality_{phase}.json"
            self.write(self.run/relative, report)
            key = "quality_report" if phase == "dev" else "holdout_report"
            self.manifest[key] = relative
            self.manifest[key+"_sha256"] = pipeline.file_sha256(self.run/relative)
        self.write(self.run/"generator/manifest.json", self.manifest)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value)

    def test_complete_chat_evidence_passes_and_keeps_base_and_prefix_outputs(self):
        reports = pipeline.require_quality_release(self.config)
        for report in reports.values():
            self.assertEqual(len(report["baseline"]["cases"]), 2)
            self.assertEqual(len(report["candidate"]["cases"]), 2)
            self.assertTrue(report["comparison"]["passed"])

    def test_loss_only_completed_training_is_not_a_releasable_model(self):
        self.write(self.run/"generator/manifest.json", {"status":"complete", "best_metric":"train_reply_ce"})
        with self.assertRaisesRegex(ValueError, "development chat quality"):
            pipeline.require_quality_release(self.config)
        with self.assertRaisesRegex(ValueError, "development chat quality"):
            pipeline.finalize_run(self.config)

    def test_a_missing_holdout_report_blocks_release(self):
        manifest = {**self.manifest}
        manifest.pop("holdout_report")
        self.write(self.run/"generator/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "holdout"):
            pipeline.require_quality_release(self.config)

    def test_changed_prefix_planner_prompt_or_suite_invalidates_evidence(self):
        changes = (self.run/"generator/best.pt", self.run/"planner/best.pt", self.root/"dev.jsonl")
        for path in changes:
            with self.subTest(path=path):
                original = path.read_bytes()
                replacement = b"X"+original[1:] if path.suffix == ".pt" else original.replace("消息".encode(), "来信".encode())
                path.write_bytes(replacement)
                with self.assertRaisesRegex(ValueError, "SHA-256|Stale"):
                    pipeline.require_quality_release(self.config)
                path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "Stale"):
            pipeline.require_quality_release({**self.config, "system_prompt":"你是另一个角色。"})

    def test_rewritten_reply_cannot_reuse_old_scores_even_if_report_hash_is_updated(self):
        report = copy.deepcopy(self.reports["dev"])
        report["candidate"]["cases"][0]["reply"] = "没有收到。<think>"
        path = self.run/"generator/quality_dev.json"
        self.write(path, report)
        self.write(self.run/"generator/manifest.json", {
            **self.manifest, "quality_report_sha256":pipeline.file_sha256(path)})
        with self.assertRaisesRegex(ValueError, "inconsistent with its reply"):
            pipeline.require_quality_release(self.config)

    def test_rejected_chat_behavior_blocks_release_despite_completed_training(self):
        broken = types.SimpleNamespace(complete=lambda *args, stance_vector=None, **kwargs:
                                       "无关回复。" if stance_vector is not None else "收到。")
        report = pipeline.build_quality_report(self.config, broken, self.planner, phase="holdout",
                                               checkpoint_path=self.run/"generator/best.pt")
        self.assertEqual(report["status"], "rejected")
        path = self.run/"generator/quality_holdout.json"
        self.write(path, report)
        self.write(self.run/"generator/manifest.json", {
            **self.manifest, "holdout_report_sha256":pipeline.file_sha256(path)})
        with self.assertRaisesRegex(ValueError, "holdout chat quality rejected"):
            pipeline.require_quality_release(self.config)

    def test_holdout_cannot_be_rerun_after_a_failed_attempt(self):
        binding = pipeline.quality_binding(self.config, "holdout", self.run/"generator/best.pt")
        self.write(self.run/"generator/quality_holdout_attempt.json", dict(status="running", binding=binding))
        (self.run/"generator/quality_holdout.json").unlink()
        with self.assertRaisesRegex(ValueError, "already attempted"):
            pipeline.final_quality(self.config)

    def test_final_holdout_report_is_reused_without_generating_again(self):
        binding = pipeline.quality_binding(self.config, "holdout", self.run/"generator/best.pt")
        self.write(self.run/"generator/quality_holdout_attempt.json", dict(status="passed", binding=binding,
            report_sha256=pipeline.file_sha256(self.run/"generator/quality_holdout.json")))
        generator = types.ModuleType("emostance_qwen.paper_generator")
        generator.PaperGenerator = lambda *args, **kwargs: self.fail("Holdout must not generate again")
        with patch.dict(sys.modules, {generator.__name__:generator}):
            report = pipeline.final_quality(self.config)
        self.assertEqual(report, self.reports["holdout"])

    def test_semantic_judge_record_is_bound_to_the_actual_history_and_reply(self):
        config = {**self.config, "chat_quality_semantic_judge":True}
        verdict = json.dumps(dict(context=2, role=2, language=2, grounded=2,
                                  critical_failure=False, reason="理解上下文且表达自然。"), ensure_ascii=False)
        generator = types.SimpleNamespace(complete=lambda system, *args, **kwargs:
                                           verdict if "离线中文对话质量评审" in system else "收到。")
        report = pipeline.build_quality_report(config, generator, self.planner, phase="dev",
                                               checkpoint_path=self.run/"generator/best.pt")
        self.assertTrue(report["comparison"]["passed"])
        self.assertEqual(report["candidate"]["cases"][0]["semantic_score"], 1.)
        changed = copy.deepcopy(report)
        changed["candidate"]["cases"][0]["semantic_judge"]["messages"][0]["content"] = "另一段上下文"
        with self.assertRaisesRegex(ValueError, "does not match the actual chat"):
            pipeline.validate_quality_report(changed, config, "dev", self.run/"generator/best.pt")

    def test_holdout_cannot_become_a_second_checkpoint_selection_suite(self):
        binding = pipeline.quality_binding(self.config, "holdout", self.run/"generator/best.pt")
        self.write(self.run/"generator/quality_holdout_attempt.json", dict(status="passed", binding=binding))
        old = self.run/"generator/best.pt"
        old.write_bytes(b"another selected prefix")
        # Simulate someone selecting a new dev checkpoint after seeing holdout.
        report = pipeline.bind_quality_report(self.reports["dev"], self.config, "dev", old)
        self.write(self.run/"generator/quality_dev.json", report)
        self.write(self.run/"generator/manifest.json", {**self.manifest,
            "checkpoint_sha256":pipeline.file_sha256(old),
            "quality_report_sha256":pipeline.file_sha256(self.run/"generator/quality_dev.json")})
        with self.assertRaisesRegex(ValueError, "already attempted"):
            pipeline.final_quality(self.config)


if __name__ == "__main__":
    unittest.main()
