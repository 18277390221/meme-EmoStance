"""Explicit selection permits known quality failures without falsifying acceptance."""
from __future__ import annotations

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


class UserSelectedReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.source = self.root/"runs/deployed", self.root/"runs/trained"
        for target, value in ((pipeline, self.root),):
            stub = patch.object(target, "MODEL_DIR", value)
            stub.start()
            self.addCleanup(stub.stop)
        cpu = patch("torch.cuda.is_available", return_value=False)
        cpu.start()
        self.addCleanup(cpu.stop)
        for name in ("planner/best.pt", "generator/best.pt", "space.json"):
            self.write(self.run/name, b"selected local artifact")
        self.write(self.run/"planner/manifest.json", {"status":"complete"})
        self.write(self.source/"generator/last.pt", b"complete source checkpoint")
        self.source_manifest = dict(status="complete", global_step=3, train_examples=3,
                                    epochs_requested=1, epochs_complete=1, quality_status="rejected")
        self.write(self.source/"generator/manifest.json", self.source_manifest)
        self.config = dict(run_dir=str(self.run), system_prompt="你是meme，用简短中文回应。",
                           chat_quality_seeds=[1729, 2718], chat_quality_semantic_judge=False)
        for phase in ("dev", "holdout"):
            path = self.root/f"{phase}.jsonl"
            case = dict(id=phase+"-confirm", category="context", messages=[dict(role="user", content=phase+"请确认消息。")],
                        checks=[dict(name="confirmation", type="contains_all", values=["收到"], hard=True)])
            path.write_text(json.dumps(case, ensure_ascii=False)+"\n", encoding="utf-8")
            self.config[f"chat_quality_{phase}_file"] = str(path)
        self.generator = types.SimpleNamespace(complete=lambda *args, stance_vector=None, **kwargs:
                                               "无关回复。" if stance_vector is not None else "收到。")
        self.planner = types.SimpleNamespace(predict=lambda *args: {"target_vector":np.asarray([1., 0.])})
        self.manifest = {**self.source_manifest, "selection_criterion":pipeline.USER_SELECTED_RELEASE,
                         "checkpoint_sha256":pipeline.file_sha256(self.run/"generator/best.pt")}
        self.record = dict(format_version=1, release_mode=pipeline.USER_SELECTED_RELEASE, status="pending_quality",
            user_selection=dict(authorized=True, accept_known_quality_failures=True, source="user",
                                authorization_text="选择充分训练的最好模型用于推理。", selection_reason="已知质量问题后选择最终完整训练模型。"),
            selected_checkpoint=dict(source_run=str(self.source), source_checkpoint=str(self.source/"generator/last.pt"),
                source_checkpoint_sha256=pipeline.file_sha256(self.source/"generator/last.pt"),
                source_training_manifest_sha256=pipeline.file_sha256(self.source/"generator/manifest.json"), global_step=3),
            binding=pipeline.deployment_selection_binding(self.config), quality_reports={}, quality_status="rejected")
        self.save_dev()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value)

    def save_dev(self):
        report = pipeline.build_quality_report(self.config, self.generator, self.planner, phase="dev",
                                               checkpoint_path=self.run/"generator/best.pt")
        report["global_step"] = 3
        path = self.run/"generator/quality_dev.json"
        self.write(path, report)
        self.manifest.update(quality_report="generator/quality_dev.json", quality_report_sha256=pipeline.file_sha256(path),
                             quality_status=report["status"])
        self.record.update(binding=pipeline.deployment_selection_binding(self.config), quality_status=report["status"])
        self.record["quality_reports"]["dev"] = dict(path=self.manifest["quality_report"], sha256=self.manifest["quality_report_sha256"])
        self.write(self.run/"generator/manifest.json", self.manifest)
        self.write(self.run/"deployment_selection.json", self.record)

    def assess(self):
        generator = types.ModuleType("emostance_qwen.paper_generator")
        generator.PaperGenerator = lambda *args, **kwargs: self.generator
        planner = types.ModuleType("emostance_qwen.paper_planner")
        planner.load_planner = lambda *args, **kwargs: self.planner
        with patch.dict(sys.modules, {generator.__name__:generator, planner.__name__:planner}):
            return pipeline.assess_selected_quality(self.config)

    def test_selected_rejected_model_can_serve_without_claiming_quality_passed(self):
        report = self.assess()
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(pipeline.load_deployment_selection(self.config)["quality_status"], "rejected")
        self.assertEqual(set(pipeline.require_inference_release(self.config)), {"dev", "holdout"})
        with self.assertRaisesRegex(ValueError, "development chat quality"):
            pipeline.require_quality_release(self.config)

    def test_no_selection_record_preserves_the_original_strict_guard(self):
        (self.run/"deployment_selection.json").unlink()
        with patch.object(pipeline, "require_quality_release", return_value={"strict":True}) as strict:
            self.assertEqual(pipeline.require_inference_release(self.config), {"strict":True})
        strict.assert_called_once_with(self.config)

    def test_missing_known_failure_authorization_cannot_bypass_acceptance(self):
        self.record["user_selection"]["accept_known_quality_failures"] = False
        self.write(self.run/"deployment_selection.json", self.record)
        with self.assertRaisesRegex(ValueError, "explicit user authorization"):
            self.assess()

    def test_pending_selection_cannot_be_served_without_independent_holdout(self):
        with self.assertRaisesRegex(ValueError, "assessment is not complete"):
            pipeline.require_inference_release(self.config)

    def test_declared_complete_but_short_training_is_rejected(self):
        self.source_manifest["global_step"] = 2
        path = self.source/"generator/manifest.json"
        self.write(path, self.source_manifest)
        self.record["selected_checkpoint"].update(global_step=2, source_training_manifest_sha256=pipeline.file_sha256(path))
        self.write(self.run/"deployment_selection.json", self.record)
        with self.assertRaisesRegex(ValueError, "all requested training"):
            self.assess()

    def test_source_prefix_planner_space_and_configuration_are_bound(self):
        self.assess()
        for path in (self.source/"generator/last.pt", self.source/"generator/manifest.json",
                     self.run/"generator/best.pt", self.run/"planner/best.pt", self.run/"space.json"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"changed "+original)
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    pipeline.require_inference_release(self.config)
                path.write_bytes(original)
        with self.assertRaisesRegex(ValueError, "configuration SHA-256 mismatch"):
            pipeline.require_inference_release({**self.config, "system_prompt":"新的角色设定。"})

    def test_generation_technical_errors_cannot_be_overridden_by_user_selection(self):
        def complete(*args, stance_vector=None, **kwargs):
            if stance_vector is not None:
                raise RuntimeError("evaluation failed")
            return "收到。"
        self.generator = types.SimpleNamespace(complete=complete)
        self.save_dev()
        with self.assertRaisesRegex(ValueError, "incomplete generation evaluation"):
            self.assess()

    def test_invalid_semantic_judge_cannot_be_overridden_by_user_selection(self):
        self.config["chat_quality_semantic_judge"] = True
        self.generator = types.SimpleNamespace(complete=lambda system, *args, stance_vector=None, **kwargs:
            "非法JSON" if "离线中文对话质量评审" in system else "收到。")
        self.save_dev()
        with self.assertRaisesRegex(ValueError, "invalid semantic judge evaluation"):
            self.assess()

    def test_selected_holdout_is_not_generated_again_after_assessment(self):
        self.assess()
        generator = types.ModuleType("emostance_qwen.paper_generator")
        generator.PaperGenerator = lambda *args, **kwargs: self.fail("Must not repeat holdout inference")
        with patch.dict(sys.modules, {generator.__name__:generator}):
            self.assertEqual(pipeline.assess_selected_quality(self.config)["status"], "rejected")

    def test_final_manifest_preserves_failure_and_records_user_selection(self):
        self.assess()
        self.write(self.run/"graph/manifest.json", {"num_clusters":2})
        self.write(self.run/"response_index_manifest.json", {"status":"complete", "split":"train"})
        self.write(self.run/"evaluation.json", {"dev":{}, "test":{}})
        self.write(self.run/"response_vectors.npz", b"index")
        self.write(self.run/"response_anchors.jsonl", b"anchors")
        manifest = pipeline.finalize_run(self.config)
        self.assertEqual(manifest["quality_status"], "rejected")
        self.assertEqual(manifest["release_mode"], pipeline.USER_SELECTED_RELEASE)
        self.assertEqual(manifest["quality_selection_criterion"], pipeline.USER_SELECTED_RELEASE)
        self.assertIn("deployment_selection.json", manifest["artifact_sha256"])


if __name__ == "__main__":
    unittest.main()
