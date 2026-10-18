"""Loss-only training can finish for review without any chat-quality release."""
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from emostance_qwen import paper_pipeline as pipeline
from model_qwen.tools import monitor_training, status, workflow


class LossOnlyPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root/"runs/qwen3_8b_v3_loss_ce"
        self.data = self.root/"data"
        self.config = dict(run_dir=str(self.run), data_dir=str(self.data), version="qwen3_8b_v3_loss_ce",
            model_version="3.0.0", data_version="2.0.0", generator_selection_mode="dev_response_ce",
            chat_quality_enabled=False, prefix_epochs=3, planner_epochs=3, seed=42,
            planner_model_path=str(self.root/"planner"), base_model_path=str(self.root/"base"),
            evaluation_subset_per_split=512, system_prompt="保持简短中文回复。")
        self.config_path = self.root/"paper_config_v3_loss.json"
        self.write(self.config_path, self.config)
        scoped = patch.object(pipeline, "MODEL_DIR", self.root)
        scoped.start()
        self.addCleanup(scoped.stop)
        for name in ("planner/best.pt", "generator/best.pt", "space.json", "response_vectors.npz", "response_anchors.jsonl"):
            self.write(self.run/name, b"new version artifact")
        self.write(self.run/"planner/manifest.json", {"status":"complete"})
        self.write(self.run/"graph/manifest.json", {"status":"complete", "num_clusters":2})
        self.rows = {}
        for split in ("train", "dev", "test"):
            rows = [dict(sample_id=split+str(index), split=split, history=[dict(role="A", text="你好")],
                         next_role="B", response="短" if index == 0 else "稍长", base_ce=1. if index == 0 else 3.,
                         target_meme_distribution={"meme-0001":1.}) for index in range(2)]
            self.rows[split] = rows
            path = self.data/f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
        self.write(self.run/"generator/dev_subset.json", {"split":"dev", "sample_ids":["dev0", "dev1"], "seed":42})
        self.generator_manifest = dict(status="complete", generator_selection_mode="dev_response_ce",
            selection_criterion="dev_response_ce", quality_status="not_evaluated", chat_quality_status="not_evaluated",
            version=self.config["version"], model_version=self.config["model_version"], data_version=self.config["data_version"],
            best_dev_response_ce=1.5, best_step=2, best_epoch=1, global_step=6,
            checkpoint_sha256=pipeline.file_sha256(self.run/"generator/best.pt"))
        self.write(self.run/"generator/manifest.json", self.generator_manifest)
        self.selection = dict(best_checkpoint_sha256=self.generator_manifest["checkpoint_sha256"],
            version=self.config["version"], model_version=self.config["model_version"], data_version=self.config["data_version"],
            dev_subset_sha256=pipeline.file_sha256(self.run/"generator/dev_subset.json"), best_dev_response_ce=1.5,
            best_step=2, best_epoch=1, global_step=6, control="planner_predicted", weighting="reply_eos_token_weighted")
        self.write(self.run/"generator/loss_selection.json", self.selection)
        self.write(self.run/"response_index_manifest.json", dict(status="complete", split="train",
            train_file_sha256=pipeline.file_sha256(self.data/"train.jsonl"),
            planner_checkpoint_sha256=pipeline.file_sha256(self.run/"planner/best.pt"),
            space_sha256=pipeline.file_sha256(self.run/"space.json")))

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value)

    def evaluate(self):
        space = types.SimpleNamespace(targets=lambda distribution:(np.asarray([1.,0.]), np.asarray([2.,0.])))
        planner = types.ModuleType("emostance_qwen.paper_planner")
        planner.load_planner = lambda *args:types.SimpleNamespace(space=space,
            predict_rows=lambda rows:{"target_vector":np.asarray([[3.,0.] for row in rows])})
        generator = types.ModuleType("emostance_qwen.paper_generator")
        def forbidden_chat(*args, **kwargs):
            raise AssertionError("Loss-only evaluation must never generate or judge chat")
        generator.PaperGenerator = lambda *args, **kwargs:types.SimpleNamespace(
            model=types.SimpleNamespace(eval=lambda:None), complete=forbidden_chat, generate=forbidden_chat,
            response_ids=lambda response:torch.zeros(2 if response == "短" else 4),
            loss=lambda row,vector:row["base_ce"]+float(vector[0])/10)
        with patch.dict(sys.modules, {planner.__name__:planner, generator.__name__:generator}), \
             patch.object(pipeline, "require_inference_release", side_effect=forbidden_chat), \
             patch.object(pipeline, "require_quality_release", side_effect=forbidden_chat), \
             patch.object(torch.cuda, "empty_cache", return_value=None):
            return pipeline.evaluate_loss_only(self.config)

    def test_loss_evaluation_weights_reply_eos_tokens_and_never_uses_chat_gates(self):
        result = self.evaluate()
        self.assertAlmostEqual(result["dev"]["generator"]["token_ce"]["predicted_control"], (1.3*2+3.3*4)/6)
        self.assertEqual(result["dev"]["generator"]["reply_eos_tokens"], 6)
        self.assertEqual(result["dev"]["generator"]["sample_ids"], ["dev0", "dev1"])
        self.assertFalse(result["test_used_for_checkpoint_selection"])
        self.assertEqual(result["chat_quality_status"], "not_evaluated")

    def test_completed_loss_version_is_ready_for_review_without_inference_authorization(self):
        self.evaluate()
        manifest = json.loads((self.run/"manifest.json").read_text())
        self.assertEqual(manifest["status"], "ready_for_review")
        self.assertEqual(manifest["training_status"], "complete")
        self.assertEqual(manifest["model_version"], "3.0.0")
        self.assertEqual(manifest["data_version"], "2.0.0")
        self.assertEqual(manifest["quality_status"], "not_evaluated")
        self.assertFalse(manifest["inference_enabled"])
        self.assertEqual(manifest["inference_release_status"], "not_authorized")
        with self.assertRaisesRegex(ValueError, "development chat quality"):
            pipeline.require_quality_release(self.config)
        with self.assertRaisesRegex(ValueError, "development chat quality"):
            pipeline.require_inference_release(self.config)

    def test_changed_best_checkpoint_or_selection_evidence_blocks_training_finalization(self):
        path = self.run/"generator/best.pt"
        old = path.read_bytes()
        path.write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            pipeline.require_loss_only_training(self.config)
        path.write_bytes(old)
        self.write(self.run/"generator/loss_selection.json", {**self.selection, "best_step":5})
        with self.assertRaisesRegex(ValueError, "stale or inconsistent"):
            pipeline.require_loss_only_training(self.config)

    def test_loss_configuration_requires_explicit_version_and_disabled_chat_evaluation(self):
        for change, expected in (({"version":""}, "version"), ({"chat_quality_enabled":True}, "chat_quality_enabled")):
            self.write(self.config_path, {**self.config, **change})
            with self.assertRaisesRegex(ValueError, expected):
                pipeline.load_config(self.config_path)

    def test_training_version_mismatch_rejects_reused_artifacts(self):
        for name in ("version", "model_version", "data_version"):
            with self.subTest(name=name, artifact="generator"):
                self.write(self.run/"generator/manifest.json", {**self.generator_manifest, name:"old-version"})
                with self.assertRaisesRegex(ValueError, "version mismatch: " + name):
                    pipeline.require_loss_only_training(self.config)
            self.write(self.run/"generator/manifest.json", self.generator_manifest)
            with self.subTest(name=name, artifact="selection"):
                self.write(self.run/"generator/loss_selection.json", {**self.selection, name:"old-version"})
                with self.assertRaisesRegex(ValueError, "version mismatch: " + name):
                    pipeline.require_loss_only_training(self.config)
            self.write(self.run/"generator/loss_selection.json", self.selection)

    def test_running_status_uses_latest_dev_event_instead_of_partial_selection(self):
        self.write(self.run/"workflow_status.json", dict(status="running", stage="generator", pid=12345))
        event = dict(event="dev_response_ce", step=1000, best_step=1000, best_dev_response_ce=1.2)
        (self.run/"generator/loss_metrics.jsonl").write_text(json.dumps(event)+"\n", encoding="utf-8")
        with patch.object(status, "ROOT", self.root), patch.object(status.os, "kill"):
            report = status.collect_status(self.config_path)
        self.assertEqual(report["loss_selection"]["best_step"], 1000)
        self.assertEqual(report["loss_selection_source"], "live_dev_response_ce")
        with patch.object(monitor_training, "alive", return_value=True):
            snapshot = monitor_training.snapshot(self.run, self.config)
        self.assertEqual(snapshot["loss_selection"]["best_step"], 1000)
        self.assertEqual(snapshot["loss_selection_source"], "live_dev_response_ce")

    def test_workflow_skips_old_smoke_chat_and_inference_for_new_loss_version(self):
        names = [name for name,command in workflow.build_stages(self.config, self.config_path)]
        self.assertEqual(names, ["planner", "generator", "index", "evaluate"])
        quality_config = {**self.config, "generator_selection_mode":"chat_quality"}
        previous = [name for name,command in workflow.build_stages(quality_config, self.config_path)]
        self.assertIn("quality", previous)
        self.assertIn("inference_smoke", previous)

    def test_status_and_monitor_distinguish_training_readiness_from_active_inference(self):
        self.evaluate()
        self.write(self.run/"workflow_status.json", dict(status="ready_for_review", stage="ready_for_review", completed_stages=["evaluate"]))
        self.write(self.root/"inference_config.json", {"run_dir":"runs/previous_selected"})
        self.write(self.root/"runs/previous_selected/manifest.json", {"status":"complete", "quality_status":"rejected"})
        with patch.object(status, "ROOT", self.root):
            report = status.collect_status(self.config_path)
        self.assertTrue(report["training_complete"])
        self.assertTrue(report["ready_for_review"])
        self.assertEqual(report["chat_quality"]["status"], "not_evaluated")
        self.assertEqual(report["active_inference"]["run_dir"], str(self.root/"runs/previous_selected"))
        self.assertEqual(report["active_inference"]["quality_status"], "rejected")
        with patch.object(monitor_training, "alive", return_value=False):
            snapshot = monitor_training.snapshot(self.run, self.config)
        self.assertEqual(snapshot["status"], "ready_for_review")
        self.assertTrue(snapshot["training_complete"])
        self.assertFalse(snapshot["final_model_complete"])
        self.assertFalse(snapshot["real_inference_complete"])
        self.assertEqual(snapshot["alerts"], [])


if __name__ == "__main__":
    unittest.main()
