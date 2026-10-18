"""Development-CE selection and exact resume without loading a backbone."""
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
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emostance_qwen import paper_generator as training


class TinyLossGenerator:
    """Real optimizer gradients with controlled out-of-sample CE measurements."""
    training_ids = []
    observed_dev_vectors = []

    def __init__(self, config, checkpoint=None):
        self.config, self.device = config, torch.device("cpu")
        self.base_model_path = Path(config["base_model_path"])
        self.model = nn.Sequential(nn.Linear(1, 1), nn.Dropout(.2))
        self.model.requires_grad_(False)
        self.model.config = SimpleNamespace(use_cache=True)
        self.projector = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projector.weight.fill_(1)
        self.checkpoint_state = torch.load(checkpoint, weights_only=False) if checkpoint else None
        self.updates = 0
        if self.checkpoint_state is not None:
            self.projector.load_state_dict(self.checkpoint_state["projector"])
            self.updates = self.checkpoint_state["global_step"]

    def architecture(self):
        return {"vector_dim": 1}

    def response_ids(self, text):
        return torch.ones((1, 2 if text == "短" else 6), dtype=torch.long)

    def loss(self, row, vector):
        if row["split"] == "train":
            self.updates += 1
            type(self).training_ids.append(row["sample_id"])
            scale = 1 + .01 * (random.random() + np.random.rand() + float(torch.rand(())))
            return self.projector.weight.square().sum() * scale
        if torch.is_grad_enabled() or self.model.training or self.projector.training:
            raise AssertionError("Development loss must be eval-only without gradients")
        type(self).observed_dev_vectors.append(np.asarray(vector).tolist())
        # Deliberate RNG consumption proves evaluation does not change training.
        random.random(), np.random.rand(), torch.rand(())
        values = self.config.get("test_dev_ce", {"1": 3.0, "2": 1.0, "3": 4.0})
        value = values[str(self.updates)] + row.get("dev_ce_offset", 0)
        return torch.tensor(value)

    def distillation_losses(self, *args, **kwargs):
        raise AssertionError("Pure reply CE must never call teacher distillation")


class LossOnlyTrainingTests(unittest.TestCase):
    def setUp(self):
        TinyLossGenerator.training_ids = []
        TinyLossGenerator.observed_dev_vectors = []
        self.cuda = patch("torch.cuda.is_available", return_value=False)
        self.cuda.start()
        self.addCleanup(self.cuda.stop)

    def fixture(self, root):
        rows = [{"sample_id": f"train-{i}", "split": "train", "history": [{"role": "A", "text": "你好"}],
                 "next_role": "B", "response": "长回复无需按角色长度筛掉", "target_meme_distribution": {"m": 1.0}}
                for i in range(3)]
        dev = [{**rows[0], "sample_id": "dev-0", "split": "dev", "response": "短", "dev_ce_offset": 0},
               {**rows[0], "sample_id": "dev-1", "split": "dev", "response": "长", "dev_ce_offset": 2}]
        space = SimpleNamespace(vector_dim=1, meme_ids=["m"], centroids=np.ones((1, 1)),
            membership=np.ones((1, 1)), prototypes=np.ones((1, 1)),
            targets=lambda labels: (np.ones(1), np.asarray([99.0])))
        config = dict(base_model_path=str(root / "backbone"), prefix_epochs=1, prefix_learning_rate=.05,
            gradient_checkpointing=False, generator_runtime_checkpoint_interval=1, generator_dev_eval_interval=1,
            generator_selection_mode="dev_response_ce", chat_quality_enabled=False,
            prefix_teacher_kl_weight=0, language_preservation_weight=0, language_preservation_ce_weight=0,
            prefix_dev_subset_size=2, prefix_ce_max_response_chars=0, prefix_ce_exclude_persona_claims=False,
            model_version="3.0.0", version="qwen3_8b_v3_loss_ce", data_version="2.0.0")
        planner = SimpleNamespace(predict_rows=lambda subset: {"target_vector": np.full((len(subset), 1), 7.0)})
        return rows, dev, space, config, planner

    def run_training(self, root, *, config_override=None, dev_override=None):
        rows, dev, space, config, planner = self.fixture(root)
        config.update(config_override or {})
        if dev_override is not None:
            dev = dev_override
        with patch.object(training, "PaperGenerator", TinyLossGenerator), \
             patch("emostance_qwen.paper_planner.load_planner", return_value=planner), \
             patch("emostance_qwen.chat_quality.evaluate_chat_quality", side_effect=AssertionError("No chat evaluator")), \
             patch("emostance_qwen.paper_pipeline.build_quality_report", side_effect=AssertionError("No chat report")):
            return training.train_generator(rows, space, root, config, dev_rows=dev)

    def test_best_uses_token_weighted_predicted_control_dev_ce_not_latest_train_loss(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.run_training(root)
            best = torch.load(root / "generator/best.pt", weights_only=False)
            last = torch.load(root / "generator/last.pt", weights_only=False)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(best["global_step"], 2)
            self.assertEqual(last["global_step"], 3)
            self.assertLess(float(last["projector"]["weight"].square().sum()),
                            float(best["projector"]["weight"].square().sum()))
            self.assertAlmostEqual(result["best_dev_response_ce"], (1 * 2 + 3 * 6) / 8)
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["selection_criterion"], "dev_response_ce")
            self.assertEqual(result["quality_status"], "not_evaluated")
            self.assertEqual(set(TinyLossGenerator.training_ids), {"train-0", "train-1", "train-2"})
            self.assertTrue(all(vector == [7.0] for vector in TinyLossGenerator.observed_dev_vectors))
            self.assertEqual(best["selection_metric"]["control"], "planner_predicted")
            self.assertNotIn("optimizer", best)
            self.assertNotIn("torch_rng", best)
            self.assertTrue(best["inference_only"])
            self.assertIn("optimizer", last)
            selection = json.loads((root / "generator/loss_selection.json").read_text())
            self.assertEqual(selection["best_checkpoint_sha256"], training._file_sha256(root / "generator/best.pt"))
            self.assertEqual(selection["dev_subset_sha256"], training._file_sha256(root / "generator/dev_subset.json"))
            self.assertFalse((root / "generator/quality_dev.json").exists())
            self.assertFalse((root / "generator/quality_baseline_dev.json").exists())

    def test_resume_does_not_repeat_updates_and_preserves_all_training_rngs(self):
        with tempfile.TemporaryDirectory() as full_name, tempfile.TemporaryDirectory() as resume_name:
            full, resumed = Path(full_name), Path(resume_name)
            self.run_training(full)
            self.assertEqual(len(TinyLossGenerator.training_ids), 3)
            TinyLossGenerator.training_ids.clear()
            partial = self.run_training(resumed, config_override={"generator_max_steps": 1})
            self.assertEqual(partial["status"], "incomplete")
            self.assertEqual(partial["global_step"], 1)
            result = self.run_training(resumed)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(len(TinyLossGenerator.training_ids), 3)
            a = torch.load(full / "generator/last.pt", weights_only=False)
            b = torch.load(resumed / "generator/last.pt", weights_only=False)
            self.assertTrue(torch.equal(a["projector"]["weight"], b["projector"]["weight"]))
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            self.assertEqual(a["python_rng"], b["python_rng"])
            self.assertTrue(np.array_equal(a["numpy_rng"][1], b["numpy_rng"][1]))
            self.assertEqual(a["best_step"], b["best_step"])
            self.assertEqual(a["next_epoch"], 1)
            self.assertEqual(b["next_offset"], 0)

    def test_epoch_ends_evaluate_even_when_interval_is_longer_than_epoch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.run_training(root, config_override={"prefix_epochs": 2, "generator_dev_eval_interval": 100,
                "test_dev_ce": {"3": 2., "6": 1.}})
            metrics = [json.loads(line) for line in (root / "generator/loss_metrics.jsonl").read_text().splitlines()]
            self.assertEqual([entry["step"] for entry in metrics], [3, 6])
            self.assertEqual(result["best_step"], 6)
            self.assertEqual(result["best_epoch"], 2)
            self.assertEqual(result["epochs_complete"], 2)

    def test_test_rows_and_changed_dev_data_cannot_select_or_resume_a_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _, dev, _, _, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "development rows"):
                self.run_training(root, dev_override=[{**dev[0], "split": "test"}])
            self.run_training(root, config_override={"generator_max_steps": 1})
            with self.assertRaisesRegex(ValueError, "different data/config"):
                self.run_training(root, dev_override=[{**dev[0], "response": "changed"}, dev[1]])

    def test_slim_best_is_not_accepted_as_a_resume_checkpoint_when_last_is_missing(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.run_training(root, config_override={"generator_max_steps": 1})
            (root / "generator/last.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "inference-only best.pt"):
                self.run_training(root)

    def test_pure_ce_rejects_accidental_quality_kl_anchor_or_filter_settings(self):
        overrides = [{"chat_quality_enabled": True}, {"prefix_teacher_kl_weight": 1},
                     {"language_preservation_weight": 1}, {"prefix_ce_max_response_chars": 40},
                     {"prefix_ce_exclude_persona_claims": True}]
        for override in overrides:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as name:
                with self.assertRaises(ValueError):
                    self.run_training(Path(name), config_override=override)

    def test_loss_only_does_not_read_old_quality_or_anchor_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.run_training(root, config_override={
                "chat_quality_dev_file": str(root / "missing-chat.jsonl"),
                "language_preservation_train_file": str(root / "missing-anchors.jsonl")})
            self.assertEqual(result["status"], "complete")

    def test_nonfinite_development_loss_saves_resumable_last_but_never_selects_best(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with self.assertRaisesRegex(RuntimeError, "Development reply CE is nonfinite"):
                self.run_training(root, config_override={"test_dev_ce": {"1": float("nan")}})
            last = torch.load(root / "generator/last.pt", weights_only=False)
            self.assertEqual(last["global_step"], 1)
            self.assertFalse((root / "generator/best.pt").exists())

    def test_smoke_api_forces_single_step_mode_without_dev_selection(self):
        with patch.object(training, "train_generator", return_value={"status": "complete"}) as train:
            result = training.smoke_train_step({"sample_id": "one"}, object(), "/tmp/smoke", {
                "generator_selection_mode": "dev_response_ce", "prefix_epochs": 3, "chat_quality_enabled": False})
        self.assertEqual(result["status"], "complete")
        smoke = train.call_args.args[3]
        self.assertEqual(smoke["generator_selection_mode"], "smoke_only")
        self.assertEqual(smoke["generator_max_steps"], 1)
        self.assertEqual(smoke["prefix_epochs"], 1)

    def test_smoke_api_with_zero_kl_runs_one_actual_cpu_optimizer_update(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows, _, space, config, _ = self.fixture(root)
            with patch.object(training, "PaperGenerator", TinyLossGenerator):
                result = training.smoke_train_step(rows[0], space, root, config)
            self.assertEqual(result["global_step"], 1)
            self.assertEqual(result["selection_criterion"], "smoke_only")
            self.assertEqual(len(TinyLossGenerator.training_ids), 1)
            self.assertFalse((root / "generator/dev_subset.json").exists())


if __name__ == "__main__":
    unittest.main()
