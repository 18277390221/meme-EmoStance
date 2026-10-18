"""Training selection must follow actual dev chat scores rather than train CE."""
import hashlib
import json
from pathlib import Path
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


class TinyGenerator:
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.base_model_path = Path(config["base_model_path"])
        self.device = torch.device("cpu")
        self.model = nn.Linear(1, 1)
        self.model.requires_grad_(False)
        self.model.config = SimpleNamespace(use_cache=True)
        self.projector = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projector.weight.fill_(1.0)
        self.checkpoint_state = None
        if checkpoint:
            self.checkpoint_state = torch.load(checkpoint, weights_only=False)
            self.projector.load_state_dict(self.checkpoint_state["projector"])

    def architecture(self):
        return {"vector_dim": 1}

    def distillation_losses(self, row, vector, temperature):
        value = self.projector.weight.square().sum()
        return {"ce": value, "teacher_kl": value * .1}

    def loss(self, row, vector):
        return self.projector.weight.square().sum()

    def response_ids(self, text):
        return torch.tensor([[1, 2]])


class TrainingSelectionTests(unittest.TestCase):
    def fixture(self, root):
        rows = [{"sample_id": str(i), "split": "train", "history": [{"role": "A", "text": "你好"}],
                 "next_role": "B", "response": "你好呀", "target_meme_distribution": {"m": 1.0}}
                for i in range(3)]
        space = SimpleNamespace(vector_dim=1, meme_ids=["m"], centroids=np.ones((1, 1)),
            membership=np.ones((1, 1)), prototypes=np.ones((1, 1)),
            targets=lambda labels: (np.ones(1), np.ones(1)))
        config = {"run_dir": str(root), "base_model_path": str(root / "backbone"),
            "prefix_epochs": 1, "prefix_learning_rate": .05, "gradient_checkpointing": False,
            "chat_quality_enabled": True, "chat_quality_dev_file": str(root / "dev.jsonl"),
            "chat_quality_interval": 1, "prefix_teacher_kl_weight": 1.0,
            "prefix_dev_subset_size": 1, "generator_runtime_checkpoint_interval": 100}
        (root / "dev.jsonl").write_text('{}\n')
        dev = [{**rows[0], "split": "dev"}]
        planner = SimpleNamespace(predict_rows=lambda rows: {"target_vector": np.ones((len(rows), 1))})
        return rows, space, config, dev, planner

    def run_training(self, root, qualities, with_anchor=False):
        rows, space, config, dev, planner = self.fixture(root)
        if with_anchor:
            anchor = {"sample_id": "anchor", "split": "preservation_train", "history": [{"role": "A", "text": "你是谁？"}],
                      "next_role": "B", "response": "我是meme小猫。"}
            path = root / "anchors.jsonl"
            path.write_text(json.dumps(anchor, ensure_ascii=False) + "\n", encoding="utf-8")
            config.update(language_preservation_train_file=str(path), language_preservation_interval=1,
                          language_preservation_ce_weight=1.0)
            planner.predict = lambda history, role: {"target_vector": np.ones(1)}
        outcomes = iter(qualities)

        def quality_report(*args, **kwargs):
            score, passed = next(outcomes)
            return {"status": "passed" if passed else "rejected", "candidate": {"score": score},
                    "comparison": {"reasons": []}, "checkpoint_sha256": "placeholder"}

        def bind(report, config, phase, checkpoint):
            return {**report, "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}

        with patch.object(training, "PaperGenerator", TinyGenerator), \
             patch("emostance_qwen.paper_planner.load_planner", return_value=planner), \
             patch("emostance_qwen.chat_quality.load_cases", return_value=[]), \
             patch("emostance_qwen.chat_quality.evaluate_chat_quality", return_value={"score": .5}), \
             patch("emostance_qwen.paper_pipeline.build_quality_report", side_effect=quality_report), \
             patch("emostance_qwen.paper_pipeline.bind_quality_report", side_effect=bind):
            result = training.train_generator(rows, space, root, config, dev_rows=dev)
        return result

    def test_lower_training_loss_does_not_replace_higher_chat_quality(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.run_training(root, [(1.0, True), (.8, True), (.7, True)])
            best = torch.load(root / "generator/best.pt", weights_only=False)
            last = torch.load(root / "generator/last.pt", weights_only=False)
            self.assertEqual(best["global_step"], 1)
            self.assertEqual(last["global_step"], 3)
            self.assertLess(float(last["projector"]["weight"].square().sum()),
                            float(best["projector"]["weight"].square().sum()))
            self.assertEqual(result["selection_criterion"], "chat_quality_dev_then_dev_response_ce")
            self.assertEqual(result["quality_status"], "dev_passed")

    def test_all_rejected_candidates_never_write_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = self.run_training(root, [(.9, False), (.95, False), (1.0, False)])
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["quality_status"], "rejected")
            self.assertIsNone(result["best_checkpoint"])
            self.assertFalse((root / "generator/best.pt").exists())
            self.assertTrue((root / "generator/last.pt").exists())

    def test_chinese_corrective_anchors_train_both_supervision_and_teacher_kl(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            self.run_training(root, [(1.0, True)] * 3, with_anchor=True)
            rows = [json.loads(line) for line in (root / "generator/progress.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["anchor_ce"] > 0 and row["anchor_teacher_kl"] > 0 for row in rows))
            for row in rows:
                self.assertAlmostEqual(row["loss"], row["reply_ce"] + row["teacher_kl"] +
                                       row["anchor_ce"] + row["anchor_teacher_kl"], places=5)


if __name__ == "__main__":
    unittest.main()
