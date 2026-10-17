"""Independent train-only space and paper-controller invariants."""

import copy
from contextlib import redirect_stdout
import io
import json
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from torch import nn

from emostance_meme.paper_space import ChineseHashedTfidf, build_space
from emostance_meme.paper_planner import PaperOrientationPlanner, serialize_row, training_statistics
from emostance_meme.paper_generator import PaperGenerator, StancePrefixProjector, generation_prompt
from emostance_meme.space import EmotionSpace
from emostance_meme.paper_engine import PaperMemeEngine, PaperResponseIndex, build_response_index, distribution_metrics


class PaperSpaceAuditTests(unittest.TestCase):
    @staticmethod
    def inputs():
        votes = [(["m1", "m1", "m2", "m2"], [5, 5, 1, 1]),
                 (["m1", "m2", "m2", "m2"], [1, 3, 3, 3]),
                 (["m2", "m2", "m3", "m3"], [5, 5, 2, 2])]
        turns = []
        for i, (memes, confidences) in enumerate(votes):
            total = sum(confidences)
            distribution = {m: sum(c for m2, c in zip(memes, confidences) if m2 == m) / total for m in set(memes)}
            turns.append({"dialogue_id": "d", "turn_id": i, "split": "train", "text": ["今天开心", "一起庆祝", "好呀谢谢"][i],
                          "meme_distribution": distribution, "weight": total / 20,
                          "annotations": [{"annotator": f"paper{j+1}", "meme_id": m, "confidence": c, "reason": "ignored"} for j, (m, c) in enumerate(zip(memes, confidences))]})
        samples = [{"split": "train", "target_meme_distribution": turns[i]["meme_distribution"]} for i in [1, 2]]
        return turns, samples

    def test_train_only_graph_rejects_eval_turns_and_eval_prototypes(self):
        turns, samples = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            turns[0]["split"] = "dev"
            with self.assertRaisesRegex(ValueError, "training utterances"):
                build_space(turns, samples, temp, {})
            turns[0]["split"] = "train"
            samples[0]["split"] = "test"
            with self.assertRaisesRegex(ValueError, "training samples"):
                build_space(turns, samples, temp, {})

    def test_raw_vote_shrinkage_and_confidence_centroid_match_reference_equations(self):
        turns, samples = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            space, _ = build_space(turns, samples, temp, {"vector_dim": 32, "centroid_shrinkage_tau": 50, "seed": 42})
            arrays = np.load(Path(temp) / "graph/graph_arrays.npz")
            # Raw supporter counts are independent of confidence and differ
            # from both the number of nonzero utterances and the soft mass.
            np.testing.assert_array_equal(arrays["counts"], [3, 7, 2])
            np.testing.assert_allclose(arrays["shrinkage_alpha"], np.asarray([3, 7, 2]) / np.asarray([53, 57, 52]))
            q = np.asarray([[turn["meme_distribution"].get(m, 0) for m in space.meme_ids] for turn in turns])
            features = ChineseHashedTfidf(32).fit_transform([turn["text"] for turn in turns])
            weights = q * np.asarray([turn["weight"] for turn in turns])[:, None]
            raw = (weights.T @ features) / weights.sum(axis=0)[:, None]
            alpha = arrays["shrinkage_alpha"][:, None]
            expected = alpha * raw + (1-alpha) * features.mean(axis=0)
            np.testing.assert_allclose(space.centroids, expected, atol=1e-12)
            np.testing.assert_allclose(arrays["effective_counts"], weights.sum(axis=0))

    def test_cluster_prototypes_use_training_target_soft_mass_without_extra_confidence(self):
        turns, samples = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            space, _ = build_space(turns, samples, temp, {"vector_dim": 32, "seed": 42})
            votes = np.asarray([[row["target_meme_distribution"].get(m, 0) for m in space.meme_ids] for row in samples])
            target_cluster = votes @ space.membership
            target_vector = votes @ space.centroids
            mass = target_cluster.sum(axis=0)
            expected = (target_cluster.T @ target_vector) / np.maximum(mass[:, None], 1e-12)
            expected[mass <= 1e-12] = target_vector.mean(axis=0)
            np.testing.assert_allclose(space.prototypes, expected, atol=1e-12)
            np.testing.assert_allclose(space.membership.sum(axis=1), 1)

    def test_reason_text_cannot_influence_graph_or_prototypes(self):
        turns, samples = self.inputs()
        other = copy.deepcopy(turns)
        for turn in other:
            for annotation in turn["annotations"]:
                annotation["reason"] = "different annotation rationale"
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a, _ = build_space(turns, samples, first, {"seed": 42})
            b, _ = build_space(other, samples, second, {"seed": 42})
            for name in ("centroids", "membership", "prototypes"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_chinese_features_and_idf_do_not_refit_on_evaluation_text(self):
        encoder = ChineseHashedTfidf(256)
        encoder.fit_transform(["今天开心", "明天见"])
        old_idf = encoder.idf.copy()
        vector = encoder.transform(["全新验证句子"])
        self.assertEqual(vector.shape, (1, 256))
        self.assertGreater(np.linalg.norm(vector), 0)
        np.testing.assert_array_equal(old_idf, encoder.idf)
        punctuation = encoder.transform(["！？"])
        self.assertGreater(np.linalg.norm(punctuation), 0)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(16, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class CapturingFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.decoder = nn.Linear(8, 16)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels, use_cache):
        self.captured_labels = labels.detach().clone()
        self.captured_attention = attention_mask.detach().clone()
        logits = self.decoder(inputs_embeds.cumsum(dim=1))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


class PaperPlannerAuditTests(unittest.TestCase):
    @staticmethod
    def space():
        return EmotionSpace(["m1", "m2"], np.eye(2), np.eye(2), np.asarray([[2., 1.], [4., 3.]]))

    def test_context_serialization_never_contains_reference_reply_or_labels(self):
        row = {"history": [{"role": "A", "text": "用户已说内容"}], "source_role": "A", "next_role": "B",
               "response": "绝不可见的目标回复", "target_meme_distribution": {"m1": 1}}
        text = serialize_row(row)
        self.assertIn("用户已说内容", text)
        self.assertEqual(text, "Dialogue:\nA: 用户已说内容")
        self.assertNotIn("绝不可见", text)
        self.assertNotIn("m1", text)
        self.assertEqual(generation_prompt(row["history"], "B"), "Dialogue:\nA: 用户已说内容\nB:")

    def test_source_uncertainty_disables_transition_prior_and_vectors_use_prototypes(self):
        torch.manual_seed(1)
        space = self.space()
        transitions = np.tile(np.asarray([[.95, .05], [.9, .1]], dtype=np.float32), (4, 1, 1))
        planner = PaperOrientationPlanner(TinyEncoder(), space, transitions, role_dim=2, dropout=0).eval()
        with torch.no_grad():
            planner.source_cluster.weight.zero_()
            planner.source_cluster.bias.zero_()
        output = planner(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long), torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))
        torch.testing.assert_close(output["source_distribution"], torch.tensor([[.5, .5]]))
        torch.testing.assert_close(output["gate"], torch.zeros_like(output["gate"]), atol=1e-7, rtol=0)
        torch.testing.assert_close(output["target_logits"], output["target_text_logits"], atol=1e-7, rtol=0)
        torch.testing.assert_close(output["target_vector"], output["target_distribution"] @ torch.tensor(space.prototypes, dtype=torch.float32))
        self.assertTrue(torch.isfinite(output["target_vector"]).all())

    def test_transition_counts_keep_ordered_roles_and_smoothing_normalized(self):
        rows = [{"split": "train", "history": [{"role": "A", "text": "起始"}], "next_role": "B", "source_meme_distribution": {"m1": 1}, "target_meme_distribution": {"m2": 1}},
                {"split": "train", "history": [{"role": "B", "text": "反向"}], "next_role": "A", "source_meme_distribution": {"m2": 1}, "target_meme_distribution": {"m1": 1}}]
        stats = training_statistics(rows, self.space())
        np.testing.assert_allclose(stats["transitions"].sum(axis=-1), 1)
        self.assertGreater(stats["transitions"][0, 0, 1], .9)
        self.assertGreater(stats["transitions"][1, 1, 0], .9)
        np.testing.assert_allclose(stats["transitions"][2:], .5)
        self.assertAlmostEqual(float(stats["source_weights"].mean()), 1)
        self.assertAlmostEqual(float(stats["target_weights"].mean()), 1)

    def test_eval_rows_cannot_estimate_transition_statistics(self):
        row = {"split": "test", "history": [{"role": "A", "text": "验证原文"}], "next_role": "B", "source_meme_distribution": {"m1": 1}, "target_meme_distribution": {"m2": 1}}
        with self.assertRaisesRegex(ValueError, "only use train"):
            training_statistics([row], self.space())

    def test_single_cluster_is_finite_and_realized_source_uses_saved_source_prototypes(self):
        space = EmotionSpace(["m1"], np.asarray([[1., 0.]]), np.ones((1, 1)), np.asarray([[3., 2.]]))
        planner = PaperOrientationPlanner(TinyEncoder(), space, np.ones((4, 1, 1), dtype=np.float32), role_dim=2, dropout=0,
                                          source_prototypes=np.asarray([[7., 9.]])).eval()
        output = planner(torch.tensor([[1, 2]]), torch.ones((1, 2), dtype=torch.long), torch.tensor([0]), torch.tensor([1]), torch.tensor([0]))
        torch.testing.assert_close(output["source_realized_vector"], torch.tensor([[7., 9.]]))
        torch.testing.assert_close(output["target_vector"], torch.tensor([[3., 2.]]))
        self.assertTrue(torch.isfinite(output["gate"]).all())


class PaperPrefixAuditTests(unittest.TestCase):
    def test_response_eos_loss_masks_prompt_prefix_but_backpropagates_through_frozen_lm(self):
        torch.manual_seed(7)
        generator = PaperGenerator.__new__(PaperGenerator)
        generator.config = {}
        generator.device = torch.device("cpu")
        generator.model = CapturingFrozenLM()
        generator.projector = StancePrefixProjector(3, 8, prefix_length=2, intermediate_size=8)
        generator.prompt_ids = lambda history, next_role, situation="": torch.tensor([[1, 2, 3]])
        generator.response_ids = lambda response: torch.tensor([[4, 5]])
        row = {"history": [{"role": "A", "text": "输入"}], "next_role": "B", "response": "真实回复"}
        before = {name: value.detach().clone() for name, value in generator.model.named_parameters()}
        loss = generator.loss(row, [1., 2., 3.])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(generator.model.captured_labels, torch.tensor([[-100, -100, -100, -100, -100, 4, 5]]))
        self.assertEqual(generator.model.captured_attention.tolist(), [[1] * 7])
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in generator.projector.parameters()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in generator.model.parameters()))
        for name, value in generator.model.named_parameters():
            torch.testing.assert_close(before[name], value)


class PaperInferenceAuditTests(unittest.TestCase):
    def test_engine_queries_generated_reply_emotion_and_preserves_generated_text(self):
        calls = []

        class Planner:
            def predict(self, history, next_role):
                calls.append((copy.deepcopy(history), next_role))
                return {"target_vector": np.asarray([1., 0.]), "source_vector": np.asarray([9., 9.]),
                        "source_realized_vector": np.asarray([0., 1.]),
                        "target_distribution": np.asarray([.9, .1]), "source_distribution": np.asarray([.8, .2])}

        class Generator:
            def generate(self, history, next_role, vector, max_new_tokens):
                np.testing.assert_array_equal(vector, [1., 0.])
                return "模型新生成回复"

        class Index:
            def nearest(self, vector):
                # Prototype realization differs deliberately from both the
                # planning vector [1, 0] and direct regression vector [9, 9].
                np.testing.assert_array_equal(vector, [0., 1.])
                return {"reply": "训练库中的旧回复", "meme_id": "m2"}

        engine = PaperMemeEngine.__new__(PaperMemeEngine)
        engine.config = {"max_new_tokens": 64, "do_sample": False}
        engine.planner, engine.generator, engine.index = Planner(), Generator(), Index()
        engine._lock = threading.Lock()
        result = engine.reply([{"role": "user", "content": "用户当前单句"}])
        self.assertEqual(result, {"reply": "模型新生成回复", "meme_id": "m2"})
        self.assertEqual(calls[0], ([{"role": "A", "text": "用户当前单句"}], "B"))
        self.assertEqual(calls[1], ([{"role": "A", "text": "用户当前单句"}, {"role": "B", "text": "模型新生成回复"}], "A"))

    def test_four_candidate_reranking_selects_orientation_match_before_retrieval(self):
        texts = ["第一条不合适", "第二条最贴合", "第三条普通", "第四条次优"]
        distributions = [[.1, .9], [.88, .12], [.6, .4], [.75, .25]]
        calls, generated = [], []

        class Planner:
            def predict(self, history, next_role):
                calls.append((copy.deepcopy(history), next_role))
                if next_role == "B":
                    return {"target_vector": np.asarray([3., 4.]), "target_distribution": np.asarray([.9, .1])}
                index = texts.index(history[-1]["text"])
                return {"source_vector": np.asarray([float(index + 100), 99.]),
                        "source_realized_vector": np.asarray([float(index + 1), 1.]),
                        "source_distribution": np.asarray(distributions[index])}

        class Generator:
            def generate(self, history, next_role, vector, max_new_tokens):
                np.testing.assert_array_equal(vector, [3., 4.])
                text = texts[len(generated)]
                generated.append(text)
                return text

        class Index:
            def nearest(self, vector):
                np.testing.assert_array_equal(vector, [2., 1.])
                return {"reply": "训练锚点文本", "meme_id": "meme-best"}

        engine = PaperMemeEngine.__new__(PaperMemeEngine)
        engine.config = {"num_candidates": 4, "do_sample": True}
        engine.planner, engine.generator, engine.index = Planner(), Generator(), Index()
        engine._lock = threading.Lock()
        result = engine.reply([{"role": "user", "content": "收到好消息"}])
        self.assertEqual(result, {"reply": "第二条最贴合", "meme_id": "meme-best"})
        self.assertEqual(len(generated), 4)
        self.assertEqual([role for _, role in calls], ["B", "A", "A", "A", "A"])
        self.assertTrue(all(history[:-1] == [{"role": "A", "text": "收到好消息"}] for history, _ in calls[1:]))

    def test_jsd_is_symmetric_finite_and_matches_closed_form(self):
        p = np.asarray([[1., 0.], [0., 1.]])
        q = np.asarray([[0., 1.], [1., 0.]])
        metrics = distribution_metrics(p, q)
        self.assertAlmostEqual(metrics["jsd"], np.log(2))
        self.assertEqual(metrics["overlap"], 0)
        self.assertTrue(np.isfinite(metrics["soft_ce"]))
        uniform = np.full((2, 2), .5)
        a, b = distribution_metrics(p, uniform), distribution_metrics(uniform, p)
        self.assertAlmostEqual(a["jsd"], b["jsd"])
        self.assertAlmostEqual(a["jsd"], .21576155433883568)
        self.assertAlmostEqual(a["soft_ce"], np.log(2))
        self.assertAlmostEqual(distribution_metrics(uniform, uniform)["jsd"], 0)
        self.assertEqual(distribution_metrics(uniform, uniform)["overlap"], 1)

    def test_response_bank_is_train_only_and_soft_vote_ties_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            data = Path(temp) / "data"
            root.mkdir()
            data.mkdir()
            (root / "planner").mkdir()
            (root / "planner/best.pt").write_bytes(b"dummy trained planner checkpoint")
            EmotionSpace(["m1", "m2"], np.eye(2), np.eye(2), np.eye(2)).save(root / "space.json")
            rows = [{"sample_id": "d:0->1", "dialogue_id": "d", "target_turn_id": 1, "split": "train",
                     "history": [{"role": "A", "text": "第一个问题"}], "next_role": "B",
                     "response": "训练回复", "target_meme_distribution": {"m2": .5, "m1": .5}},
                    {"sample_id": "d:1->2", "dialogue_id": "d", "target_turn_id": 2, "split": "train",
                     "history": [{"role": "A", "text": "第一个问题"}, {"role": "B", "text": "训练回复"}], "next_role": "A",
                     "response": "后续追问", "target_meme_distribution": {"m2": 1.}}]
            (data / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            # Eval files deliberately exist; the training bank must never read
            # or encode their content, even when the file has valid JSONL.
            for split in ("dev", "test"):
                (data / f"{split}.jsonl").write_text(json.dumps({"split": split, "response": "不可读评测回复"}) + "\n", encoding="utf-8")
            config = {"run_dir": str(root), "data_dir": str(data), "retrieval_metric": "cosine"}
            encoded = []

            class Planner:
                def predict_rows(self, reply_rows):
                    encoded.extend(copy.deepcopy(reply_rows))
                    # Deliberately different from gold label centroid mixtures:
                    # persistence must use the same inferred prototype path as
                    # generated-reply queries rather than gold annotation vectors.
                    return {"source_realized_vector": np.asarray([[1., 2.], [3., 1.]])}

            with patch("emostance_meme.paper_planner.load_planner", return_value=Planner()) as loader, redirect_stdout(io.StringIO()):
                report = build_response_index(config)
            loader.assert_called_once_with(root, config)
            self.assertEqual(encoded, [
                {"history": [{"role": "A", "text": "第一个问题"}, {"role": "B", "text": "训练回复"}], "source_role": "B", "next_role": "A"},
                {"history": [{"role": "A", "text": "第一个问题"}, {"role": "B", "text": "训练回复"}, {"role": "A", "text": "后续追问"}], "source_role": "A", "next_role": "B"},
            ])
            self.assertEqual(report["responses"], 2)
            np.testing.assert_array_equal(np.load(root / "response_vectors.npz")["vectors"], [[1., 2.], [3., 1.]])
            index = PaperResponseIndex(config)
            result = index.nearest([1., 2.])
            self.assertEqual(result["meme_id"], "m1")
            self.assertEqual(result["reply"], "训练回复")
            self.assertEqual(result["split"], "train")
            rows[0]["split"] = "test"
            (data / "train.jsonl").write_text(json.dumps(rows[0], ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training examples only"):
                build_response_index(config)

    def test_index_loader_rejects_space_mismatch_eval_anchors_and_invalid_vectors(self):
        with tempfile.TemporaryDirectory() as temp:
            root, data = Path(temp) / "run", Path(temp) / "data"
            root.mkdir()
            data.mkdir()
            (root / "planner").mkdir()
            checkpoint_path = root / "planner/best.pt"
            checkpoint_path.write_bytes(b"dummy trained planner checkpoint")
            space_path = root / "space.json"
            EmotionSpace(["m1", "m2"], np.eye(2), np.eye(2), np.eye(2)).save(space_path)
            row = {"sample_id": "d:0->1", "dialogue_id": "d", "target_turn_id": 1, "split": "train",
                   "history": [{"role": "A", "text": "原问题"}], "next_role": "B",
                   "response": "训练回复", "target_meme_distribution": {"m1": 1.}}
            (data / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            config = {"run_dir": str(root), "data_dir": str(data)}
            fake = SimpleNamespace(predict_rows=lambda reply_rows: {"source_realized_vector": np.asarray([[2., 3.]])})
            with patch("emostance_meme.paper_planner.load_planner", return_value=fake), redirect_stdout(io.StringIO()):
                build_response_index(config)
            original_space = space_path.read_bytes()
            space_path.write_bytes(original_space + b"\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                PaperResponseIndex(config)
            space_path.write_bytes(original_space)
            original_checkpoint = checkpoint_path.read_bytes()
            checkpoint_path.write_bytes(original_checkpoint + b"changed")
            with self.assertRaisesRegex(ValueError, "planner checkpoint"):
                PaperResponseIndex(config)
            checkpoint_path.write_bytes(original_checkpoint)
            train_path = data / "train.jsonl"
            original_train = train_path.read_bytes()
            train_path.write_bytes(original_train + b"\n")
            with self.assertRaisesRegex(ValueError, "training file"):
                PaperResponseIndex(config)
            train_path.write_bytes(original_train)
            anchors_path = root / "response_anchors.jsonl"
            anchor = json.loads(anchors_path.read_text())
            anchor["split"] = "test"
            anchors_path.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                PaperResponseIndex(config)
            anchor["split"] = "train"
            anchors_path.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
            for invalid in (np.zeros((1, 2)), np.asarray([[float("nan"), 1.]]), np.ones((1, 3)), np.ones((2, 2)), np.ones(2)):
                np.savez_compressed(root / "response_vectors.npz", vectors=invalid)
                with self.assertRaises(ValueError):
                    PaperResponseIndex(config)


if __name__ == "__main__":
    unittest.main()
