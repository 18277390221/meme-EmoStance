"""Prefix training invariants, causal gradients, and exact interrupted resume."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from emostance_meme import paper_generator as paper


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    truncation_side = "left"

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, return_tensors=None):
        ids = ([1] if add_special_tokens else []) + [3 + ord(char) % 29 for char in text]
        if truncation:
            ids = ids[-max_length:]
        return {"input_ids": torch.tensor([ids]) if return_tensors else ids}


class TinyFrozenLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.decoder = nn.Linear(8, 32)
        self.config = SimpleNamespace(use_cache=False)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, labels, use_cache):
        self.labels = labels.detach().clone()
        logits = self.decoder(inputs_embeds.cumsum(1) / 8)
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


class TinyGenerator(paper.PaperGenerator):
    def __init__(self, config, checkpoint=None):
        self.config = config
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.base_model_path = Path("/local/test-backbone")
        self.tokenizer = TinyTokenizer()
        self.model = TinyFrozenLM()
        self.projector = paper.StancePrefixProjector(config["vector_dim"], 8, 2, 6)
        self.checkpoint_state = None
        if checkpoint:
            self.checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.projector.load_state_dict(self.checkpoint_state["projector"])


class TinySpace:
    vector_dim = 3
    meme_ids = ["meme-0002"]
    centroids = np.array([[0.1, 0.2, 0.3]])
    membership = np.ones((1, 1))
    prototypes = centroids

    def targets(self, distribution):
        return np.ones(1), self.centroids[0]


class PaperGeneratorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.config = {"seed": 19, "prefix_epochs": 1, "prefix_learning_rate": 1e-4,
                       "gradient_checkpointing": False, "max_prompt_tokens": 20,
                       "max_training_response_tokens": 3, "vector_dim": 3, "checkpoint_interval": 1}
        self.row = {"sample_id": "d:0->1", "split": "train", "history": [{"role": "A", "text": "你好"}],
                    "next_role": "B", "response": "你也好", "target_meme_distribution": {"meme-0002": 1.0}}

    def test_official_prompt_preserves_roles_and_both_directions(self):
        history = [{"role": "A", "text": "你好"}, {"role": "B", "text": "你也好"}]
        self.assertEqual(paper.generation_prompt(history, "A"), "Dialogue:\nA: 你好\nB: 你也好\nA:")
        self.assertEqual(paper.build_model_text(history, "新年"), "Situation: 新年\nDialogue:\nA: 你好\nB: 你也好")
        with self.assertRaises(ValueError):
            paper.generation_prompt([{"role": "system", "text": "虚构角色"}], "B")

    def test_only_reply_plus_eos_is_supervised_and_frozen_lm_passes_prefix_gradient(self):
        generator = TinyGenerator(self.config)
        loss = generator.loss(self.row, [0.1, 0.2, 0.3])
        loss.backward()
        labels = generator.model.labels[0]
        targets = generator.response_ids(self.row["response"])[0]
        self.assertEqual(labels[-len(targets):].tolist(), targets.tolist())
        self.assertTrue((labels[:-len(targets)] == -100).all())
        self.assertEqual(int(labels[-1]), generator.tokenizer.eos_token_id)
        self.assertTrue(all(parameter.grad is None for parameter in generator.model.parameters()))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in generator.projector.parameters()), 0)

    def test_left_truncation_and_response_content_limit_keep_eos(self):
        generator = TinyGenerator(self.config)
        history = [{"role": "A", "text": "历史" * 100}, {"role": "B", "text": "最近"}]
        ids = generator.prompt_ids(history, "A")[0].tolist()
        untruncated = generator.tokenizer(paper.generation_prompt(history, "A"))["input_ids"]
        self.assertEqual(ids, untruncated[-20:])
        self.assertEqual(len(generator.response_ids("很长的回复文本")[0]), 4)

    def test_resume_matches_uninterrupted_prefix_training_and_checkpoints_exclude_lm(self):
        second = {**self.row, "sample_id": "d:1->2", "next_role": "A", "response": "继续说",
                  "history": self.row["history"] + [{"role": "B", "text": self.row["response"]}]}
        with tempfile.TemporaryDirectory() as temporary, patch.object(paper, "PaperGenerator", TinyGenerator):
            root = Path(temporary)
            full = paper.train_generator([self.row, second], TinySpace(), root / "full", self.config)
            first = paper.train_generator([self.row, second], TinySpace(), root / "resumed", {**self.config, "generator_max_steps": 1})
            self.assertEqual(first["status"], "incomplete")
            self.assertEqual(first["global_step"], 1)
            resumed = paper.train_generator([self.row, second], TinySpace(), root / "resumed", self.config)
            self.assertEqual(full["status"], "complete")
            self.assertEqual(resumed["global_step"], 2)
            states = [torch.load(root / name / "generator/last.pt", weights_only=False) for name in ("full", "resumed")]
            for key in states[0]["projector"]:
                torch.testing.assert_close(states[0]["projector"][key], states[1]["projector"][key], rtol=0, atol=0)
            self.assertNotIn("model", states[0])
            self.assertNotIn("lm", states[0])
            lines = (root / "resumed/generator/progress.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(line)["step"] for line in lines], [1, 2])

    def test_dev_examples_cannot_enter_generator_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "training rows only"):
                paper.train_generator([{**self.row, "split": "dev"}], TinySpace(), temporary, self.config)

    def test_real_tiny_mistral_supports_nonreentrant_checkpointed_prefix_gradients(self):
        from transformers import MistralConfig, MistralForCausalLM
        generator = TinyGenerator(self.config)
        generator.model = MistralForCausalLM(MistralConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
            bos_token_id=1, eos_token_id=2, pad_token_id=0))
        generator.model.requires_grad_(False)
        generator.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        generator.model.train()
        loss = generator.loss(self.row, [0.1, 0.2, 0.3])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in generator.model.parameters()))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in generator.projector.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
