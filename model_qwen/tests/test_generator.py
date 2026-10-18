"""Gradient and supervision checks with a real tiny Qwen3 transformer."""
import sys
from pathlib import Path
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emostance_qwen.paper_generator import PaperGenerator, StancePrefixProjector, chat_messages


class GeneratorTrainingTests(unittest.TestCase):
    def test_prefix_has_gradient_through_frozen_qwen_and_only_reply_is_supervised(self):
        torch.manual_seed(42)
        generator = PaperGenerator.__new__(PaperGenerator)
        generator.device = torch.device("cpu")
        generator.model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=128))
        generator.model.requires_grad_(False)
        generator.projector = StancePrefixProjector(4, 16, 2, 8)
        generator.prompt_ids = lambda *args: torch.tensor([[1, 2, 3]])
        generator.response_ids = lambda *args: torch.tensor([[4, 5, 6]])
        captured = {}
        original = generator.model.forward

        def capture(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        generator.model.forward = capture
        loss = generator.loss({"history": [{"role": "A", "text": "你好"}],
                               "next_role": "B", "response": "你好呀"}, [1, 0, 0, 0])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(captured["labels"].tolist(), [[-100] * 5 + [4, 5, 6]])
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in generator.projector.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in generator.model.parameters()))

    def test_both_training_directions_keep_observed_speaker_as_user(self):
        rows = [{"role": "A", "text": "甲"}, {"role": "B", "text": "乙"}]
        self.assertEqual(chat_messages(rows[:1], "B"), [{"role": "user", "content": "甲"}])
        self.assertEqual(chat_messages(rows, "A"), [{"role": "assistant", "content": "甲"},
                                                 {"role": "user", "content": "乙"}])


if __name__ == "__main__":
    unittest.main()
