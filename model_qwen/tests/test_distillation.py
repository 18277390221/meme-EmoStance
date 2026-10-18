"""CPU regressions for aligned teacher KL and reproducible training inputs."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emostance_qwen.paper_generator import (
    PaperGenerator, StancePrefixProjector, _training_signature, response_ce_weight,
)


def tiny_generator(response=(4, 5, 6)):
    # Local construction never loads a backbone or initializes a CUDA device.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        generator = PaperGenerator.__new__(PaperGenerator)
        generator.device = torch.device("cpu")
        generator.model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=128, attention_dropout=0.0))
        generator.model.requires_grad_(False)
        generator.model.eval()
        generator.projector = StancePrefixProjector(4, 16, 2, 8)
    generator.prompt_ids = lambda *args: torch.tensor([[1, 2, 3]])
    generator.response_ids = lambda *args: torch.tensor([list(response)])
    return generator


ROW = {"history": [{"role": "A", "text": "你好"}],
       "next_role": "B", "response": "你好呀"}
VECTOR = [1.0, 0.0, 0.0, 0.0]


class DistillationLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def test_reply_ce_matches_full_masked_loss_including_first_token_and_eos(self):
        for response in ((6,), (4, 5, 6)):
            with self.subTest(response=response):
                generator = tiny_generator(response)
                expected = generator.loss(ROW, VECTOR)
                actual = generator.distillation_losses(ROW, VECTOR)["ce"]
                torch.testing.assert_close(actual, expected)

    def test_teacher_kl_matches_explicit_full_sequence_reply_positions(self):
        generator = tiny_generator()
        prompt = generator.prompt_ids()
        response = generator.response_ids()
        ids = torch.cat([prompt, response], dim=1)
        prefix_length, prompt_length = generator.projector.prefix_length, prompt.shape[1]
        count = response.shape[1]
        with torch.no_grad():
            teacher_logits = generator.model(input_ids=ids, use_cache=False).logits
            teacher_logits = teacher_logits[:, prompt_length - 1:prompt_length + count - 1]
            embeds = generator.model.get_input_embeddings()(ids)
            prefix = generator.projector(generator._stance(VECTOR))
            student_logits = generator.model(inputs_embeds=torch.cat([prefix, embeds], dim=1),
                                               use_cache=False).logits
            student_logits = student_logits[:, prefix_length + prompt_length - 1:
                                               prefix_length + prompt_length + count - 1]
        for temperature in (1.0, 2.0):
            with self.subTest(temperature=temperature):
                expected = F.kl_div(
                    (student_logits.float() / temperature).log_softmax(-1),
                    (teacher_logits.float() / temperature).log_softmax(-1),
                    log_target=True, reduction="sum") * temperature ** 2 / count
                actual = generator.distillation_losses(ROW, VECTOR, temperature)["teacher_kl"]
                self.assertTrue(torch.isfinite(actual))
                self.assertGreaterEqual(float(actual.detach()), 0.0)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-7)

    def test_distillation_backpropagates_to_prefix_without_base_gradients(self):
        generator = tiny_generator()
        losses = generator.distillation_losses(ROW, VECTOR)
        (losses["ce"] + losses["teacher_kl"]).backward()
        self.assertTrue(all(parameter.grad is None for parameter in generator.model.parameters()))
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in generator.projector.parameters()))
        self.assertGreater(sum(float(parameter.grad.abs().sum())
                               for parameter in generator.projector.parameters()), 0.0)

    def test_checkpointing_allows_main_backward_then_independent_anchor_kl_backward(self):
        generator = tiny_generator()
        generator.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        generator.model.train()
        main = generator.distillation_losses(ROW, VECTOR)
        (main["ce"] + main["teacher_kl"]).backward()
        before = [parameter.grad.clone() for parameter in generator.projector.parameters()]
        anchor = generator.distillation_losses(ROW, [0.0, 1.0, 0.0, 0.0])
        anchor["teacher_kl"].backward()
        after = [parameter.grad for parameter in generator.projector.parameters()]
        self.assertTrue(any(not torch.equal(first, second) for first, second in zip(before, after)))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in after))
        self.assertTrue(all(parameter.grad is None for parameter in generator.model.parameters()))
        self.assertTrue(generator.model.training)
        self.assertTrue(generator.model.is_gradient_checkpointing)


class CompatibleReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {"prefix_ce_max_response_chars": 40,
                       "prefix_ce_exclude_persona_claims": True}

    def test_long_reference_and_conflicting_identity_disable_only_hard_reference_ce(self):
        examples = (("长" * 41, "exceeds_role_length"),
                    ("我的名字是小明。", "conflicting_name_claim"),
                    ("我叫小王。", "conflicting_name_claim"),
                    ("我是ChatGPT。", "conflicting_identity_claim"),
                    ("我是人类。", "conflicting_identity_claim"))
        for response, reason in examples:
            with self.subTest(response=response):
                row = {"response": response}
                original = dict(row)
                self.assertEqual(response_ce_weight(row, self.config), (0.0, reason))
                self.assertEqual(row, original)

    def test_meme_and_causative_utterances_keep_hard_reference_ce(self):
        for response in ("我叫meme。", "我是meme，你的桌面小猫。", "我叫你先喝口水呀。",
                         "我叫他明天再来。", "我叫大家一起休息。", "短" * 40):
            with self.subTest(response=response):
                self.assertEqual(response_ce_weight({"response": response}, self.config),
                                 (1.0, "compatible_reference"))


class TrainingSignatureTests(unittest.TestCase):
    def test_signature_tracks_ce_policy_development_rows_and_planner_weights(self):
        with tempfile.TemporaryDirectory(prefix="qwen_signature_test_") as temporary:
            root = Path(temporary)
            (root / "planner").mkdir()
            planner = root / "planner/best.pt"
            planner.write_bytes(b"planner checkpoint one")
            config = {"run_dir": str(root), "base_model_path": str(root / "base"),
                      "prefix_ce_max_response_chars": 40,
                      "prefix_ce_exclude_persona_claims": True}
            space = SimpleNamespace(meme_ids=["m1"], centroids=[[1.0]],
                                    membership=[[1.0]], prototypes=[[1.0]])
            rows = [{"sample_id": "train1", **deepcopy(ROW), "target_meme_distribution": {"m1": 1.0}}]
            dev = [{"sample_id": "dev1", "split": "dev", **deepcopy(ROW)}]
            original = _training_signature(rows, space, config, dev)
            for change in ({"prefix_ce_max_response_chars": 39},
                           {"prefix_ce_exclude_persona_claims": False}):
                with self.subTest(change=change):
                    self.assertNotEqual(_training_signature(rows, space, {**config, **change}, dev), original)
            changed_dev = deepcopy(dev)
            changed_dev[0]["response"] = "不同的开发集回复"
            self.assertNotEqual(_training_signature(rows, space, config, changed_dev), original)
            planner.write_bytes(b"planner checkpoint two")
            self.assertNotEqual(_training_signature(rows, space, config, dev), original)


if __name__ == "__main__":
    unittest.main()
