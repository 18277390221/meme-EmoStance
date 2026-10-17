import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from emostance_meme.backbone import LocalMistral
from emostance_meme.engine import MemeEngine
from emostance_meme.networks import OrientationPredictor, PrefixProjector, reply_loss, soft_cross_entropy
from emostance_meme.retrieval import ResponseIndex


class CapturingLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.decoder = nn.Linear(8, 16)
        self.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask, labels, use_cache):
        self.labels = labels.detach().clone()
        self.attention = attention_mask.detach().clone()
        # Cumulative context gives the prefix a causal path into reply loss.
        logits = self.decoder(inputs_embeds.cumsum(1))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_soft_confidence_weighted_loss(self):
        logits = torch.tensor([[3., -2.], [-2., 3.]])
        targets = torch.tensor([[1., 0.], [1., 0.]])
        high_first = soft_cross_entropy(logits, targets, torch.tensor([4., 1.]))
        high_second = soft_cross_entropy(logits, targets, torch.tensor([1., 4.]))
        self.assertLess(high_first, high_second)

    def test_role_aware_soft_probabilities_and_single_region(self):
        for clusters in [1, 3]:
            predictor = OrientationPredictor(8, 5, torch.randn(clusters, 4), torch.full((clusters, clusters), 1. / clusters))
            output = predictor(torch.randn(2, 8), torch.randn(2, 8))
            self.assertEqual(output['vector'].shape, (2, 4))
            self.assertTrue(torch.isfinite(output['vector']).all())
            torch.testing.assert_close(output['probability'].sum(-1), torch.ones(2))

    def test_only_reply_and_eos_have_labels_and_lm_stays_frozen(self):
        model = CapturingLM()
        prefix = PrefixProjector(3, 6, 8, 2)
        loss = reply_loss(model, prefix, torch.tensor([[.1, .2, .3]]), torch.tensor([[1, 3, 4]]), torch.tensor([[5, 6, 2]]))
        loss.backward()
        self.assertEqual(model.labels.tolist(), [[-100] * 5 + [5, 6, 2]])
        self.assertTrue(torch.all(model.attention == 1))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in prefix.parameters()), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_actual_tiny_mistral_inputs_embeds_generate_new_ids_only(self):
        # Construct a tiny random transformer locally; no pretrained files or
        # model downloads, and no training command is executed by this test.
        from transformers import MistralConfig, MistralForCausalLM
        model = MistralForCausalLM(MistralConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=None)).eval()
        ids = model.generate(inputs_embeds=torch.randn(1, 5, 16), attention_mask=torch.ones(1, 5, dtype=torch.long),
                             max_new_tokens=2, do_sample=False, pad_token_id=0, eos_token_id=None)
        self.assertEqual(ids.shape, (1, 2))

    def test_meme_comes_from_generated_reply_emotion_and_keeps_new_text(self):
        class FakeBackbone:
            device = torch.device('cpu')
            def features(self, conversations):
                return np.array([[1., 0.]], dtype=np.float32)
            def generate(self, *args, **kwargs):
                return '模型新生成的回复'
        class FakePredictor:
            def __call__(self, *args):
                return {'vector': torch.tensor([[1., 0.]])}
            def utterance(self, features):
                return {'vector': torch.tensor([[0., 1.]])}
        engine = MemeEngine.__new__(MemeEngine)
        engine._lock = threading.Lock()
        engine.backbone = FakeBackbone()
        engine.predictor = FakePredictor()
        engine.projector = object()
        engine.config = {'max_reply_tokens': 8, 'do_sample': False, 'temperature': .7, 'top_p': .9}
        engine.index = ResponseIndex([
            {'reply': '原始目标对应的旧回复', 'meme_id': 'meme-0001', 'vector': [1., 0.]},
            {'reply': '生成情绪最近的旧回复', 'meme_id': 'meme-0002', 'vector': [0., 1.]},
        ])
        result = engine.reply([{'role': 'user', 'content': '你好'}])
        self.assertEqual(result, {'reply': '模型新生成的回复', 'meme_id': 'meme-0002'})


if __name__ == '__main__':
    unittest.main()
