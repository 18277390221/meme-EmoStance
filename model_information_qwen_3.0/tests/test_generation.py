"""Real CPU tensor checks with fake tokenizer/model; never load a base model."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import importlib
import unittest

import torch
from torch import nn

from model_information_qwen_v3.generation import LocalChatGenerator
from model_information_qwen_v3.model import _runtime
PaperGenerator = importlib.import_module(f"{_runtime()}.paper_generator").PaperGenerator


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.calls = []

    def ids(self, text):
        result = []
        for char in text:
            if char not in self.char_to_id:
                index = len(self.char_to_id) + 10
                self.char_to_id[char] = index
                self.id_to_char[index] = char
            result.append(self.char_to_id[char])
        return result

    def render(self, messages):
        return "".join(f"[{item['role']}]{item['content']}" for item in messages)

    def apply_chat_template(self, messages, **options):
        self.calls.append({"messages": copy.deepcopy(messages), "options": dict(options)})
        if options != {"tokenize": True, "return_dict": True, "return_tensors": "pt", "add_generation_prompt": True, "enable_thinking": False}:
            raise AssertionError("Chat must tokenize its official template directly")
        ids = torch.tensor([[1] + self.ids(self.render(messages))], dtype=torch.long)
        self.calls[-1]["encoded_ids"] = ids.clone()
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens):
        return "".join(self.id_to_char.get(int(index), "") for index in ids if int(index) != self.eos_token_id)


class TinyModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.embedding = nn.Embedding(2000, 4)
        self.requires_grad_(False)
        self.config = SimpleNamespace(max_position_embeddings=10000)
        self.tokenizer = tokenizer
        self.calls = []
        self.output_text = "回答包含 A: 与 B:，保留它们。"
        self.after_eos = ""

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **options):
        self.calls.append({**options, "inference_mode": torch.is_inference_mode_enabled(), "grad_enabled": torch.is_grad_enabled()})
        continuation = self.tokenizer.ids(self.output_text) + [self.tokenizer.eos_token_id] + self.tokenizer.ids(self.after_eos)
        generated = torch.tensor([continuation], dtype=torch.long)
        return torch.cat([options["input_ids"], generated], dim=1) if "input_ids" in options else generated


class TinyProjector(nn.Module):
    prefix_length = 2
    vector_dim = 3

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, vector):
        return self.linear(vector.float()).view(1, 2, 4)


class TinyRawGenerator(PaperGenerator):
    def __init__(self):
        self.config = {"system_prompt": "你是温和的中文聊天伙伴。", "max_prompt_tokens": 4096}
        self.device = torch.device("cpu")
        self.tokenizer = TinyTokenizer()
        self.model = TinyModel(self.tokenizer)
        self.projector = TinyProjector()
        self.stance_calls = 0

    def _stance(self, vector):
        self.stance_calls += 1
        value = torch.as_tensor(vector, dtype=torch.float32).reshape(1, -1)
        if value.shape[1] != 3 or not torch.isfinite(value).all():
            raise ValueError("Expected a finite 3-dimensional stance")
        return value


class LocalChatGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.raw = TinyRawGenerator()
        self.generator = LocalChatGenerator(self.raw)
        self.messages = [{"role": "user", "content": "我不想去。"}]
        self.system = "你是温和的中文聊天伙伴。"

    def complete(self, **options):
        return self.generator.complete(self.system, self.messages, max_new_tokens=64, **options)

    def test_shares_existing_model_tokenizer_projector_without_new_weights(self):
        self.assertIs(self.generator.model, self.raw.model)
        self.assertIs(self.generator.tokenizer, self.raw.tokenizer)
        self.assertIs(self.generator.projector, self.raw.projector)

    def test_plain_generation_strips_prompt_and_does_not_use_emotion_prefix(self):
        result = self.complete()
        self.assertEqual(result, self.raw.model.output_text)
        call = self.raw.model.calls[-1]
        self.assertIn("input_ids", call)
        self.assertNotIn("inputs_embeds", call)
        self.assertEqual(self.raw.stance_calls, 0)
        self.assertFalse(call["do_sample"])
        self.assertEqual(call["no_repeat_ngram_size"], 0)
        self.assertNotIn("temperature", call)
        self.assertNotIn("top_p", call)
        self.assertTrue(call["inference_mode"])
        self.assertFalse(call["grad_enabled"])

    def test_prefix_generation_retains_real_projector_and_decodes_all_continuation(self):
        vector = [.1, .2, .3]
        result = self.complete(stance_vector=vector)
        self.assertEqual(result, self.raw.model.output_text)
        call = self.raw.model.calls[-1]
        self.assertNotIn("input_ids", call)
        self.assertIn("inputs_embeds", call)
        self.assertEqual(self.raw.stance_calls, 1)
        expected = self.raw.projector(torch.tensor([vector])).detach()
        self.assertTrue(torch.allclose(call["inputs_embeds"][:, :2], expected))
        self.assertEqual(call["inputs_embeds"].shape[:2], call["attention_mask"].shape)
        self.assertTrue(all(parameter.grad is None for parameter in self.raw.projector.parameters()))

    def test_prefix_is_cast_to_existing_embedding_dtype(self):
        self.raw.model.embedding.half()
        self.complete(stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.model.calls[-1]["inputs_embeds"].dtype, torch.float16)

    def test_leading_system_is_present_in_actual_template_input(self):
        self.complete()
        call = self.raw.tokenizer.calls[-1]
        self.assertEqual(call["messages"], [{"role": "system", "content": self.system}, *self.messages])
        self.assertTrue(call["options"]["add_generation_prompt"])
        self.assertIs(call["options"]["enable_thinking"], False)
        self.assertEqual(len(self.raw.model.calls), 1)
        self.assertEqual(self.raw.model.calls[0]["num_return_sequences"], 1)
        self.assertEqual(self.raw.model.calls[0]["num_beams"], 1)

    def test_budget_discards_oldest_whole_pair_and_keeps_latest_negation(self):
        history = [{"role": "user", "content": "旧问题" * 20}, {"role": "assistant", "content": "旧回答" * 20},
                   {"role": "user", "content": "最近问题"}, {"role": "assistant", "content": "最近回答"}, *self.messages]
        original = copy.deepcopy(history)
        kept = [{"role": "system", "content": self.system}, *history[2:]]
        budget = len(self.raw.tokenizer.ids(self.raw.tokenizer.render(kept))) + 1
        adapter = LocalChatGenerator(self.raw, max_prompt_tokens=budget)
        adapter.complete(self.system, history, max_new_tokens=64)
        self.assertEqual(self.raw.tokenizer.calls[-1]["messages"], kept)
        self.assertEqual(self.raw.tokenizer.calls[-1]["messages"][-1]["content"], "我不想去。")
        self.assertEqual(history, original)

    def test_system_and_current_user_over_budget_raise_without_model_generation(self):
        adapter = LocalChatGenerator(self.raw, max_prompt_tokens=10)
        with self.assertRaisesRegex(ValueError, "system prompt exceed"):
            adapter.complete(self.system, self.messages, max_new_tokens=64)
        self.assertEqual(self.raw.model.calls, [])

    def test_actual_model_context_reserves_prefix_and_output_space(self):
        self.raw.model.config.max_position_embeddings = 64
        with self.assertRaisesRegex(ValueError, "No space for the prompt"):
            self.complete(stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.model.calls, [])

    def test_normal_speaker_labels_survive_qwen_decoding(self):
        self.assertEqual(self.complete(), self.raw.model.output_text)
        self.assertEqual(self.complete(stance_vector=[1, 2, 3]), self.raw.model.output_text)

    def test_empty_model_completion_is_an_error(self):
        self.raw.model.output_text = ""
        with self.assertRaisesRegex(RuntimeError, "empty reply"):
            self.complete()

    def test_unexpected_reasoning_is_rejected_without_regeneration(self):
        self.raw.model.output_text = "<think>内部推理</think>回答"
        with self.assertRaisesRegex(RuntimeError, "reasoning"):
            self.complete(stance_vector=[1, 2, 3])
        self.assertEqual(len(self.raw.model.calls), 1)

    def test_sampling_settings_are_explicit_and_reply_repeat_option_is_retained(self):
        self.complete(do_sample=True, temperature=.8, top_p=.95, no_repeat_ngram_size=4)
        call = self.raw.model.calls[-1]
        self.assertEqual(call["temperature"], .8)
        self.assertEqual(call["top_p"], .95)
        self.assertEqual(call["no_repeat_ngram_size"], 4)
        self.assertEqual(call["eos_token_id"], self.raw.tokenizer.eos_token_id)

    def test_training_and_serving_share_identical_non_thinking_template(self):
        history = [{"role": "A", "text": self.messages[0]["content"]}]
        training_ids = self.raw.prompt_ids(history, "B")
        self.complete(stance_vector=[1, 2, 3])
        self.assertTrue(torch.equal(training_ids, self.raw.tokenizer.calls[-1]["encoded_ids"]))

    def test_stance_validation_is_delegated_to_original_raw_generator(self):
        for vector in ([1, 2], [1, float("nan"), 3]):
            with self.subTest(vector=vector), self.assertRaisesRegex(ValueError, "finite 3-dimensional"):
                self.complete(stance_vector=vector)

    def test_invalid_roles_and_assistant_ending_are_rejected(self):
        cases = ([{"role": "system", "content": "bad"}],
                 [{"role": "assistant", "content": "bad"}],
                 [self.messages[0], {"role": "user", "content": "bad"}],
                 [self.messages[0], {"role": "assistant", "content": "unfinished"}])
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.generator.complete(self.system, messages, max_new_tokens=64)

    def test_invalid_generation_arguments_are_rejected_before_model_use(self):
        for options in ({"max_new_tokens": True}, {"max_new_tokens": 0}, {"max_new_tokens": 64, "do_sample": "yes"},
                        {"max_new_tokens": 64, "no_repeat_ngram_size": -1}, {"max_new_tokens": 64, "do_sample": True, "temperature": 0},
                        {"max_new_tokens": 64, "do_sample": True, "top_p": 1.1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.generator.complete(self.system, self.messages, **options)
        self.assertEqual(self.raw.model.calls, [])


if __name__ == "__main__":
    unittest.main()
