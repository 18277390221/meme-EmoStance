"""Real CPU tensor checks with fake tokenizer/model; never load a base model."""

from __future__ import annotations

import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from model_information.generation import LocalChatGenerator


def generation_prompt(history, next_role):
    """The fake archive exposes the same formatter API as trained archives."""
    return "Dialogue:\n" + "\n".join(f"{item['role']}: {item['text']}" for item in history) + f"\n{next_role}:"


class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.calls = []
        self.text_calls = []

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
        if options != {"tokenize": True, "return_dict": True, "return_tensors": "pt", "add_generation_prompt": True}:
            raise AssertionError("Chat must tokenize its official template directly")
        ids = torch.tensor([[1] + self.ids(self.render(messages))], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def __call__(self, text, **options):
        self.text_calls.append({"text": text, "options": dict(options)})
        if options != {"add_special_tokens": True, "truncation": False, "return_tensors": "pt"}:
            raise AssertionError("Prefix replies must add training BOS without truncating current text")
        ids = torch.tensor([[self.bos_token_id] + self.ids(text)], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens):
        return "".join(self.id_to_char.get(index, "") for index in ids)


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

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 8)

    def forward(self, vector):
        return self.linear(vector.float()).view(1, 2, 4)


class TinyRawGenerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = {"max_prompt_tokens": 384}
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
        system = "" if options.get("stance_vector") is not None else self.system
        return self.generator.complete(system, self.messages, max_new_tokens=64, **options)

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

    def test_disabled_system_preserves_dialogue_and_trained_prefix(self):
        history = [{"role": "user", "content": "你睡了吗"},
                   {"role": "assistant", "content": "还醒着呢"},
                   {"role": "user", "content": "是"}]
        before = copy.deepcopy(history)
        for system in ("", " \n "):
            for stance in (None, [.1, .2, .3]):
                with self.subTest(system=system, stance=stance):
                    reply = self.generator.complete(system, history, max_new_tokens=64, stance_vector=stance)
                    self.assertEqual(reply, self.raw.model.output_text)
                    if stance is not None:
                        self.assertEqual(self.raw.tokenizer.text_calls[-1]["text"],
                                         "Dialogue:\nA: 你睡了吗\nB: 还醒着呢\nA: 是\nB:")
                        expected = self.raw.projector(torch.tensor([stance])).detach()
                        self.assertTrue(torch.allclose(self.raw.model.calls[-1]["inputs_embeds"][:, :2], expected))
                    else:
                        self.assertEqual(self.raw.tokenizer.calls[-1]["messages"], before)
                    self.assertEqual(history, before)

    def test_prefix_uses_training_tokens_and_embeddings_without_chat_template(self):
        history = [{"role": "user", "content": "你睡了吗"},
                   {"role": "assistant", "content": "你呀，你怎么了？"},
                   {"role": "user", "content": " 不取消第2项。\n保留原文 "}]
        before = copy.deepcopy(history)
        self.generator.complete("", history, max_new_tokens=64, stance_vector=[0., 0., 0.])
        prompt = "Dialogue:\nA: 你睡了吗\nB: 你呀，你怎么了？\nA:  不取消第2项。\n保留原文 \nB:"
        self.assertEqual(self.raw.tokenizer.text_calls[-1]["text"], prompt)
        self.assertEqual(self.raw.tokenizer.calls, [])
        ids = torch.tensor([[self.raw.tokenizer.bos_token_id] + self.raw.tokenizer.ids(prompt)])
        self.assertEqual((ids == self.raw.tokenizer.bos_token_id).sum().item(), 1)
        call = self.raw.model.calls[-1]
        expected_tokens = self.raw.model.get_input_embeddings()(ids)
        self.assertTrue(torch.equal(call["inputs_embeds"][:, 2:], expected_tokens))
        self.assertEqual(len(self.raw.model.calls), 1)
        self.assertEqual(self.raw.stance_calls, 1)
        self.assertEqual(history, before)

    def test_prefix_budget_discards_whole_old_pair_and_keeps_current_input(self):
        history = [{"role": "user", "content": "旧问题" * 20},
                   {"role": "assistant", "content": "旧回答" * 20},
                   {"role": "user", "content": "最近问题"},
                   {"role": "assistant", "content": "最近回答"},
                   {"role": "user", "content": "不要取消第2项。"}]
        before = copy.deepcopy(history)
        kept = "Dialogue:\nA: 最近问题\nB: 最近回答\nA: 不要取消第2项。\nB:"
        adapter = LocalChatGenerator(self.raw, max_prompt_tokens=len(kept) + 1)
        adapter.complete("", history, max_new_tokens=64, stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.tokenizer.text_calls[-1]["text"], kept)
        self.assertEqual(history, before)
        self.assertEqual(self.raw.tokenizer.calls, [])

    def test_prefix_training_limit_rejects_long_current_input_before_generation(self):
        with self.assertRaisesRegex(ValueError, "384-token prompt budget"):
            self.generator.complete("", [{"role": "user", "content": "不要取消第2项。" * 60}],
                                    max_new_tokens=64, stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.model.calls, [])

    def test_prefix_rejects_system_prompt_instead_of_reintroducing_chat_template(self):
        with self.assertRaisesRegex(ValueError, "role_prompt_enabled=false"):
            self.generator.complete(self.system, self.messages, max_new_tokens=64, stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.model.calls, [])
        self.assertEqual(self.raw.tokenizer.calls, [])

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
        with self.assertRaisesRegex(ValueError, "system prompt and current user text were not truncated"):
            adapter.complete(self.system, self.messages, max_new_tokens=64)
        self.assertEqual(self.raw.model.calls, [])

    def test_actual_model_context_reserves_prefix_and_output_space(self):
        self.raw.model.config.max_position_embeddings = 64
        with self.assertRaisesRegex(ValueError, "leave no room"):
            self.complete(stance_vector=[1, 2, 3])
        self.assertEqual(self.raw.model.calls, [])

    def test_eos_stops_before_following_tokens_and_normal_speaker_text_survives(self):
        self.raw.model.after_eos = "不应包含的尾巴"
        self.assertEqual(self.complete(), self.raw.model.output_text)
        self.assertEqual(self.complete(stance_vector=[1, 2, 3]), self.raw.model.output_text)

    def test_empty_model_completion_is_an_error(self):
        self.raw.model.output_text = ""
        with self.assertRaisesRegex(RuntimeError, "empty completion"):
            self.complete()

    def test_sampling_settings_are_explicit_and_reply_repeat_option_is_retained(self):
        self.complete(do_sample=True, temperature=.8, top_p=.95, no_repeat_ngram_size=4)
        call = self.raw.model.calls[-1]
        self.assertEqual(call["temperature"], .8)
        self.assertEqual(call["top_p"], .95)
        self.assertEqual(call["no_repeat_ngram_size"], 4)
        self.assertEqual(call["eos_token_id"], self.raw.tokenizer.eos_token_id)

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
