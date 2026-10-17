"""Training-format replies and instruction-based ASR sharing archived weights.

The raw archive is immutable. This adapter supplies new inference prompts to
its already loaded model/tokenizer/projector, without creating another model
or changing trained weights. Callers serialize complete turns around shared
model access. A missing stance means plain instruction generation, useful for
conservative speech-transcript correction; a stance uses the archived training
dialogue formatter and retains the trained prefix.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import torch
from .errors import PromptTooLong


class LocalChatGenerator:
    def __init__(self, raw_generator: Any, max_prompt_tokens: int = 4096):
        if isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int) or max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        self.raw_generator = raw_generator
        self.model = raw_generator.model
        self.tokenizer = raw_generator.tokenizer
        self.projector = raw_generator.projector
        self.device = raw_generator.device
        self.max_prompt_tokens = max_prompt_tokens
        # Each archive is imported in its own private package. Reuse that
        # version's training formatter rather than maintaining another copy.
        runtime = importlib.import_module(type(raw_generator).__module__)
        self._dialogue_formatter = getattr(runtime, "generation_prompt", None)
        self._training_prompt_tokens = getattr(raw_generator, "config", {}).get(
            "max_prompt_tokens", max_prompt_tokens)
        if (isinstance(self._training_prompt_tokens, bool)
                or not isinstance(self._training_prompt_tokens, int)
                or self._training_prompt_tokens < 1):
            raise ValueError("Archived max_prompt_tokens must be a positive integer")

    @staticmethod
    def _messages(system_prompt: str, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be text; use an empty string to disable it")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty user/assistant history")
        copied = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("Each message must be an object with role and content")
            expected = "user" if index % 2 == 0 else "assistant"
            if message.get("role") != expected:
                raise ValueError("Messages must alternate user/assistant, starting with user; system is supplied separately")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Every message must contain nonempty text")
            copied.append({"role": expected, "content": content})
        if copied[-1]["role"] != "user":
            raise ValueError("Generation history must end with the current user message")
        return copied

    def _prompt(self, system_prompt: str, messages: list[dict[str, str]], budget: int, *,
                emotion_conditioned: bool = False) -> dict[str, torch.Tensor]:
        remaining = self._messages(system_prompt, messages)
        if emotion_conditioned:
            if system_prompt.strip():
                raise ValueError("Emotion-prefix replies use the trained Dialogue format without a system prompt; "
                                 "set role_prompt_enabled=false")
            if not callable(self._dialogue_formatter):
                raise RuntimeError("Archived generator does not expose its training generation_prompt formatter")
            budget = min(budget, self._training_prompt_tokens)
        while True:
            if emotion_conditioned:
                history = [{"role": "A" if item["role"] == "user" else "B", "text": item["content"]}
                           for item in remaining]
                prompt = self._dialogue_formatter(history, "B")
                # Training adds BOS through the tokenizer. Measure the full
                # input before pruning; never silently truncate current text.
                encoded = self.tokenizer(prompt, add_special_tokens=True,
                                         truncation=False, return_tensors="pt")
            else:
                # Optional ASR repair uses plain instruction generation and
                # the official template, without an emotion prefix.
                template_messages = remaining
                if system_prompt.strip():
                    template_messages = [{"role": "system", "content": system_prompt}, *remaining]
                encoded = self.tokenizer.apply_chat_template(
                    template_messages,
                    tokenize=True, return_dict=True, return_tensors="pt",
                    add_generation_prompt=True,
                )
            if not hasattr(encoded, "get"):
                raise RuntimeError("Chat tokenizer did not return input_ids")
            ids = encoded.get("input_ids")
            if not torch.is_tensor(ids) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise RuntimeError("Chat tokenizer must return one nonempty token sequence")
            if ids.shape[1] <= budget:
                mask = encoded.get("attention_mask")
                if mask is None:
                    mask = torch.ones_like(ids)
                if not torch.is_tensor(mask) or mask.shape != ids.shape:
                    raise RuntimeError("Chat attention mask does not match input_ids")
                return {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device)}
            if len(remaining) == 1:
                subject = "Training dialogue prompt" if emotion_conditioned else "System prompt plus current user message"
                preserved = ("The current user text was not truncated." if emotion_conditioned
                             else "The system prompt and current user text were not truncated.")
                raise PromptTooLong(
                    f"{subject} requires {ids.shape[1]} tokens, "
                    f"exceeding the {budget}-token prompt budget; shorten the current request. "
                    f"{preserved}"
                )
            # Only whole, oldest user/assistant pairs may be discarded. Keep
            # the latest user text intact, including any negation or number.
            remaining = remaining[2:]

    @torch.inference_mode()
    def complete(self, system_prompt: str, messages: list[dict[str, str]], *,
                 max_new_tokens: int, stance_vector: Any = None,
                 do_sample: bool = False, temperature: float = .7,
                 top_p: float = .9, no_repeat_ngram_size: int = 0) -> str:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        if isinstance(no_repeat_ngram_size, bool) or not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
        if do_sample:
            if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Sampling temperature must be finite and positive")
            if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
                raise ValueError("Sampling top_p must be in (0, 1]")
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("The local chat tokenizer must define EOS")
        prefix_length = int(self.projector.prefix_length) if stance_vector is not None else 0
        budget = self.max_prompt_tokens
        context_length = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(context_length, int) and not isinstance(context_length, bool) and context_length > 0:
            budget = min(budget, context_length - max_new_tokens - prefix_length)
        if budget < 1:
            raise ValueError("Requested output and emotion prefix leave no room in the model context window")
        encoded = self._prompt(system_prompt, messages, budget,
                               emotion_conditioned=stance_vector is not None)
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        self.model.eval()
        options = {"max_new_tokens": max_new_tokens, "max_length": None,
                   "do_sample": do_sample, "no_repeat_ngram_size": no_repeat_ngram_size,
                   "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos,
                   "eos_token_id": eos, "use_cache": True,
                   "return_dict_in_generate": False}
        if do_sample:
            options.update(temperature=temperature, top_p=top_p)
        if stance_vector is None:
            # Plain instruction generation for correction: no emotional prefix.
            output = self.model.generate(input_ids=ids, attention_mask=mask, **options)
        else:
            self.projector.eval()
            stance = self.raw_generator._stance(stance_vector)
            token_embeds = self.model.get_input_embeddings()(ids)
            prefix = self.projector(stance).to(dtype=token_embeds.dtype, device=token_embeds.device)
            if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[1] != prefix_length or prefix.shape[2] != token_embeds.shape[2]:
                raise RuntimeError("The trained emotion prefix has an incompatible shape")
            inputs = torch.cat([prefix, token_embeds], dim=1)
            full_mask = torch.cat([torch.ones((1, prefix_length), dtype=mask.dtype, device=mask.device), mask], dim=1)
            output = self.model.generate(inputs_embeds=inputs, attention_mask=full_mask, **options)
        sequences = getattr(output, "sequences", output)
        if not torch.is_tensor(sequences) or sequences.ndim != 2 or sequences.shape[0] != 1:
            raise RuntimeError("The local model did not return one generated token sequence")
        # input_ids generation includes its prompt; inputs_embeds-only generation
        # in the archived Transformers stack returns continuation tokens only.
        continuation = sequences[0, ids.shape[1]:] if stance_vector is None else sequences[0]
        continuation_ids = continuation.detach().cpu().tolist()
        eos_ids = {int(value) for value in eos} if isinstance(eos, (list, tuple)) else {int(eos)}
        for index, token in enumerate(continuation_ids):
            if token in eos_ids:
                continuation_ids = continuation_ids[:index]
                break
        text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("The local model generated an empty completion")
        # Normal text may legitimately contain A:/B: or quoted dialogue. EOS,
        # rather than speaker-label string splitting, controls the completion.
        return text.strip()
