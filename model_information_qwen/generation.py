"""Use the trained Qwen generator's identical chat template and prefix path."""

from __future__ import annotations

import math

from .errors import PromptTooLong


class LocalChatGenerator:
    """Delegate each completion once to the generator used during training."""

    def __init__(self, raw_generator, max_prompt_tokens=4096):
        if isinstance(max_prompt_tokens, bool) or not isinstance(max_prompt_tokens, int) or max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be a positive integer")
        if not callable(getattr(raw_generator, "complete", None)):
            raise TypeError("The trained Qwen generator must implement complete()")
        self.raw_generator = raw_generator
        self.max_prompt_tokens = max_prompt_tokens
        self.model = raw_generator.model
        self.tokenizer = raw_generator.tokenizer
        self.projector = raw_generator.projector
        self.device = raw_generator.device

    def complete(self, system_prompt, messages, *, max_new_tokens,
                 stance_vector=None, do_sample=False, temperature=.7,
                 top_p=.9, no_repeat_ngram_size=0):
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be a boolean")
        if isinstance(no_repeat_ngram_size, bool) or not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Sampling temperature must be finite and positive")
        if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("Sampling top_p must be in (0, 1]")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty text")
        if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
            raise ValueError("Generation requires alternating user/assistant history ending in user")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("Messages must alternate user/assistant, beginning and ending with user")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Every message must contain nonempty text")
        try:
            return self.raw_generator.complete(
                system_prompt, messages, max_new_tokens=max_new_tokens,
                stance_vector=stance_vector, do_sample=do_sample,
                temperature=temperature, top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_prompt_tokens=self.max_prompt_tokens,
            )
        except ValueError as exc:
            if type(exc).__name__ == "PromptTooLong":
                raise PromptTooLong(str(exc)) from exc
            raise
