from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .config import local_backbone


class LocalMistral:
    def __init__(self, path, *, device="auto", dtype="auto", max_context_tokens=1024):
        # Absolute directory + local_files_only: model and tokenizer never fall
        # back to a Hub identifier or fetch remote custom code.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.path = local_backbone(path)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if dtype == "auto":
            self.dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if self.device.type == "cuda" else torch.float32
        else:
            self.dtype = getattr(torch, dtype)
        if self.device.type == "cpu" and self.dtype == torch.float16:
            raise ValueError("CPU 请使用 float32 或 bfloat16")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(str(self.path), local_files_only=True, trust_remote_code=False,
            torch_dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation="sdpa").to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.max_context_tokens = max_context_tokens

    def prompt_ids(self, messages):
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("生成上下文必须以 user 消息结束")
        messages = [dict(message) for message in messages]
        def template(items):
            value = self.tokenizer.apply_chat_template(items, tokenize=True, add_generation_prompt=True)
            return value.tolist() if hasattr(value, "tolist") else value
        while True:
            ids = template(messages)
            if len(ids) <= self.max_context_tokens or len(messages) <= 1:
                break
            # Remove an entire oldest user/assistant pair, preserving roles.
            messages = messages[2:]
        if len(ids) > self.max_context_tokens:
            # Trim only content, then reapply the full template. Cutting the
            # templated IDs would remove [INST] and invalidate speaker roles.
            content_ids = self.tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
            overhead = len(template([{"role": "user", "content": ""}]))
            budget = self.max_context_tokens - overhead
            if budget < 1:
                raise ValueError("max_context_tokens 太小，无法保留完整聊天模板")
            while True:
                content = self.tokenizer.decode(content_ids[-budget:], skip_special_tokens=True)
                ids = template([{"role": "user", "content": content}])
                excess = len(ids) - self.max_context_tokens
                if excess <= 0:
                    break
                budget -= max(1, excess)
                if budget < 1:
                    raise ValueError("无法在 max_context_tokens 内保留聊天模板与用户内容")
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def response_ids(self, text, max_tokens):
        ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens - 1]
        ids.append(self.tokenizer.eos_token_id)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def features(self, conversations, batch_size=1):
        if not conversations:
            return np.empty((0, self.hidden_size), dtype=np.float32)
        self.model.eval()
        result = []
        for start in range(0, len(conversations), batch_size):
            rows = [self.prompt_ids(messages)[0] for messages in conversations[start:start + batch_size]]
            length = max(len(row) for row in rows)
            ids = torch.full((len(rows), length), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
            attention = torch.zeros_like(ids)
            for index, row in enumerate(rows):
                ids[index, -len(row):] = row
                attention[index, -len(row):] = 1
            # Call the transformer body, avoiding vocabulary logits and the
            # tuple of all layer hidden states during feature caching.
            output = self.model.model(input_ids=ids, attention_mask=attention, use_cache=False, return_dict=True)
            pooled = output.last_hidden_state[:, -1].float()
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            result.append(pooled.cpu().numpy())
        return np.concatenate(result)

    @torch.inference_mode()
    def generate(self, messages, vector, projector, *, max_new_tokens=128, do_sample=False, temperature=0.7, top_p=0.9):
        self.model.eval()
        projector.eval()
        ids = self.prompt_ids(messages)
        embeddings = self.model.get_input_embeddings()(ids)
        prefix = projector(torch.as_tensor(vector, dtype=torch.float32, device=self.device).reshape(1, -1)).to(embeddings.dtype)
        combined = torch.cat([prefix, embeddings], dim=1)
        options = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                   "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id,
                   "use_cache": True}
        if do_sample:
            options.update(temperature=temperature, top_p=top_p)
        # With inputs_embeds-only, HF returns only generated IDs. Supplying
        # input_ids as well would introduce a misleading prefix-length offset.
        generated = self.model.generate(inputs_embeds=combined, attention_mask=torch.ones(combined.shape[:2], dtype=torch.long, device=self.device), **options)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError("模型生成了空回复，请检查训练质量与解码配置")
        return text
