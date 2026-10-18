"""Frozen Qwen3 with an EmoStance continuous prefix and native chat prompts.

Only the two-layer stance projector is trained. Training and inference use
the same Qwen chat template with enable_thinking=False, in both A/B directions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import local_backbone


class PromptTooLong(ValueError):
    """The intact current input cannot fit the configured prompt budget."""


def _role(value: str) -> str:
    result = {"A": "A", "B": "B", "user": "A", "assistant": "B"}.get(value)
    if result is None:
        raise ValueError(f"Expected A/B or user/assistant role, got {value!r}")
    return result


def build_model_text(history: list[dict[str, Any]], situation: str = "") -> str:
    """Equivalent to official latent_stance_control.data.build_model_text."""
    if not history:
        raise ValueError("Generation requires a nonempty dialogue history")
    parts = []
    for message in history:
        text = message.get("text", message.get("content"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Every historical turn needs its original nonempty text")
        parts.append(f"{_role(message.get('role'))}: {text}")
    context = "\n".join(parts)
    return (f"Situation: {situation}\nDialogue:\n{context}" if situation else f"Dialogue:\n{context}")


def generation_prompt(history: list[dict[str, Any]], next_role: str, situation: str = "") -> str:
    return build_model_text(history, situation) + f"\n{_role(next_role)}:"


def chat_messages(history, next_role):
    """Make the predicted speaker assistant, including training B -> A rows."""
    target = _role(next_role)
    if not history:
        raise ValueError("Generation needs a nonempty history")
    messages = []
    for item in history:
        content = item.get("text", item.get("content"))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("History contains empty text")
        messages.append({"role": "assistant" if _role(item["role"]) == target else "user", "content": content})
    if messages[-1]["role"] != "user":
        raise ValueError("The observed speaker must differ from the next speaker")
    return messages


class StancePrefixProjector(nn.Module):
    """Paper architecture: stance_dim -> LM hidden size -> P * LM hidden."""

    def __init__(self, vector_dim: int, hidden_size: int, prefix_length: int = 8,
                 intermediate_size: int = 4096):
        super().__init__()
        self.vector_dim = vector_dim
        self.hidden_size = hidden_size
        self.prefix_length = prefix_length
        self.intermediate_size = intermediate_size
        self.net = nn.Sequential(nn.Linear(vector_dim, intermediate_size), nn.Tanh(),
                                 nn.Linear(intermediate_size, prefix_length * hidden_size))

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim != 2 or vector.shape[1] != self.vector_dim:
            raise ValueError(f"Stance input must have shape [batch, {self.vector_dim}]")
        return self.net(vector.float()).view(vector.shape[0], self.prefix_length, self.hidden_size)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolved_device_dtype(config: dict[str, Any]) -> tuple[torch.device, torch.dtype]:
    value = config.get("device", "cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this Python environment")
    dtype_name = config.get("dtype", "bfloat16" if device.type == "cuda" else "float32")
    if dtype_name == "auto":
        dtype_name = "bfloat16" if device.type == "cuda" else "float32"
    if dtype_name not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"Unsupported generator dtype: {dtype_name}")
    if device.type == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This CUDA device does not support the configured bfloat16 dtype")
    return device, getattr(torch, dtype_name)


class PaperGenerator:
    def __init__(self, config: dict[str, Any], checkpoint: str | Path | None = None):
        self.config = dict(config)
        if self.config.get("enable_thinking", False) is not False:
            raise ValueError("This training run requires enable_thinking=False")
        self.device, self.dtype = _resolved_device_dtype(self.config)
        self.base_model_path = local_backbone(self.config.get("base_model_path", self.config.get("base_model")))
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base_model_path), local_files_only=True,
                                                      trust_remote_code=False, use_fast=True)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Local tokenizer must define EOS")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.base_model_path), local_files_only=True, trust_remote_code=False,
            dtype=self.dtype, low_cpu_mem_usage=True, attn_implementation="sdpa").to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.projector = StancePrefixProjector(
            int(self.config.get("vector_dim", 256)), int(self.model.config.hidden_size),
            int(self.config.get("prefix_length", 8)), int(self.config.get("prefix_hidden_dim", 4096))).to(self.device)
        self.projector.eval()
        self.checkpoint_state = None
        if checkpoint is None and self.config.get("run_dir"):
            directory = Path(self.config["run_dir"]) / "generator"
            checkpoint = next((path for path in (directory / "best.pt", directory / "last.pt") if path.exists()), None)
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint)
            if checkpoint_path.is_dir():
                checkpoint_path = next((path for path in (checkpoint_path / "best.pt", checkpoint_path / "last.pt") if path.exists()), checkpoint_path / "best.pt")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            architecture = self.architecture()
            if state.get("architecture") != architecture:
                raise ValueError(f"Generator checkpoint architecture mismatch: expected {architecture}")
            if Path(state["base_model_path"]).resolve() != self.base_model_path:
                raise ValueError("Generator checkpoint references a different local backbone")
            if state.get("prompt_format") != "qwen3_chat_no_thinking_v1":
                raise ValueError("Generator checkpoint has a different chat template")
            self.projector.load_state_dict(state["projector"])
            self.checkpoint_state = state

    def architecture(self) -> dict[str, int]:
        return {"vector_dim": self.projector.vector_dim, "lm_hidden_size": self.projector.hidden_size,
                "prefix_length": self.projector.prefix_length, "prefix_hidden_dim": self.projector.intermediate_size}

    def _prompt_ids(self, messages, system_prompt, budget, *, strict=True):
        remaining = list(messages)
        if not remaining or remaining[-1]["role"] != "user":
            raise ValueError("Generation history must end with a user turn")
        while True:
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, *remaining],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt")
            ids = encoded["input_ids"]
            if ids.shape[1] <= budget:
                return ids.to(self.device)
            if len(remaining) > 1:
                # Preserve whole turns; never cut chat control tokens.
                remaining = remaining[2:] if len(remaining) > 2 else remaining[1:]
                continue
            if strict:
                raise PromptTooLong(f"Current request and system prompt exceed {budget} tokens")
            # Long training utterances may be truncated, keeping their most
            # recent text and reapplying the complete template and system.
            content_ids = self.tokenizer(remaining[0]["content"], add_special_tokens=False)["input_ids"]
            excess = ids.shape[1] - budget
            if len(content_ids) <= excess:
                raise ValueError("System prompt exceeds the training token budget")
            remaining = [{"role": "user", "content": self.tokenizer.decode(content_ids[excess:])}]

    def prompt_ids(self, history: list[dict[str, Any]], next_role: str, situation: str = "") -> torch.Tensor:
        system = self.config["system_prompt"]
        if situation:
            system += "\n对话背景：" + situation
        return self._prompt_ids(chat_messages(history, next_role), system,
                                int(self.config.get("max_prompt_tokens", 512)), strict=False)

    def response_ids(self, response: str) -> torch.Tensor:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Training response must be a nonempty string")
        # The official generator caps response content at 128, then adds EOS.
        ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        ids = ids[:int(self.config.get("max_training_response_tokens", 128))]
        return torch.tensor([ids + [self.tokenizer.eos_token_id]], dtype=torch.long, device=self.device)

    def _stance(self, vector: Any) -> torch.Tensor:
        stance = torch.as_tensor(vector, dtype=torch.float32, device=self.device).reshape(1, -1)
        if stance.shape[1] != self.projector.vector_dim or not torch.isfinite(stance).all():
            raise ValueError(f"Expected finite {self.projector.vector_dim}-dimensional stance vector")
        return stance

    def loss(self, row: dict[str, Any], vector: Any) -> torch.Tensor:
        """Reply+EOS CE only; deliberately retain gradients through frozen LM."""
        prompt_ids = self.prompt_ids(row["history"], row["next_role"], row.get("situation", ""))
        response_ids = self.response_ids(row["response"])
        ids = torch.cat([prompt_ids, response_ids], dim=1)
        embeds = self.model.get_input_embeddings()(ids).detach()
        prefix = self.projector(self._stance(vector)).to(embeds.dtype)
        inputs = torch.cat([prefix, embeds], dim=1)
        labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=self.device)
        labels[:, self.projector.prefix_length + prompt_ids.shape[1]:] = response_ids
        output = self.model(inputs_embeds=inputs,
                            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device),
                            labels=labels, use_cache=False)
        return output.loss

    def distillation_losses(self, row, vector, temperature=1.0):
        """Aligned reply CE and KL(no-prefix teacher || conditioned student).

        The frozen backbone is shared: there is no second 8B model. Computing
        only the R reply logits avoids allocating vocabulary scores for every
        system/history/prefix token. The teacher has no prefix at all.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Distillation temperature must be positive")
        prompt = self.prompt_ids(row["history"], row["next_role"], row.get("situation", ""))
        response = self.response_ids(row["response"])
        ids = torch.cat([prompt, response[:, :-1]], dim=1)
        count = response.shape[1]
        # The frozen teacher is deterministic; keep the student path's original
        # mode so training checkpoint recomputation remains enabled.
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                teacher = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    logits_to_keep=count, use_cache=False).logits
                teacher_logp = (teacher.float() / temperature).log_softmax(-1).detach()
                del teacher
        finally:
            self.model.train(was_training)
        embeds = self.model.get_input_embeddings()(ids).detach()
        prefix = self.projector(self._stance(vector)).to(embeds.dtype)
        inputs = torch.cat([prefix, embeds], dim=1)
        student = self.model(inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device),
            logits_to_keep=count, use_cache=False).logits.float()
        if student.shape != teacher_logp.shape or student.shape[1] != count:
            raise RuntimeError("Teacher/student reply logits are not aligned")
        ce = F.cross_entropy(student.reshape(-1, student.shape[-1]), response.reshape(-1))
        kl = F.kl_div((student / temperature).log_softmax(-1), teacher_logp,
                      log_target=True, reduction="sum") * (temperature ** 2) / response.numel()
        return {"ce": ce, "teacher_kl": kl}

    @torch.inference_mode()
    def generate(self, history: list[dict[str, Any]], next_role: str, vector: Any, max_new_tokens: int = 128) -> str:
        return self.complete(self.config["system_prompt"], chat_messages(history, next_role),
            max_new_tokens=max_new_tokens, stance_vector=vector,
            do_sample=bool(self.config.get("do_sample", True)),
            temperature=float(self.config.get("temperature", .7)),
            top_p=float(self.config.get("top_p", .8)),
            no_repeat_ngram_size=int(self.config.get("no_repeat_ngram_size", 4)),
            max_prompt_tokens=int(self.config.get("chat_quality_max_prompt_tokens", self.config.get("max_prompt_tokens", 512))),
            top_k=int(self.config.get("top_k", 20)))

    @torch.inference_mode()
    def complete(self, system_prompt, messages, *, max_new_tokens, stance_vector=None,
                 do_sample=False, temperature=.7, top_p=.9, no_repeat_ngram_size=0,
                 max_prompt_tokens=None, top_k=20):
        """Generate exactly one reply; never build or rerank a candidate pool."""
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be nonempty")
        if not isinstance(do_sample, bool):
            raise ValueError("do_sample must be boolean")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("Invalid generation sampling settings")
        self.model.eval()
        self.projector.eval()
        self.model.config.use_cache = True
        prefix_length = self.projector.prefix_length if stance_vector is not None else 0
        budget = min(int(max_prompt_tokens or self.config.get("max_prompt_tokens", 512)),
            self.model.config.max_position_embeddings - max_new_tokens - prefix_length)
        if budget < 1:
            raise ValueError("No space for the prompt in the model context")
        ids = self._prompt_ids(messages, system_prompt, budget)
        options = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,
                   "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id,
                   "use_cache": True, "max_length": None, "return_dict_in_generate": False,
                   "num_beams": 1, "num_return_sequences": 1,
                   "no_repeat_ngram_size": no_repeat_ngram_size}
        if options["do_sample"]:
            options.update(temperature=temperature, top_p=top_p, top_k=top_k)
        if stance_vector is None:
            output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), **options)
            continuation = output[0, ids.shape[1]:]
        else:
            embeds = self.model.get_input_embeddings()(ids)
            prefix = self.projector(self._stance(stance_vector)).to(embeds.dtype)
            inputs = torch.cat([prefix, embeds], dim=1)
            output = self.model.generate(inputs_embeds=inputs,
                attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device), **options)
            continuation = output[0]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True).strip()
        # The template already emits the empty thinking block. Reject an
        # unexpected hidden reasoning continuation instead of exposing it.
        if "<think>" in text or "</think>" in text:
            raise RuntimeError("Qwen returned reasoning in non-thinking mode")
        if not text.strip():
            raise RuntimeError("Generator returned an empty reply")
        return text.strip()


def _training_signature(rows: list[dict[str, Any]], space: Any, config: dict[str, Any], dev_rows=None) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = {key: row[key] for key in ("sample_id", "history", "next_role", "response", "target_meme_distribution") if key in row}
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    for name in ("meme_ids", "centroids", "membership", "prototypes"):
        value = getattr(space, name)
        digest.update(json.dumps(value.tolist() if hasattr(value, "tolist") else value, sort_keys=True).encode())
    keys = ("seed", "prefix_epochs", "prefix_length", "prefix_hidden_dim", "prefix_learning_rate",
            "max_prompt_tokens", "max_training_response_tokens", "gradient_checkpointing", "vector_dim",
            "base_model_path", "system_prompt", "enable_thinking", "prefix_teacher_kl_weight",
            "prefix_teacher_temperature", "language_preservation_interval", "language_preservation_weight",
            "language_preservation_ce_weight",
            "chat_quality_enabled", "chat_quality_interval", "chat_quality_seeds", "chat_quality_max_new_tokens",
            "chat_quality_temperature", "chat_quality_top_p", "chat_quality_top_k", "chat_quality_semantic_judge",
            "chat_quality_no_repeat_ngram_size", "chat_quality_max_prompt_tokens", "chat_quality_judge_max_new_tokens",
            "chat_quality_judge_max_prompt_tokens", "prefix_dev_subset_size",
            "prefix_ce_max_response_chars", "prefix_ce_exclude_persona_claims")
    digest.update(json.dumps({key: config.get(key) for key in keys}, sort_keys=True).encode())
    digest.update(json.dumps({key: value for key, value in config.items()
                             if key.startswith("chat_quality_") and key != "chat_quality_holdout_file"},
                            sort_keys=True).encode())
    for row in dev_rows or []:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())
    planner_path = Path(config["run_dir"]) / "planner/best.pt"
    if planner_path.is_file():
        with planner_path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
    digest.update(b"qwen3_chat_no_thinking_v1")
    download = Path(config["base_model_path"]) / "download_manifest.json"
    if download.is_file():
        digest.update(download.read_bytes())
    for key in ("chat_quality_dev_file", "language_preservation_train_file"):
        if config.get(key):
            digest.update(Path(config[key]).read_bytes())
    digest.update(b"quality_preserving_distillation_v2")
    return digest.hexdigest()


def response_ce_weight(row, config):
    """Do not teach targets that explicitly contradict the deployed persona."""
    import re
    text = row["response"]
    maximum = int(config.get("prefix_ce_max_response_chars", 0))
    if maximum and len(text) > maximum:
        return 0.0, "exceeds_role_length"
    if config.get("prefix_ce_exclude_persona_claims", False):
        named = re.search(r"我(?:的名字(?:是|叫)|名叫|叫)(?!你|他|她|它|大家|我们)([^，。！？、\s]{1,20})", text)
        if named and "meme" not in named.group(1).lower():
            return 0.0, "conflicting_name_claim"
        if re.search(r"我是(?:ChatGPT|人工智能|语言模型|AI助手|人类|机器人)", text, re.I):
            return 0.0, "conflicting_identity_claim"
    return 1.0, "compatible_reference"


def _loss_only_signature(train_rows, dev_rows, space, config):
    """Separate identity from the older quality-preserving training format."""
    keys = ("seed", "version", "model_version", "data_version", "prefix_epochs", "prefix_learning_rate", "prefix_length",
            "prefix_hidden_dim", "vector_dim", "max_prompt_tokens", "max_training_response_tokens",
            "base_model_path", "system_prompt", "enable_thinking", "generator_selection_mode",
            "prefix_dev_subset_size", "generator_dev_eval_interval", "gradient_checkpointing")
    digest = hashlib.sha256(b"qwen3_reply_ce_dev_selection_v1")
    for value in (train_rows, dev_rows, {key: config.get(key) for key in keys}):
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode("utf-8"))
    for name in ("meme_ids", "centroids", "membership", "prototypes"):
        value = getattr(space, name)
        digest.update(json.dumps(value.tolist() if hasattr(value, "tolist") else value,
                                 sort_keys=True, allow_nan=False).encode("utf-8"))
    for path in (Path(config["run_dir"]) / "planner/best.pt",
                 Path(config["base_model_path"]) / "download_manifest.json"):
        if path.is_file():
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _training_rng_state():
    return dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_training_rng(state):
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng"):
        torch.cuda.set_rng_state_all(state["cuda_rng"])


@contextmanager
def _loss_evaluation_state(generator):
    """A dev forward pass must not change stochastic training or mixed modes."""
    rng = _training_rng_state()
    modules = list(dict.fromkeys([*generator.model.modules(), *generator.projector.modules()]))
    modes = [(module, module.training) for module in modules]
    cache = generator.model.config.use_cache
    generator.model.eval()
    generator.projector.eval()
    try:
        with torch.inference_mode():
            yield
    finally:
        for module, training in modes:
            module.training = training
        generator.model.config.use_cache = cache
        _restore_training_rng(rng)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _train_loss_only_generator(train_rows, space, run_dir, config, dev_rows):
    """Explicit full training selected only by fixed development reply CE.

    No generation, response-quality judge, no-prefix teacher, reference
    filtering, or language-preservation anchors are used by this path.
    """
    if config.get("chat_quality_enabled", False):
        raise ValueError("dev_response_ce selection requires chat_quality_enabled=false")
    if not dev_rows or any(row.get("split", row.get("source_split", "dev")) not in
                           {"dev", "valid", "validation"} for row in dev_rows):
        raise ValueError("Loss-only checkpoint selection requires development rows, never train/test rows")
    for key in ("prefix_teacher_kl_weight", "language_preservation_weight", "language_preservation_ce_weight"):
        value = float(config.get(key, 0.0))
        if not math.isfinite(value) or value != 0:
            raise ValueError(f"Pure reply CE requires {key}=0")
    if config.get("prefix_ce_max_response_chars", 0) or config.get("prefix_ce_exclude_persona_claims", False):
        raise ValueError("Pure reply CE must not filter reference responses by persona or length")
    epochs = int(config.get("prefix_epochs", 1))
    subset_size = int(config.get("prefix_dev_subset_size", 128))
    checkpoint_interval = int(config.get("generator_runtime_checkpoint_interval", config.get("checkpoint_interval", 100)))
    eval_interval = int(config.get("generator_dev_eval_interval", checkpoint_interval))
    step_limit = config.get("generator_max_steps")
    learning_rate = float(config.get("prefix_learning_rate", 1e-4))
    if min(epochs, subset_size, checkpoint_interval, eval_interval) < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Loss-only epochs, evaluation/checkpoint intervals, subset size, and learning rate must be positive")
    if step_limit is not None and (type(step_limit) is not int or step_limit < 1):
        raise ValueError("generator_max_steps must be a positive integer")
    config = dict(config)
    config["generator_dev_eval_interval"] = eval_interval
    config["prefix_dev_subset_size"] = subset_size
    directory = Path(run_dir).resolve() / "generator"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path, last_path, best_path = directory / "manifest.json", directory / "last.pt", directory / "best.pt"
    signature = _loss_only_signature(train_rows, dev_rows, space, config)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("training_signature") != signature:
            raise ValueError("Existing loss-only run has different data/config; choose a new versioned run_dir")
        if previous.get("status") == "complete":
            if not best_path.is_file() or not last_path.is_file():
                raise FileNotFoundError("Completed loss-only training requires independent best.pt and last.pt")
            if previous.get("checkpoint_sha256") != _file_sha256(best_path):
                raise ValueError("Completed loss-only best checkpoint SHA-256 mismatch")
            return previous
    if best_path.exists() and not last_path.exists():
        raise FileNotFoundError("Cannot resume loss-only training from inference-only best.pt without resumable last.pt")
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = PaperGenerator(config, checkpoint=last_path if last_path.exists() else None)
    optimizer = torch.optim.AdamW(generator.projector.parameters(), lr=learning_rate)
    epoch, offset, global_step, epoch_loss, epoch_steps = 0, 0, 0, 0.0, 0
    best_loss, best_dev_ce, best_step, best_epoch = float("inf"), float("inf"), None, None
    last_evaluation_step, history = -1, []
    if generator.checkpoint_state is not None:
        state = generator.checkpoint_state
        if state.get("training_signature") != signature or state.get("generator_selection_mode") != "dev_response_ce":
            raise ValueError("Loss-only checkpoint belongs to a different training run or selection mode")
        optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(generator.device)
        epoch, offset, global_step = state["next_epoch"], state["next_offset"], state["global_step"]
        epoch_loss, epoch_steps = state["epoch_loss_sum"], state["epoch_steps"]
        best_loss, history = state["best_loss"], state["history"]
        best_dev_ce, best_step, best_epoch = state["best_dev_response_ce"], state["best_step"], state["best_epoch"]
        last_evaluation_step = state["last_evaluation_step"]
        setup_rng = state
        if best_step is not None and not best_path.exists():
            raise FileNotFoundError("The resumable loss-only run is missing its selected best.pt")
    else:
        setup_rng = _training_rng_state()
    from .paper_planner import load_planner
    planner = load_planner(run_dir, config)
    indices = sorted(np.random.default_rng(seed).choice(len(dev_rows), size=min(subset_size, len(dev_rows)), replace=False).tolist())
    dev_subset = [dev_rows[index] for index in indices]
    dev_vectors = np.asarray(planner.predict_rows(dev_subset)["target_vector"], dtype=np.float32)
    if dev_vectors.shape != (len(dev_subset), int(space.vector_dim)) or not np.isfinite(dev_vectors).all():
        raise ValueError("Development planner produced invalid fixed control vectors")
    subset_path = directory / "dev_subset.json"
    subset = dict(format_version=1, split="dev", seed=seed, indices=indices,
                  sample_ids=[row.get("sample_id", f"dev:{index}") for index, row in zip(indices, dev_subset)],
                  examples=len(dev_subset), source_examples=len(dev_rows), training_signature=signature,
                  control="planner_predicted", weighting="reply_eos_token_weighted")
    if subset_path.exists() and json.loads(subset_path.read_text(encoding="utf-8")) != subset:
        raise ValueError("The fixed development subset has changed during loss-only training")
    _atomic_json(subset_path, subset)
    _atomic_json(directory / "training_target_policy.json", dict(counts={"all_reference_responses": len(train_rows)},
        all_rows_use_reply_ce=True, teacher_kl_enabled=False, language_preservation_anchors_enabled=False,
        chat_quality_evaluation_enabled=False, reference_filtering_enabled=False))
    # Loading/predicting with the planner must not consume training RNG draws.
    _restore_training_rng(setup_rng)
    gradient_checkpointing = bool(config.get("generator_runtime_gradient_checkpointing", config.get("gradient_checkpointing", True)))
    if gradient_checkpointing:
        generator.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    generator.model.config.use_cache = False
    generator.model.train()
    generator.projector.train()
    started, steps_this_call = time.time(), 0

    def checkpoint(best=False):
        state = dict(format_version=2, base_model_path=str(generator.base_model_path),
            prompt_format="qwen3_chat_no_thinking_v1", architecture=generator.architecture(),
            projector=generator.projector.state_dict(), optimizer=optimizer.state_dict(), training_signature=signature,
            generator_selection_mode="dev_response_ce", selection_criterion="dev_response_ce", objective="reply_ce",
            next_epoch=epoch, next_offset=offset, global_step=global_step, epoch_loss_sum=epoch_loss,
            epoch_steps=epoch_steps, best_loss=best_loss, best_dev_response_ce=best_dev_ce,
            best_step=best_step, best_epoch=best_epoch, last_evaluation_step=last_evaluation_step, history=history,
            selection_metric=dict(name="dev_response_ce", value=best_dev_ce, step=best_step, epoch=best_epoch,
                                  control="planner_predicted", weighting="reply_eos_token_weighted"),
            model_version=config.get("model_version"), version=config.get("version"),
            data_version=config.get("data_version"), **_training_rng_state())
        if best:
            excluded = {"optimizer", "torch_rng", "python_rng", "numpy_rng", "cuda_rng"}
            state = {key: value for key, value in state.items() if key not in excluded}
            state.update(inference_only=True, resumable=False)
        else:
            state.update(inference_only=False, resumable=True)
        _atomic_checkpoint(best_path if best else last_path, state)

    def assess_loss():
        nonlocal best_dev_ce, best_step, best_epoch, last_evaluation_step
        if global_step == last_evaluation_step:
            return
        total_ce, tokens = 0.0, 0
        with _loss_evaluation_state(generator):
            for row, vector in zip(dev_subset, dev_vectors):
                count = int(generator.response_ids(row["response"]).numel())
                value = float(generator.loss(row, vector))
                if count < 1 or not math.isfinite(value):
                    raise RuntimeError("Development reply CE is nonfinite or has no response/EOS tokens")
                total_ce += value * count
                tokens += count
        dev_ce = total_ce / tokens
        selected = dev_ce < best_dev_ce
        last_evaluation_step = global_step
        if selected:
            best_dev_ce, best_step, best_epoch = dev_ce, global_step, epoch + 1
            checkpoint(best=True)
        record = dict(event="dev_response_ce", step=global_step, epoch=epoch + 1,
                      dev_response_ce=dev_ce, best_dev_response_ce=best_dev_ce, best_step=best_step,
                      best_epoch=best_epoch, selected=selected, examples=len(dev_subset), reply_eos_tokens=tokens,
                      control="planner_predicted", weighting="reply_eos_token_weighted")
        with (directory / "loss_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint()

    try:
        while epoch < epochs:
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
            for position in range(offset, len(order)):
                row = train_rows[int(order[position])]
                distribution = row.get("target_meme_distribution", row.get("meme_distribution"))
                if not distribution:
                    raise ValueError(f"Missing target annotations: {row.get('sample_id', position)}")
                _, vector = space.targets(distribution)
                optimizer.zero_grad(set_to_none=True)
                loss = generator.loss(row, vector)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite reply CE at step {global_step + 1}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.projector.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(grad_norm)) or float(grad_norm) == 0:
                    raise RuntimeError("No finite gradient reached the prefix projector")
                optimizer.step()
                value = float(loss.detach().cpu())
                global_step += 1
                steps_this_call += 1
                epoch_loss += value
                epoch_steps += 1
                offset = position + 1
                progress = dict(stage="generator", epoch=epoch + 1, epoch_step=offset,
                    epoch_total=len(train_rows), step=global_step, loss=value, reply_ce=value,
                    teacher_kl=0.0, reply_ce_weight=1.0, anchor_teacher_kl=0.0, anchor_ce=0.0,
                    selection_mode="dev_response_ce", prefix_grad_norm=float(grad_norm),
                    elapsed_seconds=round(time.time() - started, 3))
                del loss
                with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                if global_step % 20 == 0 or steps_this_call == 1:
                    print(json.dumps(progress), flush=True)
                if global_step % eval_interval == 0:
                    assess_loss()
                elif global_step % checkpoint_interval == 0:
                    checkpoint()
                if (directory / "STOP_REQUESTED").exists() or (step_limit is not None and steps_this_call >= step_limit):
                    checkpoint()
                    break
            if offset < len(train_rows):
                break
            assess_loss()
            metric = epoch_loss / epoch_steps
            history.append(dict(epoch=epoch + 1, train_reply_eos_ce=metric, steps=epoch_steps))
            best_loss = min(best_loss, metric)
            epoch += 1
            offset, epoch_loss, epoch_steps = 0, 0.0, 0
            checkpoint()
            if (directory / "STOP_REQUESTED").exists() or (step_limit is not None and steps_this_call >= step_limit):
                break
        # An intentional partial call is also resumable and has an honest best
        # evaluated-so-far checkpoint; it is never marked full training complete.
        if global_step > last_evaluation_step:
            assess_loss()
        checkpoint()
    except BaseException:
        checkpoint()
        raise
    finally:
        if gradient_checkpointing:
            generator.model.gradient_checkpointing_disable()
        generator.model.eval()
        generator.model.config.use_cache = True
        generator.projector.eval()
    if best_step is None or not best_path.is_file():
        raise RuntimeError("Loss-only training did not produce a finite development-selected checkpoint")
    status = "complete" if epoch >= epochs else "incomplete"
    best_sha = _file_sha256(best_path)
    selection = dict(format_version=1, generator_selection_mode="dev_response_ce", selection_criterion="dev_response_ce",
        best_checkpoint_sha256=best_sha, dev_subset_sha256=_file_sha256(subset_path), best_dev_response_ce=best_dev_ce,
        best_step=best_step, best_epoch=best_epoch, global_step=global_step,
        control="planner_predicted", weighting="reply_eos_token_weighted", selection_uses_test=False,
        chat_quality_evaluated=False, model_version=config.get("model_version"),
        version=config.get("version"), data_version=config.get("data_version"))
    _atomic_json(directory / "loss_selection.json", selection)
    manifest = dict(format_version=3, status=status, training_signature=signature,
        method="emostance_frozen_qwen3_continuous_stance_prefix", objective="reply_ce",
        generator_selection_mode="dev_response_ce", selection_criterion="dev_response_ce", best_metric="dev_response_ce",
        base_model_path=str(generator.base_model_path), architecture=generator.architecture(),
        train_examples=len(train_rows), epochs_requested=epochs, epochs_complete=epoch,
        global_step=global_step, history=history, best_dev_response_ce=best_dev_ce, best_step=best_step,
        best_epoch=best_epoch, best_loss=best_loss if math.isfinite(best_loss) else None,
        best_quality_score=None, quality_status="not_evaluated", chat_quality_status="not_evaluated",
        quality_report=None, quality_report_sha256=None, checkpoint_sha256=best_sha,
        last_checkpoint=str(last_path), best_checkpoint=str(best_path), config=config,
        model_version=config.get("model_version"), version=config.get("version"),
        data_version=config.get("data_version"), frozen_lm=True, selection_uses_test=False,
        prompt_format="qwen3_chat_no_thinking_v1", enable_thinking=False,
        loss_tokens="response content plus EOS; prompt/prefix masked", base_weights_copied=False,
        gradient_checkpointing_enabled=gradient_checkpointing, runtime_checkpoint_interval=checkpoint_interval,
        dev_evaluation_interval=eval_interval, dev_control="planner_predicted", dev_weighting="reply_eos_token_weighted",
        loss_selection_report="generator/loss_selection.json", dev_subset_file="generator/dev_subset.json")
    _atomic_json(manifest_path, manifest)
    return manifest


def train_generator(train_rows: list[dict[str, Any]], space: Any, run_dir: str | Path,
                    config: dict[str, Any], dev_rows=None) -> dict[str, Any]:
    """Train a frozen-base prefix with teacher KL and chat-based selection.

    Development chats rank periodic checkpoints; development response CE only
    breaks ties. Holdout chats are unavailable to this function. Rejected
    prefixes never produce a deployable best checkpoint.
    """
    if not train_rows:
        raise ValueError("Generator training set is empty")
    if any(row.get("split", row.get("source_split", "train")) != "train" for row in train_rows):
        raise ValueError("Generator accepts training rows only, never dev/test rows")
    config = dict(config)
    config["vector_dim"] = int(space.vector_dim)
    config["run_dir"] = str(Path(run_dir).resolve())
    selection_mode = config.get("generator_selection_mode")
    if selection_mode == "dev_response_ce":
        return _train_loss_only_generator(train_rows, space, run_dir, config, dev_rows)
    if selection_mode not in (None, "smoke_only", "chat_quality_dev_then_dev_response_ce"):
        raise ValueError(f"Unsupported generator_selection_mode: {selection_mode}")
    quality_enabled = bool(config.get("chat_quality_enabled", False))
    if quality_enabled and not dev_rows:
        raise ValueError("Chat-quality training needs development rows, never test rows")
    if quality_enabled and any(row.get("split", "dev") not in {"dev", "valid", "validation"} for row in dev_rows):
        raise ValueError("Development response selection must not use train/test rows")
    teacher_weight = float(config.get("prefix_teacher_kl_weight", 1.0))
    teacher_temperature = float(config.get("prefix_teacher_temperature", 1.0))
    if (not math.isfinite(teacher_weight) or teacher_weight < 0 or not math.isfinite(teacher_temperature)
            or teacher_temperature <= 0 or (quality_enabled and teacher_weight == 0)):
        raise ValueError("Quality-preserving training requires positive teacher KL settings; smoke may use zero KL")
    directory = Path(run_dir).resolve() / "generator"
    directory.mkdir(parents=True, exist_ok=True)
    signature = _training_signature(train_rows, space, config, dev_rows=dev_rows)
    manifest_path, last_path = directory / "manifest.json", directory / "last.pt"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("training_signature") != signature:
            raise ValueError("Existing generator run has different training data/config; choose a new run_dir")
        if previous.get("status") == "complete":
            if quality_enabled and previous.get("quality_status") == "rejected":
                return previous
            if not (directory / "best.pt").is_file():
                raise FileNotFoundError("Completed generator is missing best.pt; restore its trained checkpoint")
            return previous
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = PaperGenerator(config, checkpoint=last_path if last_path.exists() else None)
    optimizer = torch.optim.AdamW(generator.projector.parameters(), lr=float(config.get("prefix_learning_rate", 1e-4)))
    epoch, offset, global_step, epoch_loss, epoch_steps = 0, 0, 0, 0.0, 0
    best_loss, history = float("inf"), []
    best_quality_score, best_dev_ce, best_step = -1.0, float("inf"), None
    last_quality_step = -1
    if generator.checkpoint_state is not None:
        state = generator.checkpoint_state
        if state.get("training_signature") != signature:
            raise ValueError("Generator checkpoint belongs to different training data/config")
        optimizer.load_state_dict(state["optimizer"])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(generator.device)
        epoch, offset, global_step = state["next_epoch"], state["next_offset"], state["global_step"]
        epoch_loss, epoch_steps = state["epoch_loss_sum"], state["epoch_steps"]
        best_loss, history = state["best_loss"], state["history"]
        best_quality_score = state.get("best_quality_score", -1.0)
        best_dev_ce = state.get("best_dev_response_ce", float("inf"))
        best_step = state.get("best_step")
        last_quality_step = state.get("last_quality_step", -1)
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
    ce_policy = {}
    for row in train_rows:
        _, reason = response_ce_weight(row, config)
        ce_policy[reason] = ce_policy.get(reason, 0) + 1
    _atomic_json(directory / "training_target_policy.json",
                 {"counts": ce_policy, "all_rows_retained_for_teacher_kl": True, "persona_rule_is_heuristic": True,
                  "excluded_references": "Only hard-reference CE is disabled; rows still train teacher KL."})
    planner, baseline, dev_subset, dev_vectors, preservation = None, None, [], [], []
    if quality_enabled:
        from .paper_planner import load_planner
        from .chat_quality import evaluate_chat_quality, load_cases
        from .paper_pipeline import build_quality_report, bind_quality_report
        planner = load_planner(run_dir, config)
        selected = np.random.default_rng(seed).choice(len(dev_rows),
            size=min(int(config.get("prefix_dev_subset_size", 128)), len(dev_rows)), replace=False)
        dev_subset = [dev_rows[int(i)] for i in sorted(selected)]
        dev_vectors = planner.predict_rows(dev_subset)["target_vector"]
        baseline_path = directory / "quality_baseline_dev.json"
        if baseline_path.exists():
            cached = json.loads(baseline_path.read_text(encoding="utf-8"))
            if cached.get("training_signature") != signature:
                raise ValueError("Cached no-prefix baseline belongs to another training run")
            baseline = cached["report"]
        else:
            baseline = evaluate_chat_quality(generator, planner,
                load_cases(config["chat_quality_dev_file"]), config, use_prefix=False)
            _atomic_json(baseline_path, {"training_signature": signature, "report": baseline})
        anchors_path = config.get("language_preservation_train_file")
        if anchors_path:
            anchors = [json.loads(line) for line in Path(anchors_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(row.get("split") != "preservation_train" for row in anchors):
                raise ValueError("Language preservation anchors need their own training split")
            for row in anchors:
                preservation.append((row, planner.predict(row["history"], row["next_role"])["target_vector"]))
        chosen_path = directory / "quality_dev.json"
        if chosen_path.exists():
            chosen = json.loads(chosen_path.read_text(encoding="utf-8"))
            if chosen.get("training_signature") != signature:
                raise ValueError("Selected quality checkpoint belongs to another run")
            best_quality_score = float(chosen["candidate"]["score"])
            best_dev_ce = float(chosen["dev_response_ce"])
            best_step = chosen["global_step"]
    # Planner construction consumes RNG while initializing its encoder. Restore
    # the saved training RNG after every setup action, not before construction.
    if generator.checkpoint_state is not None:
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
    # Recomputation and checkpoint frequency are execution settings: changing
    # them preserves data, architecture, optimizer and the resumable signature.
    gradient_checkpointing = bool(config.get("generator_runtime_gradient_checkpointing",config.get("gradient_checkpointing", True)))
    if gradient_checkpointing:
        generator.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    generator.model.config.use_cache = False
    generator.model.train()
    generator.projector.train()
    epochs = int(config.get("prefix_epochs", 1))
    checkpoint_interval = max(1, int(config.get("generator_runtime_checkpoint_interval",config.get("checkpoint_interval", 100))))
    step_limit = config.get("generator_max_steps")
    quality_interval = max(1, int(config.get("chat_quality_interval", 2000)))
    preservation_interval = max(1, int(config.get("language_preservation_interval", 8)))
    preservation_weight = float(config.get("language_preservation_weight", 1.0))
    preservation_ce_weight = float(config.get("language_preservation_ce_weight", 1.0))
    steps_this_call = 0
    started = time.time()

    def checkpoint(next_epoch: int, next_offset: int, best: bool = False) -> None:
        state = {"format_version": 1, "base_model_path": str(generator.base_model_path),
                 "prompt_format": "qwen3_chat_no_thinking_v1",
                 "architecture": generator.architecture(), "projector": generator.projector.state_dict(),
                 "optimizer": optimizer.state_dict(), "training_signature": signature,
                 "next_epoch": next_epoch, "next_offset": next_offset, "global_step": global_step,
                 "epoch_loss_sum": epoch_loss, "epoch_steps": epoch_steps, "best_loss": best_loss,
                 "objective": "compatible_reply_ce_plus_no_prefix_teacher_kl",
                 "best_quality_score": best_quality_score, "best_dev_response_ce": best_dev_ce,
                 "best_step": best_step, "last_quality_step": last_quality_step,
                 "history": history, "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        _atomic_checkpoint(directory / ("best.pt" if best else "last.pt"), state)

    def assess_quality():
        nonlocal best_quality_score, best_dev_ce, best_step, last_quality_step
        if not quality_enabled or global_step == last_quality_step:
            return
        checkpoint(epoch, offset)
        report = build_quality_report(config, generator, planner, phase="dev",
                                      checkpoint_path=last_path, baseline=baseline)
        previous_mode = generator.model.training
        generator.model.eval()
        weighted_ce, tokens = 0.0, 0
        try:
            with torch.inference_mode():
                for row, vector in zip(dev_subset, dev_vectors):
                    n = generator.response_ids(row["response"]).numel()
                    weighted_ce += float(generator.loss(row, vector)) * n
                    tokens += n
        finally:
            generator.model.train(previous_mode)
        dev_ce = weighted_ce / tokens
        if not math.isfinite(dev_ce):
            raise RuntimeError("Nonfinite development response CE")
        report.update(global_step=global_step, dev_response_ce=dev_ce,
                      training_signature=signature, development_response_examples=len(dev_subset))
        score = float(report["candidate"]["score"])
        accepted = report["status"] == "passed"
        improved = accepted and (score > best_quality_score or
                                 (score == best_quality_score and dev_ce < best_dev_ce))
        last_quality_step = global_step
        if improved:
            best_quality_score, best_dev_ce, best_step = score, dev_ce, global_step
            checkpoint(epoch, offset, best=True)
            chosen = bind_quality_report(report, config, "dev", directory / "best.pt")
            _atomic_json(directory / "quality_dev.json", chosen)
        _atomic_json(directory / f"quality_step_{global_step:06d}.json", report)
        record = {"event": "chat_quality", "step": global_step, "score": score,
                  "baseline_score": baseline["score"], "passed": accepted, "selected": improved,
                  "dev_response_ce": dev_ce, "best_step": best_step,
                  "reasons": report["comparison"].get("reasons", [])}
        with (directory / "quality_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        # Persist RNGs only after the evaluation context has restored them.
        checkpoint(epoch, offset)

    try:
        while epoch < epochs:
            order = np.random.default_rng(seed + epoch).permutation(len(train_rows))
            for position in range(offset, len(order)):
                row = train_rows[int(order[position])]
                distribution = row.get("target_meme_distribution", row.get("meme_distribution"))
                if not distribution:
                    raise ValueError(f"Missing real target annotations: {row.get('sample_id', position)}")
                _, vector = space.targets(distribution)
                optimizer.zero_grad(set_to_none=True)
                components = (generator.distillation_losses(row, vector, teacher_temperature)
                              if teacher_weight else {"ce": generator.loss(row, vector),
                                                      "teacher_kl": torch.zeros((), device=generator.device)})
                ce_weight, _ = response_ce_weight(row, config)
                loss = ce_weight * components["ce"] + teacher_weight * components["teacher_kl"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite generator loss at step {global_step + 1}")
                loss.backward()
                anchor_kl, anchor_ce = 0.0, 0.0
                if preservation and (global_step + 1) % preservation_interval == 0:
                    anchor, anchor_vector = preservation[(global_step // preservation_interval) % len(preservation)]
                    anchor_losses = generator.distillation_losses(anchor, anchor_vector, teacher_temperature)
                    preservation_loss = preservation_weight * (anchor_losses["teacher_kl"] +
                        preservation_ce_weight * anchor_losses["ce"])
                    if not torch.isfinite(preservation_loss):
                        raise RuntimeError("Nonfinite language-preservation KL")
                    preservation_loss.backward()
                    anchor_kl = float(anchor_losses["teacher_kl"].detach())
                    anchor_ce = float(anchor_losses["ce"].detach())
                    del anchor_losses, preservation_loss
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.projector.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(grad_norm)) or float(grad_norm) == 0:
                    raise RuntimeError("No finite gradient reached the prefix projector through frozen Qwen3")
                optimizer.step()
                value = float(loss.detach().cpu()) + preservation_weight * (anchor_kl + preservation_ce_weight * anchor_ce)
                global_step += 1
                steps_this_call += 1
                epoch_loss += value
                epoch_steps += 1
                offset = position + 1
                progress = {"stage": "generator", "epoch": epoch + 1, "epoch_step": offset,
                            "epoch_total": len(train_rows), "step": global_step, "loss": value,
                            "reply_ce": float(components["ce"].detach()),
                            "teacher_kl": float(components["teacher_kl"].detach()),
                            "reply_ce_weight": ce_weight, "anchor_teacher_kl": anchor_kl,
                            "anchor_ce": anchor_ce,
                            "prefix_grad_norm": float(grad_norm), "elapsed_seconds": round(time.time() - started, 3)}
                del components, loss
                with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                if global_step % 20 == 0 or steps_this_call == 1:
                    print(json.dumps(progress), flush=True)
                if global_step % checkpoint_interval == 0:
                    checkpoint(epoch, offset)
                if quality_enabled and (global_step in (100, 500, 1000) or global_step % quality_interval == 0):
                    assess_quality()
                # Cooperative interruption saves only at complete update boundaries.
                if (directory / "STOP_REQUESTED").exists():
                    checkpoint(epoch, offset)
                    break
                if step_limit is not None and steps_this_call >= int(step_limit) and offset < len(train_rows):
                    checkpoint(epoch, offset)
                    break
            if offset < len(train_rows):
                break
            if quality_enabled:
                assess_quality()
            metric = epoch_loss / epoch_steps
            history.append({"epoch": epoch + 1, "train_objective": metric, "steps": epoch_steps})
            epoch += 1
            offset, epoch_loss, epoch_steps = 0, 0.0, 0
            if metric < best_loss:
                best_loss = metric
                if not quality_enabled:
                    checkpoint(epoch, offset, best=True)
            checkpoint(epoch, offset)
            if step_limit is not None and steps_this_call >= int(step_limit):
                break
    except BaseException:
        # A failed backward/optimizer step is not counted; offset remains at the
        # last completed update and can be replayed after recovery.
        checkpoint(epoch, offset)
        raise
    finally:
        if gradient_checkpointing:
            generator.model.gradient_checkpointing_disable()
        generator.model.eval()
        generator.model.config.use_cache = True
        generator.projector.eval()
    status = "complete" if epoch >= epochs else "incomplete"
    chosen_file = directory / "quality_dev.json"
    chosen = json.loads(chosen_file.read_text(encoding="utf-8")) if chosen_file.exists() else None
    manifest = {"format_version": 2, "status": status, "training_signature": signature,
                "method": "emostance_frozen_qwen3_continuous_stance_prefix",
                "base_model_path": str(generator.base_model_path), "architecture": generator.architecture(),
                "train_examples": len(train_rows), "epochs_requested": epochs, "epochs_complete": epoch,
                "global_step": global_step, "history": history,
                "selection_criterion": "chat_quality_dev_then_dev_response_ce" if quality_enabled else "smoke_only",
                "best_metric": "chat_quality_dev_then_dev_response_ce" if quality_enabled else "smoke_only",
                "best_quality_score": best_quality_score if chosen else None,
                "best_dev_response_ce": best_dev_ce if chosen else None, "best_step": best_step,
                "best_loss": best_loss if math.isfinite(best_loss) else None,
                "quality_status": "dev_passed" if chosen else "rejected" if quality_enabled else "not_evaluated",
                "quality_report": "generator/quality_dev.json" if chosen else None,
                "quality_report_sha256": hashlib.sha256(chosen_file.read_bytes()).hexdigest() if chosen else None,
                "checkpoint_sha256": chosen["checkpoint_sha256"] if chosen else None,
                "last_checkpoint": str(last_path),
                "best_checkpoint": str(directory / "best.pt") if (directory / "best.pt").exists() else None,
                "config": config, "frozen_lm": True, "selection_uses_test": False,
                "prompt_format": "qwen3_chat_no_thinking_v1", "enable_thinking": False,
                "truncation_side": "oldest whole turns, then training-only content suffix", "loss_tokens": "response content plus EOS; prompt/prefix masked",
                "base_weights_copied": False,
                "gradient_checkpointing_enabled": gradient_checkpointing,
                "runtime_checkpoint_interval": checkpoint_interval}
    _atomic_json(manifest_path, manifest)
    return manifest


def smoke_train_step(row: dict[str, Any], space: Any, run_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Explicit real-backbone one-update smoke API; never starts full training."""
    smoke = {**config, "prefix_epochs": 1, "generator_max_steps": 1, "checkpoint_interval": 1,
             "chat_quality_enabled": False, "generator_selection_mode": "smoke_only"}
    return train_generator([row], space, run_dir, smoke)
