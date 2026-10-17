"""Frozen local Mistral with the EmoStance continuous stance prefix.

Only the two-layer stance projector is trained. Checkpoints reference the local
backbone and contain no Mistral weights. Inputs use the official repository's
``Dialogue:\nA: ...\nB: ...`` formatting, including both response directions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import local_backbone


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
            self.projector.load_state_dict(state["projector"])
            self.checkpoint_state = state

    def architecture(self) -> dict[str, int]:
        return {"vector_dim": self.projector.vector_dim, "lm_hidden_size": self.projector.hidden_size,
                "prefix_length": self.projector.prefix_length, "prefix_hidden_dim": self.projector.intermediate_size}

    def prompt_ids(self, history: list[dict[str, Any]], next_role: str, situation: str = "") -> torch.Tensor:
        prompt = generation_prompt(history, next_role, situation)
        encoded = self.tokenizer(prompt, truncation=True, max_length=int(self.config.get("max_prompt_tokens", 384)),
                                 add_special_tokens=True, return_tensors="pt")
        return encoded["input_ids"].to(self.device)

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

    @torch.inference_mode()
    def generate(self, history: list[dict[str, Any]], next_role: str, vector: Any, max_new_tokens: int = 128) -> str:
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        self.model.eval()
        self.projector.eval()
        self.model.config.use_cache = True
        ids = self.prompt_ids(history, next_role)
        embeds = self.model.get_input_embeddings()(ids)
        prefix = self.projector(self._stance(vector)).to(embeds.dtype)
        inputs = torch.cat([prefix, embeds], dim=1)
        options = {"max_new_tokens": max_new_tokens, "do_sample": bool(self.config.get("do_sample", False)),
                   "pad_token_id": self.tokenizer.pad_token_id, "eos_token_id": self.tokenizer.eos_token_id,
                   "use_cache": True,"max_length":None,
                   "no_repeat_ngram_size":int(self.config.get("no_repeat_ngram_size",4))}
        if options["do_sample"]:
            options.update(temperature=float(self.config.get("temperature", 0.7)), top_p=float(self.config.get("top_p", 0.9)))
        # With inputs_embeds only, HF returns continuation ids, without prompt.
        output = self.model.generate(inputs_embeds=inputs,
                                     attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device), **options)
        text = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # Match the official generator cleanup: stop before an invented next turn.
        for marker in ("\nA:", "\nB:", "\nA：", "\nB："):
            text = text.split(marker, 1)[0]
        role_prefix = _role(next_role) + ":"
        if text.startswith(role_prefix):
            text = text[len(role_prefix):]
        if not text.strip():
            raise RuntimeError("Generator returned an empty reply")
        return text.strip()


def _training_signature(rows: list[dict[str, Any]], space: Any, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = {key: row[key] for key in ("sample_id", "history", "next_role", "response", "target_meme_distribution") if key in row}
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    for name in ("meme_ids", "centroids", "membership", "prototypes"):
        value = getattr(space, name)
        digest.update(json.dumps(value.tolist() if hasattr(value, "tolist") else value, sort_keys=True).encode())
    keys = ("seed", "prefix_epochs", "prefix_length", "prefix_hidden_dim", "prefix_learning_rate",
            "max_prompt_tokens", "max_training_response_tokens", "gradient_checkpointing", "vector_dim")
    digest.update(json.dumps({key: config.get(key) for key in keys}, sort_keys=True).encode())
    return digest.hexdigest()


def train_generator(train_rows: list[dict[str, Any]], space: Any, run_dir: str | Path,
                    config: dict[str, Any]) -> dict[str, Any]:
    """Train/resume the paper prefix; config.generator_max_steps limits this call.

    Example-level shuffling is deterministic by epoch. Checkpoints store the
    next epoch/offset, optimizer, CPU/CUDA RNG, and current epoch loss sum, so an
    interrupted call resumes without repeating completed updates.
    """
    if not train_rows:
        raise ValueError("Generator training set is empty")
    if any(row.get("split", row.get("source_split", "train")) != "train" for row in train_rows):
        raise ValueError("Generator accepts training rows only, never dev/test rows")
    config = dict(config)
    config["vector_dim"] = int(space.vector_dim)
    config["run_dir"] = str(Path(run_dir).resolve())
    directory = Path(run_dir).resolve() / "generator"
    directory.mkdir(parents=True, exist_ok=True)
    signature = _training_signature(train_rows, space, config)
    manifest_path, last_path = directory / "manifest.json", directory / "last.pt"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("training_signature") != signature:
            raise ValueError("Existing generator run has different training data/config; choose a new run_dir")
        if previous.get("status") == "complete":
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
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
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
    steps_this_call = 0
    started = time.time()

    def checkpoint(next_epoch: int, next_offset: int, best: bool = False) -> None:
        state = {"format_version": 1, "base_model_path": str(generator.base_model_path),
                 "architecture": generator.architecture(), "projector": generator.projector.state_dict(),
                 "optimizer": optimizer.state_dict(), "training_signature": signature,
                 "next_epoch": next_epoch, "next_offset": next_offset, "global_step": global_step,
                 "epoch_loss_sum": epoch_loss, "epoch_steps": epoch_steps, "best_loss": best_loss,
                 "history": history, "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        _atomic_checkpoint(directory / ("best.pt" if best else "last.pt"), state)

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
                loss = generator.loss(row, vector)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite generator loss at step {global_step + 1}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(generator.projector.parameters(), 1.0, error_if_nonfinite=True)
                if not math.isfinite(float(grad_norm)) or float(grad_norm) == 0:
                    raise RuntimeError("No finite gradient reached the prefix projector through frozen Mistral")
                optimizer.step()
                value = float(loss.detach().cpu())
                global_step += 1
                steps_this_call += 1
                epoch_loss += value
                epoch_steps += 1
                offset = position + 1
                progress = {"stage": "generator", "epoch": epoch + 1, "epoch_step": offset,
                            "epoch_total": len(train_rows), "step": global_step, "loss": value,
                            "prefix_grad_norm": float(grad_norm), "elapsed_seconds": round(time.time() - started, 3)}
                with (directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(progress, allow_nan=False) + "\n")
                if global_step % 20 == 0 or steps_this_call == 1:
                    print(json.dumps(progress), flush=True)
                if global_step % checkpoint_interval == 0:
                    checkpoint(epoch, offset)
                if step_limit is not None and steps_this_call >= int(step_limit) and offset < len(train_rows):
                    checkpoint(epoch, offset)
                    break
            if offset < len(train_rows):
                break
            metric = epoch_loss / epoch_steps
            history.append({"epoch": epoch + 1, "train_reply_eos_ce": metric, "steps": epoch_steps})
            epoch += 1
            offset, epoch_loss, epoch_steps = 0, 0.0, 0
            if metric < best_loss:
                best_loss = metric
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
    manifest = {"format_version": 1, "status": status, "training_signature": signature,
                "method": "emostance_frozen_mistral_continuous_stance_prefix",
                "base_model_path": str(generator.base_model_path), "architecture": generator.architecture(),
                "train_examples": len(train_rows), "epochs_requested": epochs, "epochs_complete": epoch,
                "global_step": global_step, "history": history,
                "best_metric": "train_epoch_reply_eos_ce", "best_loss": best_loss if math.isfinite(best_loss) else None,
                "last_checkpoint": str(last_path),
                "best_checkpoint": str(directory / "best.pt") if (directory / "best.pt").exists() else None,
                "config": config, "frozen_lm": True, "selection_uses_test": False,
                "truncation_side": "left", "loss_tokens": "response content plus EOS; prompt/prefix masked",
                "base_weights_copied": False,
                "gradient_checkpointing_enabled": gradient_checkpointing,
                "runtime_checkpoint_interval": checkpoint_interval}
    _atomic_json(manifest_path, manifest)
    return manifest


def smoke_train_step(row: dict[str, Any], space: Any, run_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    """Explicit real-backbone one-update smoke API; never starts full training."""
    smoke = {**config, "prefix_epochs": 1, "generator_max_steps": 1, "checkpoint_interval": 1}
    return train_generator([row], space, run_dir, smoke)
