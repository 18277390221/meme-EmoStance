"""Offline, role-aware EmoStance orientation planner.

Implements the source-entropy gate from paper Appendix C.5, including both
directions of a dyadic conversation. Only training rows construct transition
counts and class weights. The local DeBERTa encoder is trained; Qwen3 is not
used here. No original EmoStance source or model file is modified.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .config import work_path
from .space import EmotionSpace

ROLE_PAIRS = ("A->B", "B->A", "A->A", "B->B")
ROLE_IDS = {"A": 0, "B": 1}
TRANSITION_IDS = {value: index for index, value in enumerate(ROLE_PAIRS)}


def _get(config: Any, key: str, default: Any) -> Any:
    return config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)


def _options(config: Any) -> dict[str, Any]:
    return {
        "planner_model_path": str(_get(config, "planner_model_path", "/home/aida102/emoji/deberta-v3-base")),
        "seed": int(_get(config, "seed", 42)),
        "epochs": int(_get(config, "planner_epochs", 3)),
        "batch_size": int(_get(config, "planner_batch_size", 8)),
        "max_length": int(_get(config, "planner_max_length", 320)),
        "learning_rate": float(_get(config, "planner_learning_rate", 1.5e-5)),
        "weight_decay": float(_get(config, "planner_weight_decay", .01)),
        "warmup_ratio": float(_get(config, "planner_warmup_ratio", .06)),
        "role_dim": int(_get(config, "planner_role_dim", 32)),
        "dropout": float(_get(config, "planner_dropout", .1)),
        "transition_alpha": float(_get(config, "planner_transition_alpha", .05)),
        "transition_weight": float(_get(config, "planner_transition_weight", .5)),
        "source_loss_weight": float(_get(config, "planner_source_loss_weight", .4)),
        "target_loss_weight": float(_get(config, "planner_target_loss_weight", 1.)),
        "text_aux_weight": float(_get(config, "planner_text_aux_weight", .2)),
        "vector_loss_weight": float(_get(config, "planner_vector_loss_weight", .1)),
        "class_weight_power": float(_get(config, "planner_class_weight_power", .25)),
        "checkpoint_steps": int(_get(config, "planner_checkpoint_steps", 200)),
        "log_steps": int(_get(config, "planner_log_steps", 20)),
        "device": str(_get(config, "device", "auto")),
    }


def _device(requested: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)


def normalize_role(value: str) -> str:
    mapping = {"A": "A", "B": "B", "user": "A", "assistant": "B"}
    try:
        return mapping[str(value)]
    except KeyError as exc:
        raise ValueError(f"Expected role A/B or user/assistant, got {value!r}") from exc


def serialize_history(history: Sequence[Mapping[str, Any]], next_role: str = "B", situation: str = "") -> str:
    if not history:
        raise ValueError("The planner requires at least one observed utterance")
    lines = []
    for message in history:
        text = str(message.get("text", message.get("content", "")))
        if not text.strip():
            raise ValueError("History contains an empty utterance")
        lines.append(f'{normalize_role(message["role"])}: {text}')
    prefix = f"Situation: {situation}\n" if situation else ""
    normalize_role(next_role)  # The next-speaker marker enters role embeddings.
    return prefix + "Dialogue:\n" + "\n".join(lines)


def _history(row: Mapping[str, Any]) -> list[dict[str, str]]:
    history = row.get("history", row.get("messages"))
    if not isinstance(history, list) or not history:
        raise ValueError("Each source/response sample needs nonempty history")
    return [{"role": normalize_role(item["role"]), "text": str(item.get("text", item.get("content", "")))} for item in history]


def _roles(row: Mapping[str, Any]) -> tuple[str, str]:
    history = _history(row)
    source = normalize_role(row.get("source_role", row.get("role", history[-1]["role"])))
    target = normalize_role(row.get("next_role", row.get("target_role", "B" if source == "A" else "A")))
    if history[-1]["role"] != source:
        raise ValueError("source_role disagrees with the final observed history turn")
    return source, target


def serialize_row(row: Mapping[str, Any]) -> str:
    return serialize_history(_history(row), _roles(row)[1], str(row.get("situation", "")))


def _labels(rows: Sequence[Mapping[str, Any]], space: EmotionSpace) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {key: [] for key in ("source_cluster", "target_cluster", "source_vector", "target_vector")}
    for row in rows:
        for side in ("source", "target"):
            distribution = row.get(f"{side}_meme_distribution")
            if not isinstance(distribution, Mapping) or not distribution:
                raise ValueError(f"Sample {row.get('sample_id')} has no {side} meme distribution")
            cluster, vector = space.targets(distribution)
            values[f"{side}_cluster"].append(cluster)
            values[f"{side}_vector"].append(vector)
    return {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}


def training_statistics(rows: Sequence[Mapping[str, Any]], space: EmotionSpace, alpha: float = .05, power: float = .25) -> dict[str, np.ndarray]:
    """Train-only soft transitions and mean-one inverse-frequency weights."""
    if not rows or alpha <= 0 or power < 0:
        raise ValueError("Training rows, positive smoothing, and nonnegative power are required")
    for row in rows:
        if row.get("split", row.get("source_split", "train")) != "train":
            raise ValueError("Transition counts and prototypes must only use train rows")
    labels = _labels(rows, space)
    counts = np.full((len(ROLE_PAIRS), space.num_clusters, space.num_clusters), alpha, dtype=np.float64)
    for index, row in enumerate(rows):
        source, target = _roles(row)
        counts[TRANSITION_IDS[f"{source}->{target}"]] += np.outer(labels["source_cluster"][index], labels["target_cluster"][index])
    result = {"transitions": (counts / counts.sum(-1, keepdims=True)).astype(np.float32)}
    for side in ("source", "target"):
        frequencies = labels[f"{side}_cluster"].mean(0)
        weights = np.power(1. / np.maximum(frequencies, 1e-6), power)
        result[f"{side}_weights"] = (weights / weights.mean()).astype(np.float32)
        probabilities = labels[f"{side}_cluster"].astype(np.float64)
        vectors = labels[f"{side}_vector"].astype(np.float64)
        mass = probabilities.sum(0)
        prototypes = (probabilities.T @ vectors) / np.maximum(mass[:, None], 1e-12)
        prototypes[mass <= 1e-12] = vectors.mean(0)
        result[f"{side}_prototypes"] = prototypes.astype(np.float32)
    return result


class PaperOrientationPlanner(nn.Module):
    def __init__(self, encoder: nn.Module, space: EmotionSpace, transitions: np.ndarray, *, role_dim: int = 32, dropout: float = .1, transition_weight: float = .5, source_prototypes: np.ndarray | None = None):
        super().__init__()
        self.encoder = encoder
        hidden = int(encoder.config.hidden_size)
        self.num_clusters = space.num_clusters
        self.transition_weight = transition_weight
        self.dropout = nn.Dropout(dropout)
        self.source_cluster = nn.Linear(hidden, space.num_clusters)
        self.source_vector = nn.Linear(hidden, space.vector_dim)
        self.role_embedding = nn.Embedding(2, role_dim)
        self.next_role_embedding = nn.Embedding(2, role_dim)
        self.transition_embedding = nn.Embedding(len(ROLE_PAIRS), role_dim)
        target_size = hidden + role_dim * 3 + space.num_clusters + space.vector_dim
        self.target_mlp = nn.Sequential(nn.Linear(target_size, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout))
        self.target_cluster = nn.Linear(hidden // 2, space.num_clusters)
        self.graph_gate = nn.Linear(hidden // 2, 1)
        self.register_buffer("transitions", torch.as_tensor(transitions, dtype=torch.float32))
        self.register_buffer("prototypes", torch.as_tensor(space.prototypes, dtype=torch.float32))
        self.register_buffer("source_prototypes", torch.as_tensor(space.prototypes if source_prototypes is None else source_prototypes, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, role_id: torch.Tensor, next_role_id: torch.Tensor, transition_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(encoded.last_hidden_state[:, 0]).to(self.source_cluster.weight.dtype)
        source_logits = self.source_cluster(pooled)
        source_prob = source_logits.softmax(-1)
        source_vector = self.source_vector(pooled)
        target_features = torch.cat([pooled, self.role_embedding(role_id), self.next_role_embedding(next_role_id), self.transition_embedding(transition_idx), source_prob.detach(), source_vector.detach()], -1)
        hidden = self.target_mlp(target_features)
        target_text_logits = self.target_cluster(hidden)
        prior = torch.bmm(source_prob.unsqueeze(1), self.transitions[transition_idx]).squeeze(1)
        prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.num_clusters > 1:
            source_entropy = -(source_prob * source_prob.clamp_min(1e-12).log()).sum(-1, keepdim=True) / math.log(self.num_clusters)
        else:
            source_entropy = torch.zeros_like(source_prob[:, :1])
        gate = (1. - source_entropy).clamp(0., 1.) * self.graph_gate(hidden).sigmoid()
        target_logits = target_text_logits + self.transition_weight * gate * prior.clamp_min(1e-12).log()
        target_prob = target_logits.softmax(-1)
        return {"source_logits": source_logits, "source_distribution": source_prob, "source_vector": source_vector,
                "source_realized_vector": source_prob @ self.source_prototypes, "target_logits": target_logits,
                "target_text_logits": target_text_logits, "target_distribution": target_prob,
                "target_vector": target_prob @ self.prototypes, "gate": gate, "prior": prior}


class _PlannerDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int, space: EmotionSpace | None = None):
        self.rows = list(rows)
        self.encoded = tokenizer([serialize_row(row) for row in rows], max_length=max_length, truncation=True, padding=False, return_token_type_ids=False) if rows else {"input_ids": [], "attention_mask": []}
        self.labels = _labels(rows, space) if rows and space is not None else {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, target = _roles(self.rows[index])
        result = {"input_ids": self.encoded["input_ids"][index], "attention_mask": self.encoded["attention_mask"][index], "role_id": ROLE_IDS[source], "next_role_id": ROLE_IDS[target], "transition_idx": TRANSITION_IDS[f"{source}->{target}"]}
        result.update({key: value[index] for key, value in self.labels.items()})
        return result


class _Collator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        result = self.tokenizer.pad({"input_ids": [item["input_ids"] for item in batch], "attention_mask": [item["attention_mask"] for item in batch]}, return_tensors="pt", pad_to_multiple_of=8)
        for key in ("role_id", "next_role_id", "transition_idx"):
            result[key] = torch.tensor([item[key] for item in batch], dtype=torch.long)
        for key in ("source_cluster", "target_cluster", "source_vector", "target_vector"):
            if key in batch[0]:
                result[key] = torch.from_numpy(np.stack([item[key] for item in batch]))
        return result


def planner_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], source_weights: torch.Tensor, target_weights: torch.Tensor, options: Mapping[str, Any]) -> torch.Tensor:
    def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return -(labels * weights * logits.log_softmax(-1)).sum(-1).mean()
    source = weighted_ce(output["source_logits"], batch["source_cluster"], source_weights)
    target = weighted_ce(output["target_logits"], batch["target_cluster"], target_weights)
    text_aux = weighted_ce(output["target_text_logits"], batch["target_cluster"], target_weights)
    vector = F.mse_loss(output["source_vector"], batch["source_vector"]) + F.mse_loss(output["target_vector"], batch["target_vector"])
    return options["source_loss_weight"] * source + options["target_loss_weight"] * target + options["text_aux_weight"] * text_aux + options["vector_loss_weight"] * vector


def _forward(model: PaperOrientationPlanner, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(**{key: batch[key] for key in ("input_ids", "attention_mask", "role_id", "next_role_id", "transition_idx")})


@torch.no_grad()
def evaluate_planner(model: PaperOrientationPlanner, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {key: 0. for key in ("target_soft_ce", "source_soft_ce", "target_jsd", "target_vector_mse", "target_overlap", "gate")}
    count = 0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        output = _forward(model, batch)
        target = batch["target_cluster"]
        predicted = output["target_distribution"]
        midpoint = (target + predicted) * .5
        jsd = .5 * ((target * (target.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1) + (predicted * (predicted.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1))
        values = {"target_soft_ce": -(target * output["target_logits"].log_softmax(-1)).sum(-1),
                  "source_soft_ce": -(batch["source_cluster"] * output["source_logits"].log_softmax(-1)).sum(-1),
                  "target_jsd": jsd, "target_vector_mse": (output["target_vector"] - batch["target_vector"]).square().mean(-1),
                  "target_overlap": torch.minimum(target, predicted).sum(-1), "gate": output["gate"].flatten()}
        for key, value in values.items():
            totals[key] += float(value.sum().cpu())
        count += len(target)
    return {"examples": count, **{key: value / max(count, 1) for key, value in totals.items()}}


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _log(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def _save(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _signature(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]], space: EmotionSpace, options: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, rows in (("train", train_rows), ("dev", dev_rows)):
        digest.update(name.encode())
        for row in rows:
            digest.update(json.dumps(dict(row), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())
            digest.update(b"\n")
    digest.update(json.dumps(dict(options), sort_keys=True).encode())
    for value in (space.centroids, space.membership, space.prototypes):
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    digest.update(json.dumps(space.meme_ids).encode())
    return digest.hexdigest()


def train_planner(train_rows: Sequence[Mapping[str, Any]], dev_rows: Sequence[Mapping[str, Any]], space: EmotionSpace, run_dir: str | Path, config: Any) -> dict[str, Any]:
    """Train all rows and resume an interrupted run from its last checkpoint.

    ``run_dir`` must be inside model_qwen/. Best selection is exclusively development
    target soft CE; test rows must not be passed here. Confidence has already
    entered the soft meme labels, matching the paper's loss (no second weighting).
    """
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    options = _options(config)
    if not train_rows or not dev_rows:
        raise ValueError("Both nonempty training and development rows are required")
    if set(str(row["dialogue_id"]) for row in train_rows) & set(str(row["dialogue_id"]) for row in dev_rows):
        raise ValueError("A dialogue occurs in both training and development data")
    for rows, expected in ((train_rows, "train"), (dev_rows, "dev")):
        for row in rows:
            actual = row.get("split", row.get("source_split", expected))
            if actual not in ({"dev", "valid", "validation"} if expected == "dev" else {"train"}):
                raise ValueError(f"Expected {expected} planner rows, found {actual}")
    for name in ("epochs", "batch_size", "max_length", "checkpoint_steps", "log_steps"):
        if options[name] <= 0:
            raise ValueError(f"planner {name} must be positive")
    root = work_path(run_dir)
    out = root / "planner"
    out.mkdir(parents=True, exist_ok=True)
    statistics = training_statistics(train_rows, space, options["transition_alpha"], options["class_weight_power"])
    # Match the released training code: response prototypes use training target
    # occurrences, source prototypes use training source occurrences.
    space.prototypes = statistics["target_prototypes"].astype(np.float64)
    signature = _signature(train_rows, dev_rows, space, options)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_signature") != signature:
            raise ValueError("Completed planner belongs to different data/config; use a new run_dir")
        if manifest.get("status") == "complete":
            if not (out / "best.pt").is_file():
                raise FileNotFoundError("Completed planner is missing best.pt; restore its trained checkpoint")
            return manifest
    space.save(root / "space.json")
    seed = options["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(options["device"])
    model_path = Path(options["planner_model_path"]).expanduser().resolve()
    if not model_path.is_dir():
        raise ValueError(f"Local DeBERTa directory not found: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True, local_files_only=True)
    tokenizer.truncation_side = "left"  # Retain the latest observed turn.
    train_data = _PlannerDataset(train_rows, tokenizer, options["max_length"], space)
    dev_data = _PlannerDataset(dev_rows, tokenizer, options["max_length"], space)
    encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.float32)
    model = PaperOrientationPlanner(encoder, space, statistics["transitions"], role_dim=options["role_dim"], dropout=options["dropout"], transition_weight=options["transition_weight"], source_prototypes=statistics["source_prototypes"]).float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    steps_per_epoch = math.ceil(len(train_data) / options["batch_size"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps_per_epoch * options["epochs"] * options["warmup_ratio"]), steps_per_epoch * options["epochs"])
    source_weights = torch.as_tensor(statistics["source_weights"], device=device)
    target_weights = torch.as_tensor(statistics["target_weights"], device=device)
    collator = _Collator(tokenizer)
    dev_loader = DataLoader(dev_data, batch_size=options["batch_size"], shuffle=False, collate_fn=collator, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed + 100000))
    tokenizer.save_pretrained(out / "tokenizer")
    encoder.config.save_pretrained(out / "encoder_config")
    space.save(out / "emotion_space.json")
    _json(out / "config.json", {**options, "training_signature": signature, "num_clusters": space.num_clusters, "vector_dim": space.vector_dim, "role_pairs": list(ROLE_PAIRS), "gate": "(1 - normalized source entropy) * sigmoid(gate)", "confidence_weighting": "confidence-weighted soft labels; no second per-example confidence loss weight"})
    epoch_start, next_batch, global_step = 0, 0, 0
    best_dev, best_epoch = float("inf"), 0
    last_path = out / "last.pt"
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_signature") != signature:
            raise ValueError("Resume checkpoint belongs to different data/config; use a new run_dir")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch_start, next_batch, global_step = checkpoint["epoch"], checkpoint["next_batch"], checkpoint["global_step"]
        best_dev, best_epoch = checkpoint["best_dev"], checkpoint["best_epoch"]
        _restore_rng(checkpoint["rng"])
        del checkpoint
        _log(out / "metrics.jsonl", {"event": "resume", "epoch": epoch_start + 1, "next_batch": next_batch, "global_step": global_step})
    started = time.monotonic()

    def checkpoint(epoch: int, batch: int) -> None:
        _save(last_path, {"format_version": 1, "training_signature": signature, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "next_batch": batch, "global_step": global_step, "best_dev": best_dev, "best_epoch": best_epoch, "rng": _rng_state()})

    for epoch in range(epoch_start, options["epochs"]):
        indices = np.random.default_rng(seed + epoch).permutation(len(train_data)).tolist()
        # A dedicated loader generator avoids perturbing restored dropout RNG.
        loader = DataLoader(train_data, batch_size=options["batch_size"], sampler=indices, collate_fn=collator, pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        train_total, train_count, last_batch = 0., 0, 0
        for batch_index, batch in enumerate(loader):
            last_batch = batch_index + 1
            if epoch == epoch_start and batch_index < next_batch:
                continue
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = _forward(model, batch)
            loss = planner_loss(output, batch, source_weights, target_weights, options)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite planner loss at step {global_step + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            global_step += 1
            size = len(batch["input_ids"])
            train_total += float(loss.detach()) * size
            train_count += size
            if global_step % options["log_steps"] == 0:
                _log(out / "metrics.jsonl", {"event": "step", "epoch": epoch + 1, "batch": batch_index + 1, "batches": steps_per_epoch, "global_step": global_step, "loss": float(loss.detach()), "learning_rate": scheduler.get_last_lr()[0], "elapsed_seconds": time.monotonic() - started})
            if global_step % options["checkpoint_steps"] == 0:
                checkpoint(epoch, batch_index + 1)
        metrics = evaluate_planner(model, dev_loader, device)
        if metrics["target_soft_ce"] < best_dev:
            best_dev, best_epoch = metrics["target_soft_ce"], epoch + 1
            _save(out / "best.pt", {"format_version": 1, "training_signature": signature, "model": model.state_dict(), "epoch": best_epoch, "dev_metrics": metrics})
        _log(out / "metrics.jsonl", {"event": "epoch", "epoch": epoch + 1, "global_step": global_step, "train_loss": train_total / train_count if train_count else None, "trained_examples_this_session": train_count, "dev": metrics, "best_epoch": best_epoch, "best_dev_target_soft_ce": best_dev, "elapsed_seconds": time.monotonic() - started})
        next_batch = 0
        checkpoint(epoch + 1, 0)
    manifest = {"format_version": 1, "method": "EmoStance paper role-aware DeBERTa planner", "status": "complete", "training_signature": signature, "train_examples": len(train_rows), "dev_examples": len(dev_rows), "epochs": options["epochs"], "global_steps": global_step, "best_epoch": best_epoch, "best_dev_target_soft_ce": best_dev, "checkpoint": str(out / "best.pt"), "resume_checkpoint": str(out / "last.pt"), "elapsed_seconds_this_session": time.monotonic() - started}
    _json(manifest_path, manifest)
    return manifest


class PaperPlannerRuntime:
    def __init__(self, model: PaperOrientationPlanner, tokenizer: Any, space: EmotionSpace, options: Mapping[str, Any], device: torch.device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.space = space
        self.options = dict(options)
        self.device = device

    @torch.no_grad()
    def predict_rows(self, rows: Sequence[Mapping[str, Any]], batch_size: int | None = None) -> dict[str, np.ndarray]:
        if not rows:
            return {"source_distribution": np.empty((0, self.space.num_clusters), np.float32), "target_distribution": np.empty((0, self.space.num_clusters), np.float32), "source_vector": np.empty((0, self.space.vector_dim), np.float32), "source_realized_vector": np.empty((0, self.space.vector_dim), np.float32), "target_vector": np.empty((0, self.space.vector_dim), np.float32)}
        data = _PlannerDataset(rows, self.tokenizer, self.options["max_length"])
        loader = DataLoader(data, batch_size=batch_size or self.options["batch_size"], shuffle=False, collate_fn=_Collator(self.tokenizer))
        collected: dict[str, list[np.ndarray]] = {key: [] for key in ("source_distribution", "target_distribution", "source_vector", "source_realized_vector", "target_vector")}
        for batch in loader:
            output = _forward(self.model, {key: value.to(self.device) for key, value in batch.items()})
            for key in collected:
                collected[key].append(output[key].cpu().numpy())
        return {key: np.concatenate(value) for key, value in collected.items()}

    def predict(self, history: Sequence[Mapping[str, Any]], next_role: str = "B", situation: str = "") -> dict[str, np.ndarray]:
        result = self.predict_rows([{"history": list(history), "next_role": next_role, "situation": situation}])
        return {key: value[0] for key, value in result.items()}

    def evaluate_rows(self, rows: Sequence[Mapping[str, Any]], batch_size: int | None = None) -> dict[str, float]:
        data = _PlannerDataset(rows, self.tokenizer, self.options["max_length"], self.space)
        loader = DataLoader(data, batch_size=batch_size or self.options["batch_size"], shuffle=False, collate_fn=_Collator(self.tokenizer))
        return evaluate_planner(self.model, loader, self.device)


def load_planner(run_dir: str | Path, config: Any = None, *, device: str | None = None) -> PaperPlannerRuntime:
    """Read the best completed planner checkpoint without accessing the network."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    out = Path(run_dir).expanduser().resolve()
    if out.name != "planner":
        out = out / "planner"
    options = json.loads((out / "config.json").read_text(encoding="utf-8"))
    requested = device or str(_get(config, "device", options.get("device", "auto")))
    target_device = _device(requested)
    space = EmotionSpace.load(out / "emotion_space.json")
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True)
    tokenizer.truncation_side = "left"
    # Initialize from saved config; trained weights come exclusively from best.pt.
    encoder_config = AutoConfig.from_pretrained(out / "encoder_config", local_files_only=True)
    encoder = AutoModel.from_config(encoder_config)
    model = PaperOrientationPlanner(encoder, space, np.zeros((len(ROLE_PAIRS), space.num_clusters, space.num_clusters), np.float32), role_dim=options["role_dim"], dropout=options["dropout"], transition_weight=options["transition_weight"])
    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    if checkpoint.get("training_signature") != options.get("training_signature"):
        raise ValueError("Planner checkpoint/config signature mismatch")
    model.load_state_dict(checkpoint["model"])
    model.float().to(target_device)
    return PaperPlannerRuntime(model, tokenizer, space, options, target_device)
