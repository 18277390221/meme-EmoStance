from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .backbone import LocalMistral
from .config import Config, MODEL_ROOT, local_backbone, work_path
from .data import prepare_dataset, save_prepared
from .networks import OrientationPredictor, PrefixProjector, reply_loss, soft_cross_entropy
from .retrieval import ResponseIndex
from .space import EmotionSpace


def read_paths(paths):
    """Input paths are relative to model/, while external inputs are read only."""
    return [str((MODEL_ROOT / path).resolve()) if not Path(path).is_absolute() else path for path in paths or []]


def prepare(config: Config):
    if not config.annotations:
        raise ValueError("请先在 model/config.json 的 annotations 中设置数据文件路径；当前未配置训练数据。")
    dataset = prepare_dataset(read_paths(config.annotations), text_paths=read_paths(config.texts),
        max_confidence=config.max_confidence, seed=config.seed, dev_ratio=config.dev_ratio,
        test_ratio=config.test_ratio, strict=True)
    target = work_path(config.run_dir) / "prepared"
    target.mkdir(parents=True, exist_ok=True)
    save_prepared(dataset, target)
    return dataset


def _role(value):
    return {"A": "user", "B": "assistant"}.get(value, value)


def _labelled(rows):
    return [row for row in rows if row.get("meme_distribution")]


def _arrays(space, rows):
    pairs = [space.targets(row["meme_distribution"]) for row in rows]
    return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])


def _source_lookup(records):
    return {(str(row["dialogue_id"]), row["turn_id"]): row for row in records}


def _transition(space, samples, lookup):
    counts = np.full((space.num_clusters, space.num_clusters), 0.1, dtype=np.float32)
    for row in samples:
        source = lookup.get((str(row["dialogue_id"]), row["turn_id"] - 1))
        if source and _role(source["role"]) == "user" and source.get("meme_distribution"):
            q_source, _ = space.targets(source["meme_distribution"])
            q_target, _ = space.targets(row["meme_distribution"])
            counts += np.outer(q_source, q_target) * float(row.get("weight", 1.0))
    return counts / counts.sum(1, keepdims=True)


def _features(backbone, rows, *, mode, batch_size):
    if mode == "context":
        conversations = [row["messages"] for row in rows]
    elif mode == "source":
        conversations = [[{"role": "user", "content": row["messages"][-1]["content"]}] for row in rows]
    else:
        conversations = [[{"role": "user", "content": row["reply"]}] for row in rows]
    return backbone.features(conversations, batch_size=batch_size)


def _tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _weighted_mse(prediction, target, weights):
    values = (prediction.float() - target).square().mean(-1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def fit_heads(predictor, space, turn_rows, turn_features, samples, context_features, source_features, config, validation=None):
    """Train cached-feature heads, not the frozen 7B backbone."""
    device = next(predictor.parameters()).device
    q_turn, v_turn = _arrays(space, turn_rows)
    q_target, v_target = _arrays(space, samples)
    turn_weights = np.asarray([float(row.get("weight", 1.0)) for row in turn_rows], dtype=np.float32)
    target_weights = np.asarray([float(row.get("weight", 1.0)) for row in samples], dtype=np.float32)
    frequencies = (q_target * target_weights[:, None]).sum(0)
    class_weights = np.power(np.maximum(frequencies, 1e-6), -0.25)
    class_weights /= class_weights.mean()
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.head_lr, weight_decay=0.01)
    rng = np.random.default_rng(config.seed)
    best = float("inf")
    best_state = None
    history = []
    for epoch in range(config.head_epochs):
        predictor.train()
        epoch_losses = []
        # A and B are BOTH supervised; the utterance scorer will later score
        # generated B replies, not only the user's latest utterance.
        for indices in np.array_split(rng.permutation(len(turn_rows)), max(1, int(np.ceil(len(turn_rows) / config.head_batch_size)))):
            weights = _tensor(turn_weights[indices], device)
            output = predictor.utterance(_tensor(turn_features[indices], device))
            loss = 0.4 * soft_cross_entropy(output["logits"], _tensor(q_turn[indices], device), weights)
            loss += 0.1 * _weighted_mse(output["direct_vector"], _tensor(v_turn[indices], device), weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
        for indices in np.array_split(rng.permutation(len(samples)), max(1, int(np.ceil(len(samples) / config.head_batch_size)))):
            output = predictor(_tensor(context_features[indices], device), _tensor(source_features[indices], device))
            target = _tensor(q_target[indices] * class_weights, device)
            weights = _tensor(target_weights[indices], device)
            loss = soft_cross_entropy(output["logits"], target, weights)
            loss += 0.2 * soft_cross_entropy(output["prior_free_logits"], target, weights)
            loss += 0.1 * _weighted_mse(output["vector"], _tensor(v_target[indices], device), weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        predictor.eval()
        metric = float(np.mean(epoch_losses))
        val_loss = None
        if validation and validation[0]:
            val_rows, val_context, val_source = validation
            q_val, _ = _arrays(space, val_rows)
            with torch.no_grad():
                losses = []
                for start in range(0, len(val_rows), config.head_batch_size):
                    stop = start + config.head_batch_size
                    output = predictor(_tensor(val_context[start:stop], device), _tensor(val_source[start:stop], device))
                    losses.extend((-( _tensor(q_val[start:stop], device) * output["logits"].log_softmax(-1)).sum(-1)).cpu().tolist())
                val_loss = float(np.mean(losses))
            metric = val_loss
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), "dev_loss": val_loss})
        print(json.dumps(history[-1]), flush=True)
        if metric < best:
            best = metric
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in predictor.state_dict().items()})
    predictor.load_state_dict(best_state)
    predictor.eval()
    return history


def _input_hashes(config):
    result = {}
    for raw in read_paths(config.annotations) + read_paths(config.texts):
        path = Path(raw)
        paths = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".jsonl"}) if path.is_dir() else [path]
        for current in paths:
            digest = hashlib.sha256()
            with current.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[str(current)] = digest.hexdigest()
    return result


def train(config: Config):
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    run_dir = work_path(config.run_dir)
    if (run_dir / "manifest.json").exists():
        raise ValueError("该 run_dir 已有完成的模型，请使用新的运行目录，避免覆盖。")
    dataset = prepare(config)
    samples = _labelled(dataset["splits"]["train"])
    turn_rows = _labelled(dataset["turn_records"]["train"])
    if not samples or not turn_rows:
        raise ValueError("训练集没有带真实文本及表情标注的助手回复。dev/test 记录不会自动转为 train。")
    backbone = LocalMistral(config.base_model, device=config.device, dtype=config.dtype, max_context_tokens=config.max_context_tokens)
    print(f"Caching frozen Mistral features: {len(turn_rows)} labelled turns, {len(samples)} reply examples", flush=True)
    turn_features = _features(backbone, turn_rows, mode="utterance", batch_size=config.feature_batch_size)
    context_features = _features(backbone, samples, mode="context", batch_size=config.feature_batch_size)
    source_features = _features(backbone, samples, mode="source", batch_size=config.feature_batch_size)
    space = EmotionSpace.fit(turn_rows, turn_features, vector_dim=config.vector_dim, num_clusters=config.num_clusters, seed=config.seed)
    transition = _transition(space, samples, _source_lookup(turn_rows))
    predictor = OrientationPredictor(backbone.hidden_size, config.head_hidden_size, space.prototypes, transition).to(backbone.device)
    dev_all = _labelled(dataset["splits"]["dev"])
    known = set(space.meme_ids)
    dev_rows = [row for row in dev_all if set(row["meme_distribution"]).issubset(known)]
    validation = None
    if dev_rows:
        validation = (dev_rows, _features(backbone, dev_rows, mode="context", batch_size=config.feature_batch_size),
                      _features(backbone, dev_rows, mode="source", batch_size=config.feature_batch_size))
    head_history = fit_heads(predictor, space, turn_rows, turn_features, samples, context_features, source_features, config, validation)
    prefix = PrefixProjector(space.vector_dim, config.projector_hidden_size, backbone.hidden_size, config.prefix_tokens).to(backbone.device)
    optimizer = torch.optim.AdamW(prefix.parameters(), lr=config.prefix_lr, weight_decay=0.01)
    if config.gradient_checkpointing:
        backbone.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backbone.model.train()
    backbone.model.config.use_cache = False
    prefix_history = []
    weights = np.asarray([float(row.get("weight", 1.0)) for row in samples])
    mean_weight = max(float(weights.mean()), 1e-8)
    for epoch in range(config.prefix_epochs):
        losses = []
        prefix.train()
        for step, index in enumerate(np.random.permutation(len(samples))):
            row = samples[index]
            _, vector = space.targets(row["meme_distribution"])
            prompt = backbone.prompt_ids(row["messages"])
            target = backbone.response_ids(row["reply"], config.max_reply_tokens)
            loss = reply_loss(backbone.model, prefix, _tensor(vector[None], backbone.device), prompt, target)
            weighted = loss * float(weights[index]) / mean_weight
            optimizer.zero_grad(set_to_none=True)
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(prefix.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 20 == 0:
                print(json.dumps({"stage": "prefix", "epoch": epoch + 1, "step": step + 1, "loss": losses[-1]}), flush=True)
        prefix_history.append({"epoch": epoch + 1, "reply_ce": float(np.mean(losses))})
    if config.gradient_checkpointing:
        backbone.model.gradient_checkpointing_disable()
    backbone.model.config.use_cache = True
    backbone.model.eval()
    prefix.eval()
    # The index contains TRAINING REPLIES only. Their vectors use the same
    # scorer as generated replies, so distances live in the same learned space.
    response_features = _features(backbone, samples, mode="utterance", batch_size=config.feature_batch_size)
    index_records = []
    with torch.no_grad():
        for start in range(0, len(samples), config.head_batch_size):
            vectors = predictor.utterance(_tensor(response_features[start:start + config.head_batch_size], backbone.device))["vector"].cpu().numpy()
            for row, vector in zip(samples[start:start + config.head_batch_size], vectors):
                index_records.append({"dialogue_id": row["dialogue_id"], "turn_id": row["turn_id"], "reply": row["reply"],
                                      "meme_id": row["meme_id"], "vector": vector.tolist(), "weight": row["weight"], "split": "train"})
    ResponseIndex(index_records, metric=config.distance).save(run_dir / "responses.json")
    space.save(run_dir / "space.json")
    torch.save({"predictor": predictor.cpu().state_dict(), "projector": prefix.cpu().state_dict()}, run_dir / "adapters.pt")
    config_values = config.to_dict()
    config_values["base_model"] = str(local_backbone(config.base_model))
    manifest = {"format_version": 1, "status": "complete", "config": config_values,
                "feature_dim": backbone.hidden_size, "vector_dim": space.vector_dim, "num_clusters": space.num_clusters,
                "train_reply_count": len(samples), "train_dialogue_ids": sorted({str(row["dialogue_id"]) for row in samples}),
                "dev_known_meme_count": len(dev_rows), "dev_unseen_meme_count": len(dev_all) - len(dev_rows),
                "head_history": head_history, "prefix_history": prefix_history, "input_sha256": _input_hashes(config),
                "method": "local_mistral_frozen_features_role_aware_prototype_prefix",
                "meme_query": "generated_reply_orientation", "index_split": "train"}
    # A complete manifest is written LAST; incomplete runs cannot be served.
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
