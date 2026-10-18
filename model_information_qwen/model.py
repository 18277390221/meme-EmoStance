"""Load completed local Qwen EmoStance runs; no startup download or training."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "model_qwen"
DEFAULT_CONFIG_PATH = TRAINING_ROOT / "inference_config.json"
DEFAULT_MODEL_DIR = TRAINING_ROOT / "runs/qwen3_8b_v2_selected"
_IMPORT_LOCK = threading.RLock()
_DEVICE = re.compile(r"(?:auto|cpu|cuda(?::\d+)?|mps)\Z")
_DTYPES = {"auto", "float32", "float16", "bfloat16"}
_REQUIRED = (
    "manifest.json", "planner/best.pt", "planner/config.json",
    "planner/emotion_space.json", "planner/manifest.json",
    "planner/encoder_config/config.json", "planner/tokenizer/tokenizer.json",
    "planner/tokenizer/tokenizer_config.json", "generator/best.pt",
    "generator/manifest.json", "space.json", "response_vectors.npz",
    "response_anchors.jsonl", "response_index_manifest.json",
    "generator/quality_dev.json", "generator/quality_holdout.json",
)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required Qwen model file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Model JSON must contain an object: {path}")
    return value


def _runtime():
    """Import only the Qwen package, without modifying global import search paths."""
    source = TRAINING_ROOT / "src/emostance_qwen"
    package = "_local_emostance_qwen_" + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    with _IMPORT_LOCK:
        if package not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                package, source / "__init__.py", submodule_search_locations=[str(source)])
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import local Qwen runtime: {source}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[package] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(package, None)
                raise
        return package


def _configuration(model_dir=None, config_path=None):
    path = Path(config_path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    default_path = DEFAULT_CONFIG_PATH.expanduser().resolve()
    fallback = TRAINING_ROOT / "paper_config.json"
    if path == default_path and not path.is_file() and fallback.is_file():
        warnings.warn(
            f"Default Qwen inference configuration is missing: {path}; using {fallback}. "
            "All completion, quality and explicit deployment-selection checks remain required.",
            RuntimeWarning, stacklevel=2,
        )
        path = fallback.resolve()
    module = importlib.import_module(f"{_runtime()}.paper_pipeline")
    config = dict(module.load_config(path))
    if model_dir is not None:
        config["run_dir"] = str(Path(model_dir).expanduser().resolve())
    for name in ("run_dir", "data_dir", "base_model_path"):
        value = config.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Qwen configuration requires local {name}")
        selected = Path(value).expanduser()
        config[name] = str((selected if selected.is_absolute() else path.parent / selected).resolve())
    return path, config


def _weight(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local model weight is missing or empty: {path}")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"Model weight is a Git LFS pointer: {path}")


def _backbone(config):
    root = Path(config["base_model_path"])
    metadata = _read_json(root / "config.json")
    if metadata.get("model_type") != "qwen3":
        raise ValueError(f"Expected a local Qwen3 backbone, got {metadata.get('model_type')!r}: {root}")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        mapping = _read_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"Local Qwen shard index has no weight_map: {index}")
        for name in set(mapping.values()):
            if not isinstance(name, str):
                raise ValueError("Qwen shard filenames must be strings")
            shard = (root / name).resolve()
            if not shard.is_relative_to(root):
                raise ValueError(f"Qwen shard path escapes its directory: {name}")
            _weight(shard)
    else:
        weight = next((root / name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file()), None)
        if weight is None:
            raise FileNotFoundError(f"Local Qwen weights are missing: {root}; download during inference is disabled")
        _weight(weight)
    if not (root / "tokenizer.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Local Qwen tokenizer files are missing: {root}")


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _quality_release(config):
    pipeline = importlib.import_module(f"{_runtime()}.paper_pipeline")
    return pipeline.require_inference_release(config)


def _deployment_selection(config):
    pipeline = importlib.import_module(f"{_runtime()}.paper_pipeline")
    return pipeline.load_deployment_selection(config)


def _ready(config, *, verify):
    root = Path(config["run_dir"])
    manifest = _read_json(root / "manifest.json")
    for name in ("manifest.json", "planner/manifest.json", "generator/manifest.json"):
        state = _read_json(root / name)
        if state.get("status") != "complete":
            raise ValueError(f"Qwen training is not complete: {root / name} (status={state.get('status')!r})")
    for name in _REQUIRED:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required trained Qwen artifact is missing: {root / name}")
    _weight(root / "planner/best.pt")
    _weight(root / "generator/best.pt")
    # Require a passed quality release or a fully validated explicit selection,
    # including verify=False. Rejected quality reports retain their real status.
    _quality_release(config)
    train_file = Path(config["data_dir"]) / "train.jsonl"
    if not train_file.is_file():
        raise FileNotFoundError(f"Training response bank source is missing: {train_file}")
    # Index provenance is mandatory even when optional file hashes are skipped.
    index = _read_json(root / "response_index_manifest.json")
    if index.get("status") != "complete":
        raise ValueError("The Qwen response index has not completed")
    if index.get("split") != "train":
        raise ValueError("The response index must contain training examples only")
    for key, path in (
        ("train_file_sha256", train_file),
        ("planner_checkpoint_sha256", root / "planner/best.pt"),
        ("space_sha256", root / "space.json"),
    ):
        if index.get(key) != _sha256(path):
            raise ValueError(f"Response index provenance mismatch: {key}")
    trained = manifest.get("config", {})
    if isinstance(trained, dict):
        for name in ("base_model_path", "data_dir", "vector_dim", "prefix_length", "prefix_hidden_dim", "system_prompt"):
            if name in trained and trained[name] != config.get(name):
                raise ValueError(f"Serving configuration differs from the completed training run: {name}")
    files = manifest.get("files", manifest.get("artifact_sha256"))
    if verify and files is not None:
        if not isinstance(files, dict):
            raise ValueError("Invalid trained-file hash manifest")
        for relative, entry in files.items():
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError("Invalid trained-file hash path")
            path = (root / relative).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Trained-file hash path escapes the run: {relative}")
            expected = entry.get("sha256") if isinstance(entry, dict) else entry
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError(f"Invalid SHA-256 entry: {relative}")
            if _sha256(path) != expected:
                raise ValueError(f"Trained-file SHA-256 mismatch: {relative}")
    return manifest


def inspect_model_version(model_dir=None, *, config_path=None, dialogue_policy_path=None,
                          dialogue_policy_release_path=None):
    """Inspect status without constructing Torch modules or loading model weights."""
    path, config = _configuration(model_dir, config_path)
    root = Path(config["run_dir"])
    manifest = _ready(config, verify=False)
    from .dialogue_policy import resolve_policy
    policy = resolve_policy(config, path, policy_path=dialogue_policy_path, release_path=dialogue_policy_release_path)
    generator = _read_json(root / "generator/manifest.json")
    selection = _deployment_selection(config)
    quality_status = generator.get("quality_status", manifest.get("quality_status", "unknown"))
    release_mode = (selection["release_mode"] if selection is not None
                    else manifest.get("release_mode", "strict_quality_acceptance"))
    release_status = selection["status"] if selection is not None else "quality_passed"
    selected_step = (selection["selected_checkpoint"]["global_step"] if selection is not None
                     else generator.get("best_step", manifest.get("selected_step")))
    base_version = manifest.get("version", root.name)
    version = base_version+"+"+policy["version"] if policy["enabled"] else base_version
    return {"version": version, "base_model_version":base_version, "status": manifest["status"],
            "model_dir": str(root), "run_dir": str(root), "config_path": str(path),
            "data_dir": config["data_dir"], "base_model_path": config["base_model_path"],
            "base_model_exists": Path(config["base_model_path"]).is_dir(),
            "file_hashes_verified": False, "quality_status": quality_status,
            "release_status": release_status, "release_mode": release_mode,
            "selected_step": selected_step, "inference_only": True,
            "dialogue_policy": {key:value for key,value in policy.items() if key != "system_prompt"}}


def load_engine(model_dir=None, *, device=None, dtype=None, verify=True, config_path=None,
                dialogue_policy_path=None, dialogue_policy_release_path=None):
    if not isinstance(verify, bool):
        raise ValueError("verify must be a boolean")
    path, config = _configuration(model_dir, config_path)
    selected = config.get("device", "auto") if device is None else device
    if not isinstance(selected, str) or not _DEVICE.fullmatch(selected):
        raise ValueError(f"Unsupported inference device: {selected!r}")
    if selected == "auto":
        import torch
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = selected
    selected_dtype = config.get("dtype", "auto") if dtype is None else dtype
    if dtype is None and selected == "cpu":
        selected_dtype = "float32"
    if not isinstance(selected_dtype, str) or selected_dtype not in _DTYPES:
        raise ValueError(f"Unsupported inference dtype: {selected_dtype!r}")
    config["dtype"] = selected_dtype
    manifest = _ready(config, verify=verify)
    from .dialogue_policy import resolve_policy
    policy = resolve_policy(config, path, policy_path=dialogue_policy_path, release_path=dialogue_policy_release_path)
    _backbone(config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    module = importlib.import_module(f"{_runtime()}.paper_engine")
    engine = module.PaperMemeEngine(config)
    engine.dialogue_policy_state = policy
    engine.model_version = manifest.get("version", Path(config["run_dir"]).name)
    return engine


def load_chat_engine(model_dir=None, *, device=None, dtype=None, verify=True,
                     settings_path=None, config_path=None, dialogue_policy_path=None,
                     dialogue_policy_release_path=None):
    from .chat_engine import EmotionalChatEngine, load_chat_settings
    settings = load_chat_settings(settings_path)
    _, config = _configuration(model_dir, config_path)
    if settings["asr_enabled"]:
        raise ValueError("Recorded Qwen chat evaluation requires ASR editing disabled and one reply generation")
    behavior = {"max_prompt_tokens": ("chat_quality_max_prompt_tokens", "max_prompt_tokens"),
                "max_reply_tokens": ("chat_quality_max_new_tokens", "max_new_tokens"),
                "do_sample": ("do_sample", "do_sample"),
                "temperature": ("chat_quality_temperature", "temperature"),
                "top_p": ("chat_quality_top_p", "top_p"),
                "no_repeat_ngram_size": ("chat_quality_no_repeat_ngram_size", "no_repeat_ngram_size")}
    for setting, (quality_key, configured) in behavior.items():
        if settings[setting] != config.get(quality_key, config.get(configured)):
            raise ValueError(f"Chat generation settings differ from the recorded evaluation settings: {setting}")
    if config.get("top_k", 20) != 20 or config.get("chat_quality_top_k", 20) != 20:
        raise ValueError("Chat generator supports only the recorded evaluation top_k=20")
    if settings_path is not None and settings["role_prompt"] != config.get("system_prompt"):
        raise ValueError("Chat role prompt differs from the recorded Qwen chat evaluation")
    raw = load_engine(model_dir, device=device, dtype=dtype, verify=verify, config_path=config_path,
                      dialogue_policy_path=dialogue_policy_path, dialogue_policy_release_path=dialogue_policy_release_path)
    if settings_path is None and raw.config.get("system_prompt"):
        settings["role_prompt"] = raw.config["system_prompt"]
    policy = raw.dialogue_policy_state
    settings["role_prompt"] = policy["system_prompt"]
    engine = EmotionalChatEngine(raw, settings)
    engine.dialogue_policy_state = {key:value for key,value in policy.items() if key != "system_prompt"}
    base_version = getattr(raw, "model_version", Path(raw.config["run_dir"]).name)
    engine.model_version = base_version+"+"+policy["version"] if policy["enabled"] else base_version
    return engine
