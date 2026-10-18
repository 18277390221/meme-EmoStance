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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "model_qwen"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("inference_config.json")
DEFAULT_MODEL_DIR = TRAINING_ROOT / "runs/qwen3_8b_v3_loss_ce"
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
    "generator/dev_subset.json", "generator/loss_selection.json", "evaluation.json",
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


def _ready(config, *, verify, config_path=None, settings_path=None):
    from .release import require_inference_release
    root = Path(config["run_dir"])
    manifest = _read_json(root / "manifest.json")
    for name in _REQUIRED:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required trained Qwen V3 artifact is missing: {root / name}")
    _weight(root / "planner/best.pt")
    _weight(root / "generator/best.pt")
    # This independent loss-only authorization never changes the original
    # training manifest or calls the earlier version's chat-quality gate.
    require_inference_release(config, config_path or DEFAULT_CONFIG_PATH, settings_path)
    return manifest


def _create_raw_engine(config, config_path, settings_path=None):
    """Retain the shared raw reply implementation with a V3-only constructor."""
    from .release import require_inference_release
    require_inference_release(config, config_path, settings_path)
    package = _runtime()
    engine_module = importlib.import_module(f"{package}.paper_engine")
    planner_module = importlib.import_module(f"{package}.paper_planner")
    generator_module = importlib.import_module(f"{package}.paper_generator")

    class LossOnlyMemeEngine(engine_module.PaperMemeEngine):
        def __init__(self):
            import torch
            torch.set_num_threads(8)
            self.config = config
            self.planner = planner_module.load_planner(Path(config["run_dir"]), config)
            self.generator = generator_module.PaperGenerator(config, checkpoint=Path(config["run_dir"])/"generator/best.pt")
            self.index = engine_module.PaperResponseIndex(config)
            self._lock = threading.Lock()

    return LossOnlyMemeEngine()


def inspect_model_version(model_dir=None, *, config_path=None, dialogue_policy_path=None,
                          dialogue_policy_release_path=None):
    """Inspect status without constructing Torch modules or loading model weights."""
    path, config = _configuration(model_dir, config_path)
    root = Path(config["run_dir"])
    manifest = _ready(config, verify=False, config_path=path)
    from .dialogue_policy import resolve_policy
    policy = resolve_policy(config, path, policy_path=dialogue_policy_path, release_path=dialogue_policy_release_path)
    generator = _read_json(root / "generator/manifest.json")
    from .release import DEFAULT_RELEASE_PATH, RELEASE_MODE
    release = _read_json(DEFAULT_RELEASE_PATH)
    quality_status = generator.get("quality_status", "not_evaluated")
    base_version = manifest["model_version"]
    version = base_version+"+"+policy["version"] if policy["enabled"] else base_version
    return {"version": version, "base_model_version":manifest["version"], "model_version":manifest["model_version"],
            "status": manifest["status"], "training_status":manifest["training_status"],
            "model_dir": str(root), "run_dir": str(root), "config_path": str(path),
            "data_dir": config["data_dir"], "base_model_path": config["base_model_path"],
            "base_model_exists": Path(config["base_model_path"]).is_dir(),
            "file_hashes_verified": True, "quality_status": quality_status,
            "chat_quality_status":"not_evaluated", "release_status":release["status"], "release_mode":RELEASE_MODE,
            "selected_step":release["selected_checkpoint"]["step"],
            "best_dev_response_ce":release["selected_checkpoint"]["dev_response_ce"], "inference_only": True,
            "dialogue_policy": {key:value for key,value in policy.items() if key != "system_prompt"}}


def load_engine(model_dir=None, *, device=None, dtype=None, verify=True, config_path=None,
                dialogue_policy_path=None, dialogue_policy_release_path=None, settings_path=None):
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
    manifest = _ready(config, verify=verify, config_path=path, settings_path=settings_path)
    from .dialogue_policy import resolve_policy
    policy = resolve_policy(config, path, policy_path=dialogue_policy_path, release_path=dialogue_policy_release_path)
    _backbone(config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    engine = _create_raw_engine(config, path, settings_path)
    engine.dialogue_policy_state = policy
    engine.model_version = manifest["model_version"]
    return engine


def load_chat_engine(model_dir=None, *, device=None, dtype=None, verify=True,
                     settings_path=None, config_path=None, dialogue_policy_path=None,
                     dialogue_policy_release_path=None):
    from .chat_engine import EmotionalChatEngine, load_chat_settings
    settings = load_chat_settings(settings_path)
    _, config = _configuration(model_dir, config_path)
    from .release import validate_settings
    validate_settings(config, settings)
    raw = load_engine(model_dir, device=device, dtype=dtype, verify=verify, config_path=config_path,
                      dialogue_policy_path=dialogue_policy_path, dialogue_policy_release_path=dialogue_policy_release_path,
                      settings_path=settings_path)
    if settings_path is None and raw.config.get("system_prompt"):
        settings["role_prompt"] = raw.config["system_prompt"]
    policy = raw.dialogue_policy_state
    settings["role_prompt"] = policy["system_prompt"]
    engine = EmotionalChatEngine(raw, settings)
    engine.dialogue_policy_state = {key:value for key,value in policy.items() if key != "system_prompt"}
    base_version = getattr(raw, "model_version", Path(raw.config["run_dir"]).name)
    engine.model_version = base_version+"+"+policy["version"] if policy["enabled"] else base_version
    return engine
