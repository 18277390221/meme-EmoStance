"""Load the archived, trained EmoStance model without changing the archive.

The returned object is the version's real ``PaperMemeEngine``. Call this once
at service startup, then reuse ``engine.reply(messages)`` for every request.
Messages use user/assistant roles and content; replies contain reply/meme_id.
No training, model download, or placeholder-response fallback is performed.
"""

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
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_canshu" / "V2.0.0"
_IMPORT_LOCK = threading.RLock()
_DEVICE_PATTERN = re.compile(r"(?:auto|cpu(?::\d+)?|cuda(?::\d+)?|mps(?::\d+)?)\Z")
_DTYPES = {"auto", "float32", "float16", "bfloat16"}
_RUN_FILES = (
    "manifest.json", "planner/best.pt", "planner/config.json",
    "planner/emotion_space.json", "planner/manifest.json",
    "planner/encoder_config/config.json", "planner/tokenizer/tokenizer.json",
    "planner/tokenizer/tokenizer_config.json", "generator/best.pt",
    "generator/manifest.json", "space.json", "response_vectors.npz",
    "response_anchors.jsonl", "response_index_manifest.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required model file is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid model JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model JSON must contain an object: {path}")
    return value


def _directory(model_dir: str | Path | None) -> Path:
    path = Path(model_dir).expanduser().resolve() if model_dir is not None else DEFAULT_MODEL_DIR.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model version directory does not exist: {path}")
    return path


def _resolved_path(value: Any, root: Path, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"inference_config.json requires a nonempty local {key}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _load_configuration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_json(root / "version_manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Model version is not complete: {root} (status={manifest.get('status')!r})")
    config = _read_json(root / "inference_config.json")
    for key in ("run_dir", "data_dir", "base_model_path"):
        config[key] = str(_resolved_path(config.get(key), root, key))
    if not Path(config["run_dir"]).is_relative_to(root):
        raise ValueError("Archived run_dir must be inside the selected model version directory")
    return manifest, config


def _required_files(root: Path, config: dict[str, Any]) -> list[Path]:
    run = Path(config["run_dir"])
    paths = [root / "inference_config.json", root / "runtime_source/emostance_meme/__init__.py",
             root / "runtime_source/emostance_meme/paper_engine.py"]
    paths.extend(run / name for name in _RUN_FILES)
    paths.append(Path(config["data_dir"]) / "train.jsonl")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required archived model file is missing: {path}")
    for name in ("manifest.json", "planner/manifest.json", "generator/manifest.json"):
        path = run / name
        state = _read_json(path)
        if state.get("status") != "complete":
            raise ValueError(f"Archived model stage is not complete: {path} (status={state.get('status')!r})")
    for name in ("planner/best.pt", "generator/best.pt"):
        _check_weight_file(run / name)
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_files(root: Path, manifest: dict[str, Any], required: list[Path]) -> None:
    # "source_sha256" describes the pre-export source run. The exported prefix
    # checkpoint has a different hash; only "files" hashes the archived files.
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("version_manifest.json contains no archived file hashes")
    for path in required:
        if path.is_relative_to(root) and path.relative_to(root).as_posix() not in files:
            raise ValueError(f"Required file is not covered by the version manifest: {path.relative_to(root)}")
    runtime = root / "runtime_source/emostance_meme"
    for path in runtime.rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative not in files:
            raise ValueError(f"Runtime source is not covered by the version manifest: {relative}")
    for relative, metadata in files.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError(f"Invalid archive manifest path: {relative!r}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Archive manifest path escapes the model directory: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-listed model file is missing: {path}")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("sha256"), str) or not re.fullmatch(r"[0-9a-fA-F]{64}", metadata["sha256"]):
            raise ValueError(f"Invalid SHA-256 entry in version manifest: {relative}")
        expected_size = metadata.get("bytes")
        if expected_size is not None and (isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0):
            raise ValueError(f"Invalid file size in version manifest: {relative}")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"Archived file size mismatch: {relative}")
        actual = _sha256(path)
        if actual != metadata["sha256"].lower():
            raise ValueError(f"Archived SHA-256 mismatch: {relative}; expected {metadata['sha256']}, got {actual}")


def _check_weight_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Local model weight file is missing or empty: {path}")
    with path.open("rb") as stream:
        beginning = stream.read(128)
    if beginning.startswith(b"version https://git-lfs.github.com/spec/"):
        raise ValueError(f"Model weight is a Git LFS pointer rather than local weights: {path}")


def _check_local_backbone(config: dict[str, Any]) -> None:
    root = Path(config["base_model_path"])
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"Local Mistral backbone directory/config is missing: {root}; model download is disabled")
    _read_json(root / "config.json")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = _read_json(index).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Local Mistral shard index has no weight_map: {index}")
        for filename in set(weight_map.values()):
            if not isinstance(filename, str):
                raise ValueError(f"Invalid local Mistral weight filename in {index}")
            weight = (root / filename).resolve()
            if not weight.is_relative_to(root):
                raise ValueError(f"Local Mistral shard path escapes its directory: {filename}")
            _check_weight_file(weight)
    else:
        weights = next((root / name for name in ("model.safetensors", "pytorch_model.bin") if (root / name).is_file()), None)
        if weights is None:
            raise FileNotFoundError(f"Local Mistral weights are missing: {root}; model download is disabled")
        _check_weight_file(weights)
    if not any((root / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "tokenizer.model.v3", "spiece.model", "vocab.json")):
        raise FileNotFoundError(f"Local Mistral tokenizer files are missing: {root}; model download is disabled")


def _device_dtype(config: dict[str, Any], device: str | None, dtype: str | None) -> None:
    selected = config.get("device", "auto") if device is None else device
    if not isinstance(selected, str) or not _DEVICE_PATTERN.fullmatch(selected):
        raise ValueError(f"Unsupported inference device: {selected!r}")
    if selected == "auto":
        import torch
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = selected
    selected_dtype = config.get("dtype", "auto") if dtype is None else dtype
    if dtype is None and selected.split(":", 1)[0] == "cpu":
        selected_dtype = "float32"
    if not isinstance(selected_dtype, str) or selected_dtype not in _DTYPES:
        raise ValueError(f"Unsupported inference dtype: {selected_dtype!r}")
    config["dtype"] = selected_dtype


def _private_engine_class(root: Path) -> Any:
    runtime = root / "runtime_source/emostance_meme"
    package = "_archived_emostance_" + hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:20]
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(package, runtime / "__init__.py", submodule_search_locations=[str(runtime)])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import archived model runtime: {runtime}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        try:
            spec.loader.exec_module(module)
            engine_module = importlib.import_module(f"{package}.paper_engine")
        except Exception:
            for name in tuple(sys.modules):
                if name == package or name.startswith(package + "."):
                    sys.modules.pop(name, None)
            raise
    else:
        engine_module = importlib.import_module(f"{package}.paper_engine")
    cls = getattr(engine_module, "PaperMemeEngine", None)
    if cls is None:
        raise ImportError(f"Archived runtime has no PaperMemeEngine: {runtime / 'paper_engine.py'}")
    return cls


def inspect_model_version(model_dir: str | Path | None = None) -> dict[str, Any]:
    """Read lightweight version metadata without loading Torch or model weights."""
    root = _directory(model_dir)
    manifest, config = _load_configuration(root)
    return {"version": manifest.get("version", root.name), "status": manifest["status"],
            "model_dir": str(root), "run_dir": config["run_dir"], "data_dir": config["data_dir"],
            "base_model_path": config["base_model_path"], "base_model_exists": Path(config["base_model_path"]).is_dir(),
            "selected_planner_epoch": manifest.get("selected_planner_epoch"), "prefix_global_step": manifest.get("prefix_global_step"),
            "inference_only": manifest.get("inference_only"), "manifest_file_count": len(manifest.get("files", {})),
            "file_hashes_verified": False}


def load_engine(model_dir: str | Path | None = None, *, device: str | None = None,
                dtype: str | None = None, verify: bool = True) -> Any:
    """Return the selected version's real, offline ``PaperMemeEngine``.

    ``verify=False`` skips archive SHA checks only. Required local files and
    completed-stage checks remain mandatory. CPU defaults to float32 unless
    dtype is explicitly overridden. The archive's private package does not
    replace or depend on an existing ``emostance_meme`` import.
    """
    if not isinstance(verify, bool):
        raise ValueError("verify must be a boolean")
    root = _directory(model_dir)
    manifest, config = _load_configuration(root)
    _device_dtype(config, device, dtype)
    required = _required_files(root, config)
    if verify:
        _verify_files(root, manifest, required)
    _check_local_backbone(config)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Keep this enabled after loading too: lazy relative imports made during a
    # later reply must never create __pycache__ inside the immutable archive.
    sys.dont_write_bytecode = True
    with _IMPORT_LOCK:
        engine_class = _private_engine_class(root)
        return engine_class(config)


def load_chat_engine(model_dir: str | Path | None = None, *, device: str | None = None,
                     dtype: str | None = None, verify: bool = True, settings_path=None):
    """Load training-format replies and optional ASR, sharing archived weights."""
    from .chat_engine import EmotionalChatEngine, load_chat_settings
    settings = load_chat_settings(settings_path)
    if settings["role_prompt"].strip():
        raise ValueError("当前 prefix 使用训练时的 Dialogue 格式，不支持角色系统提示词；"
                         "请设置 role_prompt_enabled=false")
    raw = load_engine(model_dir, device=device, dtype=dtype, verify=verify)
    return EmotionalChatEngine(raw, settings)
