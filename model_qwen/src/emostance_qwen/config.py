"""Local-only Qwen backbone and output path validation."""
from pathlib import Path
import json

MODEL_ROOT = Path(__file__).resolve().parents[2]


def work_path(value):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else MODEL_ROOT / path).resolve()
    if not path.is_relative_to(MODEL_ROOT):
        raise ValueError(f"Generated files must remain below model_qwen/: {path}")
    return path


def local_backbone(path=None):
    root = Path(path or MODEL_ROOT / "backbones/Qwen3-8B").expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError(f"Expected the Qwen3 backbone: {root}")
    index = root / "model.safetensors.index.json"
    names = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    if not names or any(not (root / name).is_relative_to(root) or not (root / name).is_file() or (root / name).stat().st_size < 1024 for name in names):
        raise ValueError(f"Missing Qwen3 safetensors weights: {root}; run download first")
    return root
