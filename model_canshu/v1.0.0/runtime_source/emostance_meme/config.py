from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[2]


def local_backbone(path: str | Path | None = None) -> Path:
    candidates = [Path(path).expanduser()] if path else []
    if not path:
        candidates.extend([
            Path("/home/aida102/emoji/Mistral-7B-Instruct-v0.3"),
            MODEL_ROOT.parent / "EmoStance" / "Mistral-7B-Instruct-v0.3",
        ])
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "config.json").is_file():
            continue
        index = candidate / "model.safetensors.index.json"
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            if not mapping or not all((candidate / name).is_file() for name in set(mapping.values())):
                raise ValueError(f"本地模型分片缺失：{candidate}")
        elif not any((candidate / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise ValueError(f"目录中没有本地模型权重：{candidate}")
        return candidate.resolve()
    raise ValueError("未找到本地 Mistral。请在 config.json 中设置 base_model 的绝对路径；不会下载模型。")


def work_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = MODEL_ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(MODEL_ROOT.resolve()):
        raise ValueError(f"生成文件必须放在 model/ 下：{resolved}")
    return resolved


@dataclass
class Config:
    base_model: str = "/home/aida102/emoji/Mistral-7B-Instruct-v0.3"
    annotations: list[str] | None = None
    texts: list[str] | None = None
    run_dir: str = "runs/main"
    seed: int = 42
    max_confidence: float = 5
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    vector_dim: int = 32
    num_clusters: int = 8
    max_context_tokens: int = 1024
    max_reply_tokens: int = 128
    feature_batch_size: int = 1
    head_hidden_size: int = 256
    head_epochs: int = 30
    head_batch_size: int = 64
    head_lr: float = 1e-3
    prefix_tokens: int = 8
    projector_hidden_size: int = 512
    prefix_epochs: int = 1
    prefix_lr: float = 1e-4
    gradient_checkpointing: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    distance: str = "cosine"
    idle_timeout: float = 10.0
    device: str = "auto"
    dtype: str = "auto"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else MODEL_ROOT / "config.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"未知配置项：{sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self) -> None:
        import math
        integer_fields = ("seed", "vector_dim", "num_clusters", "max_context_tokens", "max_reply_tokens", "feature_batch_size", "head_hidden_size", "head_epochs", "head_batch_size", "prefix_tokens", "projector_hidden_size", "prefix_epochs")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"配置 {name} 必须为整数")
        positive = ("max_confidence", "vector_dim", "num_clusters", "max_context_tokens", "max_reply_tokens", "feature_batch_size", "head_hidden_size", "head_epochs", "head_batch_size", "head_lr", "prefix_tokens", "projector_hidden_size", "prefix_epochs", "prefix_lr", "idle_timeout")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"配置 {name} 必须为有限正数")
        if not 0 <= self.dev_ratio < 1 or not 0 <= self.test_ratio < 1 or self.dev_ratio + self.test_ratio >= 1:
            raise ValueError("dev_ratio + test_ratio 必须小于 1")
        if self.max_reply_tokens < 2:
            raise ValueError("max_reply_tokens 至少为 2，须包含回复 token 与 EOS")
        if self.distance not in {"cosine", "euclidean"}:
            raise ValueError("distance 必须为 cosine 或 euclidean")
        if self.dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError("不支持的 dtype")
        if not 0 < self.top_p <= 1 or not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("非法采样配置")
        for name in ("annotations", "texts"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value)):
                raise ValueError(f"{name} 必须为文件/文件夹路径的字符串数组")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("base_model 必须为本地模型目录")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(self.do_sample, bool):
            raise ValueError("gradient_checkpointing 和 do_sample 必须为布尔值")
        work_path(self.run_dir)

    def to_dict(self) -> dict:
        return asdict(self)
