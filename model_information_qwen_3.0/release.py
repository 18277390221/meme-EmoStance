"""Explicit, local authorization for the completed loss-selected Qwen V3 run.

This release records the user's model choice. It does not assert chat quality.
All evidence checks run before model construction, including verify=False.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path

APPLICATION_ROOT = Path(__file__).resolve().parent
DEFAULT_RELEASE_PATH = APPLICATION_ROOT / "inference_release.json"
DEFAULT_SETTINGS_PATH = APPLICATION_ROOT / "chat_config.json"
AUTHORIZATION_TEXT = "在model_information_qwen_3.0文件夹中使用该模型，其余效果和model_information_qwen中的一样，即只是换了个模型"
RELEASE_MODE = "user_selected_loss_only"
_EXPECTED_CHECKPOINT_SHA256 = "87a8ed11e093ea073633720f420f517d4bc311fb4deb864292f1dee09919a601"
_VERSIONS = dict(version="qwen3_8b_v3_loss_ce", model_version="3.0.0", data_version="2.0.0")
_ARTIFACTS = (
    "manifest.json", "generator/manifest.json", "generator/best.pt", "generator/loss_selection.json",
    "generator/dev_subset.json", "evaluation.json", "planner/manifest.json", "planner/best.pt",
    "planner/config.json", "planner/emotion_space.json", "planner/encoder_config/config.json",
    "planner/tokenizer/tokenizer.json", "planner/tokenizer/tokenizer_config.json", "space.json",
    "response_vectors.npz", "response_anchors.jsonl", "response_index_manifest.json", "graph/manifest.json",
)


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Inference release evidence must be a JSON object: {path}")
    return value


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _pipeline():
    from .model import _runtime
    return importlib.import_module(f"{_runtime()}.paper_pipeline")


def validate_settings(config, settings):
    expected = dict(max_prompt_tokens=config.get("serving_max_prompt_tokens"),
                    max_reply_tokens=config.get("max_new_tokens"), do_sample=config.get("do_sample"),
                    temperature=config.get("temperature"), top_p=config.get("top_p"),
                    no_repeat_ngram_size=config.get("no_repeat_ngram_size"),
                    role_prompt=config.get("system_prompt"), asr_enabled=False)
    fixed = dict(max_prompt_tokens=4096, max_reply_tokens=128, do_sample=True,
                 temperature=.7, top_p=.8, no_repeat_ngram_size=4, asr_enabled=False)
    if any(expected.get(key) != value for key, value in fixed.items()):
        raise ValueError("V3 serving configuration must retain the authorized existing chat settings")
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Chat settings differ from the authorized V3 serving configuration: {key}")
    if config.get("top_k", 20) != 20 or config.get("enable_thinking") is not False:
        raise ValueError("V3 inference requires top_k=20 and thinking disabled")


def _evidence(config, config_path, settings_path):
    """Validate completed training and bind it to the unchanged serving behavior."""
    for key, expected in _VERSIONS.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected authorized V3 model version: {key}")
    root = Path(config["run_dir"]).resolve()
    pipeline = _pipeline()
    stages = pipeline.require_loss_only_training(config)
    manifest, selection = _read(root/"manifest.json"), _read(root/"generator/loss_selection.json")
    generator = stages["generator"]
    if (manifest.get("status") != "ready_for_review" or manifest.get("training_status") != "complete"
            or manifest.get("inference_enabled") is not False
            or manifest.get("inference_release_status") != "not_authorized"):
        raise ValueError("V3 source must be the completed, unchanged training review run")
    for evidence in (manifest, generator, selection):
        for key, expected in _VERSIONS.items():
            if evidence.get(key) != expected:
                raise ValueError(f"V3 training evidence version mismatch: {key}")
    for evidence in (manifest, generator):
        if (evidence.get("quality_status") != "not_evaluated"
                or evidence.get("chat_quality_status") != "not_evaluated"
                or evidence.get("selection_criterion") != "dev_response_ce"):
            raise ValueError("V3 loss-only evidence must retain truthful not_evaluated chat status")
    trained = manifest.get("config")
    if not isinstance(trained, dict):
        raise ValueError("V3 training configuration is missing")
    for key, value in trained.items():
        if key not in ("device", "dtype") and config.get(key) != value:
            raise ValueError(f"Serving changed the completed V3 training configuration: {key}")
    if (generator.get("epochs_complete") != config.get("prefix_epochs")
            or generator.get("epochs_requested") != config.get("prefix_epochs")
            or generator.get("global_step") != generator.get("train_examples", 0)*config.get("prefix_epochs", 0)
            or generator.get("global_step") != selection.get("global_step")):
        raise ValueError("V3 prefix training did not complete every requested epoch")
    if selection.get("best_step") != 150000 or not math.isclose(selection.get("best_dev_response_ce", math.inf),
            3.918008093610924, rel_tol=0, abs_tol=1e-12):
        raise ValueError("V3 authorization selects the dev-CE best checkpoint at step 150000")
    if selection.get("selection_uses_test") is not False or selection.get("chat_quality_evaluated") is not False:
        raise ValueError("V3 checkpoint must be selected only by development response CE")
    artifacts = {name:_sha(root/name) for name in _ARTIFACTS}
    if artifacts["generator/best.pt"] != _EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("V3 inference authorization refers to a different selected prefix checkpoint")
    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError("V3 completed artifact hashes are missing")
    for relative, expected in recorded.items():
        path = (root/relative).resolve()
        if (Path(relative).is_absolute() or not path.is_relative_to(root)
                or not isinstance(expected, str) or _sha(path) != expected):
            raise ValueError(f"V3 trained artifact SHA-256 mismatch: {relative}")
    evaluation = _read(root/"evaluation.json")
    if (evaluation.get("evaluation_mode") != "response_ce_only"
            or evaluation.get("checkpoint_selection") != "dev_response_ce"
            or evaluation.get("checkpoint_sha256") != artifacts["generator/best.pt"]
            or evaluation.get("chat_quality_status") != "not_evaluated"
            or evaluation.get("test_used_for_checkpoint_selection") is not False
            or evaluation.get("test_used_for_training") is not False):
        raise ValueError("Completed V3 loss evaluation does not match the selected checkpoint")
    for key, expected in _VERSIONS.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"V3 loss evaluation version mismatch: {key}")
    for split in ("dev", "test"):
        result = evaluation.get(split, {})
        ce = result.get("generator", {}).get("token_ce", {})
        if (result.get("source_file_sha256") != _sha(Path(config["data_dir"])/f"{split}.jsonl")
                or any(type(ce.get(key)) not in (int, float) or not 0 <= ce[key] < math.inf
                       for key in ("gold_control", "predicted_control"))):
            raise ValueError(f"Completed V3 {split} response CE evidence is missing or stale")
    index = _read(root/"response_index_manifest.json")
    if index.get("status") != "complete" or index.get("split") != "train":
        raise ValueError("V3 response index must be completed from the training split")
    for key, path in (("train_file_sha256", Path(config["data_dir"])/"train.jsonl"),
                      ("planner_checkpoint_sha256", root/"planner/best.pt"), ("space_sha256", root/"space.json")):
        if index.get(key) != _sha(path):
            raise ValueError(f"V3 response index provenance mismatch: {key}")
    from .chat_engine import load_chat_settings
    settings_path = Path(settings_path).resolve()
    settings = load_chat_settings(settings_path)
    validate_settings(config, settings)
    binding = dict(run_dir=str(root), artifact_sha256=artifacts,
        config_file_sha256=_sha(config_path),
        configuration_sha256=_json_sha({key:value for key,value in config.items() if key not in ("device", "dtype")}),
        training_configuration_sha256=_json_sha(trained), chat_settings_sha256=_sha(settings_path),
        resolved_chat_settings_sha256=_json_sha(settings),
        system_prompt_file_sha256=_sha(config["system_prompt_file"]),
        system_prompt_sha256=hashlib.sha256(config["system_prompt"].encode()).hexdigest())
    selected = dict(path=str(root/"generator/best.pt"), sha256=artifacts["generator/best.pt"],
                    step=selection["best_step"], best_epoch=selection["best_epoch"],
                    dev_response_ce=selection["best_dev_response_ce"],
                    training_global_step=selection["global_step"], selection_criterion="dev_response_ce")
    return binding, selected, manifest


def build_inference_release(config, config_path, settings_path=None, *, authorization_text):
    """Build a truthful release record after the user's explicit model choice."""
    if authorization_text != AUTHORIZATION_TEXT:
        raise ValueError("V3 inference release requires the recorded explicit user selection")
    binding, checkpoint, _ = _evidence(config, Path(config_path).resolve(), settings_path or DEFAULT_SETTINGS_PATH)
    return dict(format_version=1, status="ready", release_mode=RELEASE_MODE, **_VERSIONS,
        quality_status="not_evaluated", chat_quality_status="not_evaluated",
        authorization=dict(user_selection=True, source="explicit_user_instruction", text=authorization_text),
        selected_checkpoint=checkpoint, binding=binding,
        training_manifest_unchanged=True, chat_quality_evaluation_performed=False,
        limitations="Authorized loss-selected model; chat reasonableness has not been evaluated.")


def require_inference_release(config, config_path, settings_path=None, release_path=None):
    """Fail closed on missing authorization, incomplete training or stale hashes."""
    record = _read(release_path or DEFAULT_RELEASE_PATH)
    expected = build_inference_release(config, config_path, settings_path,
                                       authorization_text=record.get("authorization", {}).get("text"))
    if record != expected:
        raise ValueError("V3 inference authorization is incomplete or its bound evidence has changed")
    return record
