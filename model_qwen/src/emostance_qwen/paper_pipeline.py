"""Local Qwen3-8B EmoStance training on the four-annotator V2 dataset."""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time

from .space import EmotionSpace

MODEL_DIR = Path(__file__).resolve().parents[2]
QUALITY_SELECTION_CRITERION = "chat_quality_dev_then_dev_response_ce"
USER_SELECTED_RELEASE = "user_selected_with_known_quality_failures"
LOSS_SELECTION_CRITERION = "dev_response_ce"


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=None):
    path = Path(path or MODEL_DIR/"paper_config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("run_dir", "data_dir", "base_model_path", "planner_model_path", "system_prompt_file",
                "chat_quality_dev_file", "chat_quality_holdout_file", "language_preservation_train_file",
                "planner_reuse_source"):
        if key not in config:
            if key in ("run_dir", "data_dir", "base_model_path", "planner_model_path"):
                raise ValueError(f"Missing configuration field: {key}")
            continue
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent/value if not value.is_absolute() else value).resolve())
    _run_root(config)
    if "system_prompt_file" in config:
        config["system_prompt"] = Path(config["system_prompt_file"]).read_text(encoding="utf-8").strip()
        if not config["system_prompt"]:
            raise ValueError("system_prompt_file must contain a nonempty prompt")
    config.setdefault("enable_thinking", False)
    if config["enable_thinking"] is not False:
        raise ValueError("Qwen inference requires enable_thinking=false")
    if config.get("num_candidates", 1) != 1:
        raise ValueError("Candidate generation is removed; omit num_candidates or set it to 1")
    if config.get("reranking", False) not in (False, None, "disabled", "none"):
        raise ValueError("Candidate reranking is disabled")
    if is_loss_only_training(config):
        if config.get("chat_quality_enabled") is not False:
            raise ValueError("Loss-only training requires explicit chat_quality_enabled=false")
        if not isinstance(config.get("version"), str) or not config["version"].strip():
            raise ValueError("Loss-only training requires an explicit new version")
        config.setdefault("training_objective", "response_ce_only")
    if config.get("chat_quality_enabled", False):
        paired = (("chat_quality_max_new_tokens", "max_new_tokens", 128),
                  ("chat_quality_temperature", "temperature", .7),
                  ("chat_quality_top_p", "top_p", .8), ("chat_quality_top_k", "top_k", 20),
                  ("chat_quality_no_repeat_ngram_size", "no_repeat_ngram_size", 4))
        for quality_key, runtime_key, default in paired:
            if config.get(quality_key, default) != config.get(runtime_key, default):
                raise ValueError(f"Chat quality and production generation settings must match: {runtime_key}")
        if config.get("do_sample", True) is not True:
            raise ValueError("Chat quality suites currently require production do_sample=true")
    return config


def is_loss_only_training(config):
    return config.get("generator_selection_mode") == LOSS_SELECTION_CRITERION


def _run_root(config):
    run = Path(config["run_dir"]).expanduser().resolve()
    if not run.is_relative_to(MODEL_DIR.resolve()):
        raise ValueError("Training outputs must be below model_qwen/")
    return run


def require_complete(config, *stages):
    """Do not expose partially trained checkpoints as a completed model."""
    root = _run_root(config)
    result = {}
    for stage in stages:
        path = root/stage/"manifest.json"
        if not path.is_file():
            raise ValueError(f"{stage} training has not completed: manifest is missing")
        result[stage] = json.loads(path.read_text(encoding="utf-8"))
        if result[stage].get("status") != "complete":
            raise ValueError(f"{stage} training has not completed")
    return result


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _quality_suite(config, phase):
    if phase not in ("dev", "holdout"):
        raise ValueError("Chat quality phase must be dev or holdout")
    path = config.get(f"chat_quality_{phase}_file")
    if not isinstance(path, str) or not Path(path).is_file():
        raise ValueError(f"Missing chat quality {phase} suite")
    from .chat_quality import load_cases
    return Path(path), load_cases(path)


def quality_binding(config, phase, checkpoint_path):
    """Bind recorded real outputs to the exact checkpoint, prompt and suite."""
    suite, _ = _quality_suite(config, phase)
    root = _run_root(config)
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_relative_to(root) or not checkpoint.is_file():
        raise ValueError("Quality checkpoint must be an existing file below this run")
    behavior_keys = ("base_model_path", "vector_dim", "prefix_length", "prefix_hidden_dim",
                     "max_prompt_tokens", "max_new_tokens", "do_sample", "temperature", "top_p",
                     "top_k", "no_repeat_ngram_size", "repetition_penalty", "enable_thinking")
    behavior = {key: config.get(key) for key in behavior_keys}
    behavior.update({key: value for key, value in config.items()
                     if key.startswith("chat_quality_") and not key.endswith("_file")})
    return dict(prefix_sha256=file_sha256(checkpoint),
                planner_sha256=file_sha256(root/"planner/best.pt"),
                space_sha256=file_sha256(root/"space.json"),
                system_prompt_sha256=hashlib.sha256(config["system_prompt"].encode()).hexdigest(),
                suite_file_sha256=file_sha256(suite), config_sha256=_json_sha256(behavior))


def _validate_generated_quality(report, cases, config, *, use_prefix):
    """Verify that a report retains every case and actual base/prefix reply."""
    from .chat_quality import (SEMANTIC_JUDGE_VERSION, case_system_prompt, score_reply,
                               semantic_judge_options, semantic_judge_prompt, suite_sha256)
    if not isinstance(report, dict) or report.get("use_prefix") is not use_prefix:
        raise ValueError("Chat quality report has an invalid base/prefix identity")
    if report.get("suite_sha256") != suite_sha256(cases):
        raise ValueError("Chat quality report suite does not match its source")
    prompt_hash = hashlib.sha256(config["system_prompt"].encode()).hexdigest()
    if report.get("system_prompt_sha256") != prompt_hash:
        raise ValueError("Chat quality report prompt does not match its source")
    semantic_enabled = bool(config.get("chat_quality_semantic_judge", False))
    if (report.get("semantic_judge_enabled") is not semantic_enabled
            or report.get("semantic_judge_version") != (SEMANTIC_JUDGE_VERSION if semantic_enabled else None)):
        raise ValueError("Chat quality report semantic rubric is missing or stale")
    options = dict(max_new_tokens=int(config.get("chat_quality_max_new_tokens", 128)), do_sample=True,
                   temperature=float(config.get("chat_quality_temperature", .7)),
                   top_p=float(config.get("chat_quality_top_p", .8)),
                   no_repeat_ngram_size=int(config.get("chat_quality_no_repeat_ngram_size", config.get("no_repeat_ngram_size", 4))),
                   max_prompt_tokens=int(config.get("chat_quality_max_prompt_tokens", config.get("max_prompt_tokens", 512))),
                   top_k=int(config.get("chat_quality_top_k", 20)), num_return_sequences=1)
    if report.get("generation_options") != options:
        raise ValueError("Chat quality generation settings do not match the accepted production settings")
    seeds = report.get("seeds")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or seeds != config.get("chat_quality_seeds", [1729, 2718])):
        raise ValueError("Chat quality report needs fixed, unique generation seeds")
    outputs = report.get("cases")
    expected = {(case["id"], seed): case for case in cases for seed in seeds}
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        raise ValueError("Chat quality report is missing full generated outputs")
    seen, scores, categories, hard_failures, passes = set(), [], {}, 0, 0
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("Invalid chat quality output")
        key = (output.get("id"), output.get("seed"))
        if key not in expected or key in seen:
            raise ValueError("Chat quality report contains missing or duplicate cases")
        seen.add(key)
        case = expected[key]
        if (output.get("messages") != case["messages"] or output.get("category") != case["category"]
                or output.get("system_prompt") != case_system_prompt(case, config)):
            raise ValueError("Chat quality report context or role prompt has changed")
        if not isinstance(output.get("reply"), str):
            raise ValueError("Chat quality report must retain the actual reply text")
        judge = output.get("semantic_judge")
        if semantic_enabled:
            expected_system, expected_messages = semantic_judge_prompt(case, output["reply"], config)
            if (not isinstance(judge, dict) or judge.get("system_prompt") != expected_system
                    or judge.get("messages") != expected_messages or not isinstance(judge.get("raw"), str)
                    or judge.get("generation_options") != semantic_judge_options(config)):
                raise ValueError("Semantic judge record does not match the actual chat and reply")
        elif judge is not None:
            raise ValueError("Unexpected semantic judge in quality report")
        rescored = score_reply(case, output["reply"], semantic_judge=output.get("semantic_judge"))
        if output.get("error"):
            rescored["checks"].append(dict(name="generation_error", passed=False, hard=True, detail=output["error"]))
            rescored.update(hard_pass=False, passed=False, score=0.0, hard_failures=rescored["hard_failures"]+1)
        for name in ("checks", "hard_pass", "passed", "score", "hard_failures", "rule_score", "semantic_score", "semantic"):
            if output.get(name) != rescored.get(name):
                raise ValueError(f"Chat quality report score is inconsistent with its reply: {name}")
        score = float(rescored["score"])
        scores.append(score)
        categories.setdefault(case["category"], []).append(score)
        hard_failures += int(rescored["hard_failures"])
        passes += int(rescored["passed"])
    category_scores = {name: sum(values)/len(values) for name, values in categories.items()}
    if (abs(float(report.get("score", -1))-sum(scores)/len(scores)) > 1e-12
            or abs(float(report.get("pass_rate", -1))-passes/len(scores)) > 1e-12
            or report.get("hard_failures") != hard_failures
            or not isinstance(report.get("category_scores"), dict)
            or set(report["category_scores"]) != set(category_scores)
            or any(abs(float(report["category_scores"][name])-value) > 1e-12
                   for name,value in category_scores.items())):
        raise ValueError("Chat quality summary does not match its recorded real outputs")


def build_quality_report(config, generator, planner, *, phase, checkpoint_path, baseline=None):
    """Compare one saved prefix with the unmodified Qwen on the same chats."""
    from .chat_quality import compare_quality, evaluate_chat_quality
    _, cases = _quality_suite(config, phase)
    if baseline is None:
        baseline = evaluate_chat_quality(generator, planner, cases, config, use_prefix=False)
    _validate_generated_quality(baseline, cases, config, use_prefix=False)
    candidate = evaluate_chat_quality(generator, planner, cases, config, use_prefix=True)
    _validate_generated_quality(candidate, cases, config, use_prefix=True)
    if candidate["seeds"] != baseline["seeds"]:
        raise ValueError("Base and prefix quality reports must use identical generation seeds")
    comparison = compare_quality(candidate, baseline, config)
    binding = quality_binding(config, phase, checkpoint_path)
    return dict(format_version=1, phase=phase,
                status="passed" if comparison["passed"] else "rejected",
                selection_criterion=QUALITY_SELECTION_CRITERION, binding=binding,
                checkpoint_sha256=binding["prefix_sha256"], baseline=copy.deepcopy(baseline),
                candidate=candidate, comparison=comparison)


def bind_quality_report(report, config, phase, checkpoint_path):
    """Rebind a selected dev report after copying the same prefix to best.pt."""
    if phase != "dev" or report.get("phase") != "dev":
        raise ValueError("Only development reports can be rebound during checkpoint selection")
    result = copy.deepcopy(report)
    result["binding"] = quality_binding(config, phase, checkpoint_path)
    result["checkpoint_sha256"] = result["binding"]["prefix_sha256"]
    validate_quality_report(result, config, phase, checkpoint_path)
    return result


def validate_quality_report(report, config, phase, checkpoint_path):
    from .chat_quality import compare_quality
    _, cases = _quality_suite(config, phase)
    if report.get("format_version") != 1 or report.get("phase") != phase:
        raise ValueError("Invalid chat quality report phase or version")
    if report.get("selection_criterion") != QUALITY_SELECTION_CRITERION:
        raise ValueError("Qwen prefix was not selected by development chat quality")
    expected = quality_binding(config, phase, checkpoint_path)
    if report.get("binding") != expected or report.get("checkpoint_sha256") != expected["prefix_sha256"]:
        raise ValueError("Stale chat quality report: checkpoint, planner, prompt, suite or settings changed")
    _validate_generated_quality(report.get("baseline"), cases, config, use_prefix=False)
    _validate_generated_quality(report.get("candidate"), cases, config, use_prefix=True)
    if report["baseline"]["seeds"] != report["candidate"]["seeds"]:
        raise ValueError("Base and prefix quality reports use different generation seeds")
    comparison = compare_quality(report["candidate"], report["baseline"], config)
    status = "passed" if comparison["passed"] else "rejected"
    if report.get("comparison") != comparison or report.get("status") != status:
        raise ValueError("Chat quality acceptance does not match its recorded outputs")
    return report


def _manifest_quality_report(config, manifest, phase, *, require_passed=True):
    root = _run_root(config)
    key = "quality_report" if phase == "dev" else "holdout_report"
    name = manifest.get(key)
    if not isinstance(name, str):
        raise ValueError(f"Missing Qwen {phase} chat quality acceptance report")
    path = (root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid Qwen {phase} chat quality report path")
    if manifest.get(key+"_sha256") != file_sha256(path):
        raise ValueError(f"Qwen {phase} chat quality report SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_quality_report(report, config, phase, root/"generator/best.pt")
    if require_passed and report["status"] != "passed":
        raise ValueError(f"Qwen {phase} chat quality rejected: {report['comparison']['reasons']}")
    return report


def require_quality_release(config):
    """Mandatory even when optional serving artifact verification is disabled."""
    stages = require_complete(config, "planner", "generator")
    manifest = stages["generator"]
    if manifest.get("selection_criterion") != QUALITY_SELECTION_CRITERION:
        raise ValueError("Qwen prefix was not selected by development chat quality")
    if manifest.get("quality_status") != "passed":
        raise ValueError("Qwen chat quality acceptance has not passed")
    if manifest.get("checkpoint_sha256") != file_sha256(_run_root(config)/"generator/best.pt"):
        raise ValueError("Selected Qwen prefix checkpoint SHA-256 mismatch")
    return {phase: _manifest_quality_report(config, manifest, phase) for phase in ("dev", "holdout")}


def load_deployment_selection(config):
    """Read explicit selection metadata; its presence never skips validation."""
    path = _run_root(config)/"deployment_selection.json"
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("Deployment selection must be a JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Deployment selection must contain an object")
    return value


def deployment_selection_binding(config):
    """Bind an explicit deployment to its artifacts and serving configuration."""
    root = _run_root(config)
    # Device and dtype are existing execution overrides accepted by the loader.
    configuration = {key:value for key,value in config.items() if key not in ("device", "dtype")}
    return dict(prefix_sha256=file_sha256(root/"generator/best.pt"),
                planner_sha256=file_sha256(root/"planner/best.pt"),
                space_sha256=file_sha256(root/"space.json"),
                configuration_sha256=_json_sha256(configuration))


def _require_technical_quality(report):
    """Known behavioral failures are allowed; broken evaluation is not."""
    for kind in ("baseline", "candidate"):
        evaluation = report[kind]
        for output in evaluation["cases"]:
            if output.get("error") or not output["reply"].strip():
                raise ValueError("User-selected deployment has an incomplete generation evaluation")
            if evaluation["semantic_judge_enabled"]:
                judge = output.get("semantic_judge")
                if not isinstance(judge, dict) or judge.get("error") or output.get("semantic") is None:
                    raise ValueError("User-selected deployment has an invalid semantic judge evaluation")


def _independent_quality_suites(config):
    _, dev_cases = _quality_suite(config, "dev")
    _, holdout_cases = _quality_suite(config, "holdout")
    dev_ids = {case["id"] for case in dev_cases}
    dev_contexts = {_json_sha256(case["messages"]) for case in dev_cases}
    if any(case["id"] in dev_ids or _json_sha256(case["messages"]) in dev_contexts for case in holdout_cases):
        raise ValueError("Final chat holdout must be independent from the checkpoint selection suite")


def _validate_deployment_selection(config, *, require_holdout):
    root = _run_root(config)
    record = load_deployment_selection(config)
    if record is None:
        raise ValueError("Explicit user deployment selection is missing")
    if record.get("format_version") != 1 or record.get("release_mode") != USER_SELECTED_RELEASE:
        raise ValueError("Unsupported deployment selection format or release mode")
    if record.get("status") not in (("ready",) if require_holdout else ("pending_quality", "ready")):
        raise ValueError("User-selected deployment assessment is not complete")
    authorization = record.get("user_selection")
    if (not isinstance(authorization, dict) or authorization.get("authorized") is not True
            or authorization.get("accept_known_quality_failures") is not True
            or authorization.get("source") != "user"
            or any(not isinstance(authorization.get(key), str) or not authorization[key].strip()
                   for key in ("authorization_text", "selection_reason"))):
        raise ValueError("Deployment selection requires explicit user authorization and known-quality-failure acceptance")
    selected = record.get("selected_checkpoint")
    if not isinstance(selected, dict):
        raise ValueError("User-selected checkpoint provenance is missing")
    for key in ("source_run", "source_checkpoint"):
        if not isinstance(selected.get(key), str) or not Path(selected[key]).is_absolute():
            raise ValueError("Selected checkpoint source paths must be absolute")
    source_root = Path(selected["source_run"]).resolve()
    source_checkpoint = Path(selected["source_checkpoint"]).resolve()
    if source_root == root or not source_root.is_relative_to(MODEL_DIR.resolve()):
        raise ValueError("User selection requires a separate deployment run and local training source")
    if source_checkpoint != source_root/"generator/last.pt" or not source_checkpoint.is_file():
        raise ValueError("User-selected deployment must use the fully trained final source checkpoint")
    source_manifest_path = source_root/"generator/manifest.json"
    if (not source_manifest_path.is_file()
            or selected.get("source_checkpoint_sha256") != file_sha256(source_checkpoint)
            or selected.get("source_training_manifest_sha256") != file_sha256(source_manifest_path)):
        raise ValueError("Selected checkpoint source or training manifest SHA-256 mismatch")
    source = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    positive_fields = ("global_step", "train_examples", "epochs_requested", "epochs_complete")
    if any(type(source.get(key)) is not int or source[key] < 1 for key in positive_fields):
        raise ValueError("Selected source is missing completed full-training progress")
    if (source.get("status") != "complete" or source["epochs_complete"] < source["epochs_requested"]
            or source["global_step"] != source["train_examples"]*source["epochs_requested"]
            or type(selected.get("global_step")) is not int or selected["global_step"] != source["global_step"]):
        raise ValueError("Selected source checkpoint has not completed all requested training")
    stages = require_complete(config, "planner", "generator")
    manifest = stages["generator"]
    if (manifest.get("selection_criterion") != USER_SELECTED_RELEASE
            or manifest.get("global_step") != selected["global_step"]
            or any(manifest.get(key) != source[key] for key in ("train_examples", "epochs_requested", "epochs_complete"))):
        raise ValueError("Deployment generator does not match the explicitly selected fully trained checkpoint")
    expected_binding = deployment_selection_binding(config)
    if (record.get("binding") != expected_binding
            or manifest.get("checkpoint_sha256") != expected_binding["prefix_sha256"]):
        raise ValueError("User-selected deployment artifacts or configuration SHA-256 mismatch")
    _independent_quality_suites(config)
    phases = ("dev", "holdout") if require_holdout else ("dev",)
    references = record.get("quality_reports")
    if not isinstance(references, dict):
        raise ValueError("User-selected deployment quality report references are missing")
    quality = {}
    for phase in phases:
        report = _manifest_quality_report(config, manifest, phase, require_passed=False)
        _require_technical_quality(report)
        if phase == "dev" and report.get("global_step") != selected["global_step"]:
            raise ValueError("Development chat evaluation does not match the selected training step")
        key = "quality_report" if phase == "dev" else "holdout_report"
        expected_reference = {"path":manifest[key], "sha256":manifest[key+"_sha256"]}
        if references.get(phase) != expected_reference:
            raise ValueError("User selection quality report SHA-256 references do not match")
        quality[phase] = report
    quality_status = "passed" if all(report["status"] == "passed" for report in quality.values()) else "rejected"
    if record.get("quality_status") != quality_status or manifest.get("quality_status") != quality_status:
        raise ValueError("User-selected deployment must preserve its actual quality status")
    return record, quality


def require_inference_release(config):
    """Retain strict acceptance unless the user explicitly selects a known failure."""
    if load_deployment_selection(config) is None:
        return require_quality_release(config)
    _, quality = _validate_deployment_selection(config, require_holdout=True)
    return quality


def assess_selected_quality(config):
    """Assess a fixed user-selected final prefix once; never relabel failures."""
    existing = load_deployment_selection(config)
    if existing is not None and existing.get("status") == "ready":
        return require_inference_release(config)["holdout"]
    record, _ = _validate_deployment_selection(config, require_holdout=False)
    root = _run_root(config)
    checkpoint = root/"generator/best.pt"
    binding = quality_binding(config, "holdout", checkpoint)
    directory = root/"generator"
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    with (directory/"quality_holdout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt, report_path = directory/"quality_holdout_attempt.json", directory/"quality_holdout.json"
        if attempt.exists():
            previous = json.loads(attempt.read_text(encoding="utf-8"))
            if previous.get("binding") != binding or not report_path.is_file():
                raise ValueError("Selected chat holdout was already attempted; repeated tuning is forbidden")
            if previous.get("status") in ("passed", "rejected") and previous.get("report_sha256") != file_sha256(report_path):
                raise ValueError("The once-only selected holdout report has changed")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_quality_report(report, config, "holdout", checkpoint)
        else:
            write_json(attempt, dict(status="running", binding=binding, started_at=time.time()))
            from .paper_generator import PaperGenerator
            from .paper_planner import load_planner
            generator = PaperGenerator(config, checkpoint=checkpoint)
            planner = load_planner(root, config)
            report = build_quality_report(config, generator, planner, phase="holdout", checkpoint_path=checkpoint)
            write_json(report_path, report)
        write_json(attempt, dict(status=report["status"], binding=binding, finished_at=time.time(),
                                 report_sha256=file_sha256(report_path)))
        dev_status = json.loads((root/manifest["quality_report"]).read_text(encoding="utf-8"))["status"]
        quality_status = "passed" if dev_status == report["status"] == "passed" else "rejected"
        manifest.update(quality_status=quality_status, holdout_report="generator/quality_holdout.json",
                        holdout_report_sha256=file_sha256(report_path), release_mode=USER_SELECTED_RELEASE)
        write_json(directory/"manifest.json", manifest)
        record.update(status="ready", quality_status=quality_status)
        record["quality_reports"]["holdout"] = dict(path=manifest["holdout_report"], sha256=manifest["holdout_report_sha256"])
        write_json(root/"deployment_selection.json", record)
        require_inference_release(config)
        return report


def final_quality(config):
    """Accept the selected prefix once on an independent, untuned holdout."""
    root = _run_root(config)
    stages = require_complete(config, "planner", "generator")
    manifest = stages["generator"]
    if manifest.get("selection_criterion") != QUALITY_SELECTION_CRITERION:
        raise ValueError("Final holdout requires development chat checkpoint selection")
    _manifest_quality_report(config, manifest, "dev")
    _, dev_cases = _quality_suite(config, "dev")
    _, holdout_cases = _quality_suite(config, "holdout")
    dev_ids = {case["id"] for case in dev_cases}
    dev_contexts = {_json_sha256(case["messages"]) for case in dev_cases}
    if any(case["id"] in dev_ids or _json_sha256(case["messages"]) in dev_contexts for case in holdout_cases):
        raise ValueError("Final chat holdout must be independent from the checkpoint selection suite")
    checkpoint = root/"generator/best.pt"
    binding = quality_binding(config, "holdout", checkpoint)
    directory = root/"generator"
    with (directory/"quality_holdout.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = directory/"quality_holdout_attempt.json"
        report_path = directory/"quality_holdout.json"
        if attempt.exists():
            previous = json.loads(attempt.read_text(encoding="utf-8"))
            if previous.get("binding") != binding or not report_path.exists():
                raise ValueError("Final chat holdout was already attempted; checkpoint tuning or repeated evaluation is forbidden")
            if previous.get("status") in ("passed", "rejected") and previous.get("report_sha256") != file_sha256(report_path):
                raise ValueError("The once-only final holdout report was changed after evaluation")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            validate_quality_report(report, config, "holdout", checkpoint)
        else:
            write_json(attempt, dict(status="running", binding=binding, started_at=time.time()))
            from .paper_generator import PaperGenerator
            from .paper_planner import load_planner
            generator = PaperGenerator(config, checkpoint=checkpoint)
            planner = load_planner(root, config)
            report = build_quality_report(config, generator, planner, phase="holdout", checkpoint_path=checkpoint)
            write_json(report_path, report)
        write_json(attempt, dict(status=report["status"], binding=binding, finished_at=time.time(),
                                 report_sha256=file_sha256(report_path)))
        manifest.update(quality_status=report["status"], holdout_report="generator/quality_holdout.json",
                        holdout_report_sha256=file_sha256(report_path), checkpoint_sha256=binding["prefix_sha256"])
        write_json(directory/"manifest.json", manifest)
        if report["status"] != "passed":
            raise ValueError(f"Final Qwen chat quality rejected: {report['comparison']['reasons']}")
        require_quality_release(config)
        return report


def finalize_run(config):
    if is_loss_only_training(config):
        return finalize_loss_only_run(config)
    root = _run_root(config)
    stages = require_complete(config, "planner", "generator")
    quality = require_inference_release(config)
    selection = load_deployment_selection(config)
    evaluation = json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
    index = json.loads((root/"response_index_manifest.json").read_text(encoding="utf-8"))
    graph = json.loads((root/"graph/manifest.json").read_text(encoding="utf-8"))
    if index.get("status") != "complete" or index.get("split") != "train":
        raise ValueError("The training response index must complete before finalizing")
    if not all(split in evaluation for split in ("dev", "test")):
        raise ValueError("Both development and test evaluation must complete before finalizing")
    artifact_paths = ("space.json", "planner/best.pt", "generator/best.pt",
                      "generator/quality_dev.json", "generator/quality_holdout.json",
                      "response_vectors.npz", "response_anchors.jsonl")
    artifact_hashes = {name:file_sha256(root/name) for name in artifact_paths}
    if selection is not None:
        artifact_hashes["deployment_selection.json"] = file_sha256(root/"deployment_selection.json")
    quality_status = "passed" if all(report["comparison"]["passed"] for report in quality.values()) else "rejected"
    manifest = dict(format_version=5,status="complete",quality_status=quality_status,method="EmoStance Qwen3-8B Chinese meme adaptation",
        release_mode=USER_SELECTED_RELEASE if selection is not None else "strict_quality_acceptance",
        config=config,seed=config.get("seed",42),planner=stages["planner"],generator=stages["generator"],
        artifact_sha256=artifact_hashes,
        quality_selection_criterion=USER_SELECTED_RELEASE if selection is not None else QUALITY_SELECTION_CRITERION,
        deployment_selection_file="deployment_selection.json" if selection is not None else None,
        quality_summary={phase:report["comparison"] for phase,report in quality.items()},
        graph=graph,index=index,evaluation_file=str(root/"evaluation.json"),
        evaluation_summary={split:{key:val for key,val in evaluation[split].items() if key != "generator"} for split in ("dev","test")},
        inference=dict(generation="single generation",generations_per_reply=1,reranking="disabled",enable_thinking=False,
            no_repeat_ngram_size=config.get("no_repeat_ngram_size",4),
            meme_selection="nearest training response to generated reply source-distribution @ source-prototypes vector",
            idle_timeout_seconds=config.get("idle_timeout",10)),
        original_emostance_modified=False,base_model_weights_copied=False,
        prior_run_training_checkpoints_reused=bool(config.get("planner_reuse_source") or selection is not None),
        checkpoint_reuse=dict(planner_source=config.get("planner_reuse_source"),
                              planner_reused=bool(config.get("planner_reuse_source")),
                              generator_prefix_reused=selection is not None,
                              generator_source=selection["selected_checkpoint"]["source_run"] if selection is not None else None),
        adaptations=["Chinese character/Latin word TF-IDF instead of English-only regex",
            "user-selected grouped 80/10/10 split; seed42",
            "source-entropy gate from paper, rather than target-entropy gate in released code",
            "mean MSE vector auxiliary loss follows released implementation rather than paper L2 sum notation",
            f"Leiden discovers {graph['num_clusters']} regions for this dataset",
            "Qwen3 native chat template with thinking disabled for training and inference",
            "single reply generation without a candidate pool or reranking",
            "training-response meme retrieval and ten-second conversation reset added for application"])
    for name in ("final_smoke","verification_summary"):
        if (root/f"{name}.json").exists():
            manifest[name] = json.loads((root/f"{name}.json").read_text(encoding="utf-8"))
    write_json(root/"manifest.json",manifest)
    return manifest


def require_loss_only_training(config):
    """Validate training artifacts without granting any inference release."""
    if not is_loss_only_training(config) or config.get("chat_quality_enabled") is not False:
        raise ValueError("This operation requires explicit loss-only training configuration")
    if not isinstance(config.get("version"), str) or not config["version"].strip():
        raise ValueError("Loss-only training requires an explicit new version")
    root = _run_root(config)
    stages = require_complete(config, "planner", "generator")
    generator = stages["generator"]
    if (generator.get("generator_selection_mode") != LOSS_SELECTION_CRITERION
            or generator.get("selection_criterion") != LOSS_SELECTION_CRITERION
            or generator.get("quality_status") != "not_evaluated"
            or generator.get("chat_quality_status") != "not_evaluated"):
        raise ValueError("Completed generator does not describe loss-only checkpoint selection")
    metric = generator.get("best_dev_response_ce")
    if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not 0 <= metric < float("inf"):
        raise ValueError("Loss-only training must retain a finite best development response CE")
    checkpoint_hash = file_sha256(root/"generator/best.pt")
    if generator.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Loss-selected generator checkpoint SHA-256 mismatch")
    subset_path, selection_path = root/"generator/dev_subset.json", root/"generator/loss_selection.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    for name in ("version", "model_version", "data_version"):
        expected = config.get(name)
        if generator.get(name) != expected or selection.get(name) != expected:
            raise ValueError(f"Loss-only training version mismatch: {name}")
    if (selection.get("best_checkpoint_sha256") != checkpoint_hash
            or selection.get("dev_subset_sha256") != file_sha256(subset_path)
            or selection.get("best_dev_response_ce") != metric
            or selection.get("best_step") != generator.get("best_step")
            or selection.get("global_step") != generator.get("global_step")
            or selection.get("control") != "planner_predicted"
            or selection.get("weighting") != "reply_eos_token_weighted"):
        raise ValueError("Loss-only checkpoint selection evidence is stale or inconsistent")
    if not isinstance(subset, dict) or subset.get("split", subset.get("source_split")) != "dev":
        raise ValueError("Loss-only checkpoint selection must use the development split")
    return stages


def evaluate_loss_only(config):
    """Measure dev/test response CE only; never generate or judge a chat reply."""
    import numpy as np
    import torch
    from .paper_generator import PaperGenerator
    from .paper_planner import load_planner
    stages = require_loss_only_training(config)
    root, data = _run_root(config), Path(config["data_dir"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    planner = load_planner(root, config)
    space = planner.space
    evaluation_rows = {}
    subset_size = int(config.get("evaluation_subset_per_split", 512))
    if subset_size < 1:
        raise ValueError("Loss-only evaluation requires a positive fixed subset size")
    rng = np.random.default_rng(config.get("seed", 42))
    for split in ("dev", "test"):
        rows = read_rows(data/f"{split}.jsonl")
        if not rows:
            raise ValueError(f"Loss-only {split} evaluation set is empty")
        selected = sorted(rng.choice(len(rows), size=min(subset_size, len(rows)), replace=False).tolist())
        subset = [rows[index] for index in selected]
        vectors = planner.predict_rows(subset)["target_vector"]
        evaluation_rows[split] = (subset, vectors)
    del planner
    gc.collect()
    torch.cuda.empty_cache()
    generator = PaperGenerator(config, checkpoint=root/"generator/best.pt")
    generator.model.eval()
    result = dict(version=config["version"], model_version=config.get("model_version"), data_version=config.get("data_version"),
                  evaluation_mode="response_ce_only",
                  checkpoint_selection=LOSS_SELECTION_CRITERION, chat_quality_status="not_evaluated",
                  training_objective=config.get("training_objective", "response_ce_only"),
                  checkpoint_sha256=stages["generator"]["checkpoint_sha256"],
                  test_used_for_training=False, test_used_for_checkpoint_selection=False)
    with torch.inference_mode():
        for split, (rows, predicted_vectors) in evaluation_rows.items():
            sums, tokens = {"gold_control":0., "predicted_control":0.}, 0
            for row, predicted in zip(rows, predicted_vectors):
                _, gold = space.targets(row["target_meme_distribution"])
                count = int(generator.response_ids(row["response"]).numel())
                if count < 1:
                    raise ValueError("Response evaluation must include reply and EOS tokens")
                tokens += count
                sums["gold_control"] += float(generator.loss(row, gold))*count
                sums["predicted_control"] += float(generator.loss(row, predicted))*count
            ce = {name:value/tokens for name,value in sums.items()}
            if any(not 0 <= value < float("inf") for value in ce.values()):
                raise ValueError(f"Nonfinite {split} response CE")
            result[split] = dict(examples=len(rows), generator=dict(examples=len(rows), reply_eos_tokens=tokens,
                token_ce=ce, sample_ids=[row["sample_id"] for row in rows],
                selection="seeded fixed subset; test never selects checkpoints", weighting="reply_eos_token_weighted"),
                source_file_sha256=file_sha256(data/f"{split}.jsonl"))
    del generator
    gc.collect()
    torch.cuda.empty_cache()
    write_json(root/"evaluation.json", result)
    finalize_loss_only_run(config)
    return result


def finalize_loss_only_run(config):
    """Finish a training version for review, without authorizing inference."""
    stages = require_loss_only_training(config)
    root = _run_root(config)
    evaluation = json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
    index = json.loads((root/"response_index_manifest.json").read_text(encoding="utf-8"))
    for name in ("version", "model_version", "data_version"):
        if evaluation.get(name) != config.get(name):
            raise ValueError(f"Loss-only evaluation version mismatch: {name}")
    if (evaluation.get("evaluation_mode") != "response_ce_only"
            or evaluation.get("checkpoint_selection") != LOSS_SELECTION_CRITERION
            or evaluation.get("chat_quality_status") != "not_evaluated"
            or evaluation.get("checkpoint_sha256") != stages["generator"]["checkpoint_sha256"]
            or not all(split in evaluation for split in ("dev", "test"))):
        raise ValueError("Both development and test loss evaluation must match the selected checkpoint")
    if index.get("status") != "complete" or index.get("split") != "train":
        raise ValueError("Loss-only training response index must complete before finalizing")
    for key, path in (("train_file_sha256", Path(config["data_dir"])/"train.jsonl"),
                      ("planner_checkpoint_sha256", root/"planner/best.pt"), ("space_sha256", root/"space.json")):
        if index.get(key) != file_sha256(path):
            raise ValueError(f"Loss-only response index provenance mismatch: {key}")
    names = ("space.json", "planner/best.pt", "generator/best.pt", "generator/dev_subset.json",
             "generator/loss_selection.json", "evaluation.json", "response_vectors.npz", "response_anchors.jsonl")
    manifest = dict(format_version=6, version=config["version"], model_version=config.get("model_version"),
        data_version=config.get("data_version"), status="ready_for_review", training_status="complete",
        method="EmoStance frozen Qwen3 prefix; loss-only checkpoint selection", config=config,
        generator_selection_mode=LOSS_SELECTION_CRITERION, selection_criterion=LOSS_SELECTION_CRITERION,
        training_objective=config.get("training_objective", "response_ce_only"),
        quality_status="not_evaluated", chat_quality_status="not_evaluated",
        inference_release_status="not_authorized", inference_enabled=False,
        release_mode="training_artifacts_only", planner=stages["planner"], generator=stages["generator"],
        graph=json.loads((root/"graph/manifest.json").read_text(encoding="utf-8")), index=index,
        artifact_sha256={name:file_sha256(root/name) for name in names},
        evaluation_file=str(root/"evaluation.json"), evaluation_summary={split:evaluation[split] for split in ("dev", "test")},
        prior_run_training_checkpoints_reused=bool(config.get("planner_reuse_source")),
        checkpoint_reuse=dict(planner_source=config.get("planner_reuse_source"), generator_prefix_reused=False),
        active_inference_configuration_modified=False, old_chat_quality_reports_used=False,
        note="Training and loss evaluation are complete. This version requires a separate explicit inference release decision.")
    write_json(root/"manifest.json", manifest)
    return manifest


def prepare_space(config):
    from .paper_space import build_space
    run = _run_root(config)
    run.mkdir(parents=True,exist_ok=True)
    data = Path(config["data_dir"])
    manifest_bytes = (data/"metadata/split_manifest.json").read_bytes()
    execution_keys = {"generator_runtime_gradient_checkpointing","generator_runtime_checkpoint_interval",
        "do_sample","temperature","top_p","max_new_tokens","retrieval_metric","idle_timeout",
        "num_candidates","no_repeat_ngram_size","repetition_penalty","generator_max_steps"}
    method_config = {key:value for key,value in config.items() if key not in execution_keys}
    provenance = dict(split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        split_manifest=str(data/"metadata/split_manifest.json"),config=method_config,
        original_emostance_modified=False)
    existing = run/"provenance.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        previous["config"] = {key:value for key,value in previous["config"].items() if key not in execution_keys}
        if previous != provenance:
            raise ValueError("Run configuration or dataset changed; choose a new run_dir")
    write_json(existing,provenance)
    if (run/"space.json").exists():
        return EmotionSpace.load(run/"space.json")
    space, report = build_space(read_rows(data/"annotation_rows/train.jsonl"),
        read_rows(data/"train.jsonl"),run,config)
    print(json.dumps({"stage":"graph","status":"complete",**report},ensure_ascii=False),flush=True)
    return space


def train(config,stage="all"):
    import torch
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.time()
    if stage not in ("all", "space", "planner", "generator"):
        raise ValueError(f"Unknown training stage: {stage}")
    run = _run_root(config)
    summary = {"stage":stage,"status":"running","stages":{}, "version":config.get("version", run.name),
               "generator_selection_mode":config.get("generator_selection_mode", "chat_quality")}
    write_json(run/"training_status.json", summary)
    try:
        space = prepare_space(config)
        summary["stages"]["space"] = "complete"
        if stage != "space":
            data = Path(config["data_dir"])
            train_rows, dev_rows = read_rows(data/"train.jsonl"),read_rows(data/"dev.jsonl")
            if stage in ("all","planner"):
                from .paper_planner import train_planner
                planner = train_planner(train_rows,dev_rows,space,run,config)
                write_json(run/"planner_result.json",planner)
                summary["stages"]["planner"] = planner.get("status", "incomplete")
                gc.collect()
                torch.cuda.empty_cache()
            if stage in ("all","generator") and all(value == "complete" for value in summary["stages"].values()):
                require_complete(config, "planner")
                from .paper_generator import train_generator
                generator = train_generator(train_rows,space,run,config,dev_rows=dev_rows)
                write_json(run/"generator_result.json",generator)
                summary["stages"]["generator"] = generator.get("status", "incomplete")
                gc.collect()
                torch.cuda.empty_cache()
        summary["status"] = "complete" if all(value == "complete" for value in summary["stages"].values()) else "incomplete"
    except BaseException as error:
        summary["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        summary["error_type"] = type(error).__name__
        summary["elapsed_seconds"] = time.time()-started
        write_json(run/"training_status.json",summary)
        raise
    summary["elapsed_seconds"] = time.time()-started
    write_json(run/"training_status.json",summary)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","quality","index","evaluate","chat","serve","smoke"])
    parser.add_argument("--config",default=None)
    parser.add_argument("--stage",choices=["all","space","planner","generator"],default="all")
    parser.add_argument("--text",default="今天有点累，想休息一下。")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8790)
    args = parser.parse_args()
    config = load_config(args.config) if args.command not in ("chat", "serve") else None
    if args.command == "train":
        result = train(config,args.stage)
        if result["status"] != "complete":
            raise SystemExit(2)
    elif args.command == "quality":
        print(json.dumps(final_quality(config),ensure_ascii=False,indent=2))
    elif args.command == "index":
        from .paper_engine import build_response_index
        build_response_index(config)
    elif args.command == "evaluate":
        if is_loss_only_training(config):
            result = evaluate_loss_only(config)
        else:
            from .paper_engine import evaluate
            result = evaluate(config)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command == "smoke":
        from .paper_generator import smoke_train_step
        if is_loss_only_training(config):
            version = "".join(character if character.isalnum() or character in "._-" else "_" for character in config["version"])
            smoke_run = "smoke_"+version
        else:
            smoke_run = "smoke_qwen3_v2_quality" if config.get("chat_quality_enabled", False) else "smoke_qwen3_v2"
        smoke_config = {**config,"run_dir":str(MODEL_DIR/"runs"/smoke_run)}
        space = prepare_space(smoke_config)
        rows = read_rows(Path(config["data_dir"])/"train.jsonl")
        result = smoke_train_step(rows[0],space,smoke_config["run_dir"],smoke_config)
        print(json.dumps({"stage":"smoke","status":"smoke_complete","full_training_complete":False,
                          "generator":result},ensure_ascii=False))
    else:
        import subprocess
        launcher = MODEL_DIR.parent/"model_information_qwen/run.sh"
        if not launcher.is_file():
            raise RuntimeError("Final inference launcher is missing: model_information_qwen/run.sh")
        command = ["bash",str(launcher),args.command]
        if args.config:
            command += ["--config",str(Path(args.config).resolve())]
        if args.command == "serve":
            command += ["--host",args.host,"--port",str(args.port)]
        elif args.text:
            command += ["--text",args.text]
        raise SystemExit(subprocess.run(command).returncode)


if __name__ == "__main__":
    main()
