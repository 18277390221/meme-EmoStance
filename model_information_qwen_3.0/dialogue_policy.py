"""Bind optional prompt policy to new dialogue regressions, not training quality.

The original model release must still validate first. A separate development
and once-only holdout review cover only the new dialogue policy; their success
never changes the original model's quality status.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APPLICATION_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = APPLICATION_ROOT/"dialogue_policy.json"
DEFAULT_RELEASE_PATH = APPLICATION_ROOT/"dialogue_policy_release.json"
RELEASE_SCOPE = "dialogue_repetition_regression_only"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_sha256(value):
    return _text_sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False))


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_policy(path):
    """Read diagnostic text or JSON policy without activating it."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text
    if isinstance(value, str):
        value = {"policy":value}
    if not isinstance(value, dict) or value.get("format_version", 1) != 1:
        raise ValueError("Unsupported dialogue policy format")
    enabled, mode = value.get("enabled", True), value.get("mode", "append")
    if type(enabled) is not bool or mode not in ("append", "replace"):
        raise ValueError("Dialogue policy needs enabled boolean and append/replace mode")
    policy = value.get("policy", value.get("policy_text", value.get("text")))
    if not isinstance(policy, str) or (enabled and not policy.strip()):
        raise ValueError("Enabled dialogue policy must contain nonempty text")
    return dict(format_version=1, enabled=enabled, mode=mode, text=policy.strip())


def effective_system_prompt(config, policy):
    original = config["system_prompt"]
    if not isinstance(original, str) or not original.strip():
        raise ValueError("Dialogue policy needs the original model system prompt")
    if not policy["enabled"]:
        return original
    return policy["text"] if policy["mode"] == "replace" else original+"\n\n"+policy["text"]


def policy_binding(config, config_path, policy_path, cases_path):
    """Hash the unchanged model, original release and exact new prompt inputs."""
    root = Path(config["run_dir"]).resolve()
    configuration = {key:value for key,value in config.items() if key not in ("device", "dtype")}
    policy = load_policy(policy_path)
    selection = root/"deployment_selection.json"
    return dict(prefix_sha256=_sha256(root/"generator/best.pt"),
                planner_sha256=_sha256(root/"planner/best.pt"), space_sha256=_sha256(root/"space.json"),
                configuration_sha256=_json_sha256(configuration), base_config_file_sha256=_sha256(config_path),
                base_system_prompt_sha256=_text_sha256(config["system_prompt"]),
                base_manifest_sha256=_sha256(root/"manifest.json"),
                base_generator_manifest_sha256=_sha256(root/"generator/manifest.json"),
                base_dev_report_sha256=_sha256(root/"generator/quality_dev.json"),
                base_holdout_report_sha256=_sha256(root/"generator/quality_holdout.json"),
                base_selection_sha256=_sha256(selection) if selection.is_file() else None,
                policy_file_sha256=_sha256(policy_path), cases_file_sha256=_sha256(cases_path),
                effective_system_prompt_sha256=_text_sha256(effective_system_prompt(config, policy)))


def _artifact_path(path):
    value = Path(path).expanduser()
    value = (value if value.is_absolute() else PROJECT_ROOT/value).resolve()
    if not value.is_relative_to(PROJECT_ROOT.resolve()) or not value.is_file():
        raise ValueError("Dialogue policy evidence must be an existing project artifact")
    return value


def _reference(path):
    return str(_artifact_path(path).relative_to(PROJECT_ROOT.resolve()))


def _load_suite(path, phase):
    suite = _read_json(path)
    if not isinstance(suite, dict) or suite.get("split") not in (phase, "diagnostic_"+phase):
        raise ValueError("Dialogue regression suite must declare its dev/holdout split")
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dialogue regression suite needs complete cases")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("Dialogue regression case IDs must be nonempty and unique")
    for case in cases:
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or len(messages)%2 != 1:
            raise ValueError("Dialogue regression cases need complete alternating history")
        for index, message in enumerate(messages):
            if (not isinstance(message, dict) or message.get("role") != ("user" if index%2 == 0 else "assistant")
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise ValueError("Dialogue regression history must preserve the latest user turn")
    return suite, cases


def policy_generation_options(config):
    return dict(max_new_tokens=config.get("chat_quality_max_new_tokens", config.get("max_new_tokens", 128)),
                do_sample=config.get("do_sample", True),
                temperature=config.get("chat_quality_temperature", config.get("temperature", .7)),
                top_p=config.get("chat_quality_top_p", config.get("top_p", .8)),
                top_k=config.get("chat_quality_top_k", config.get("top_k", 20)),
                no_repeat_ngram_size=config.get("chat_quality_no_repeat_ngram_size", config.get("no_repeat_ngram_size", 4)),
                max_prompt_tokens=config.get("chat_quality_max_prompt_tokens", config.get("max_prompt_tokens", 4096)),
                enable_thinking=False, num_beams=1, num_return_sequences=1)


def _validate_evidence(config, config_path, policy_path, cases_path, report_path, phase):
    _, cases = _load_suite(cases_path, phase)
    report = _read_json(report_path)
    expected_binding = policy_binding(config, config_path, policy_path, cases_path)
    if (not isinstance(report, dict) or report.get("format_version") != 1
            or report.get("status") != "complete" or report.get("binding") != expected_binding):
        raise ValueError("Dialogue policy regression report is incomplete or has stale bindings")
    if report.get("generation_options") != policy_generation_options(config):
        raise ValueError("Dialogue policy regression used different production generation settings")
    regression = report.get("regression")
    if not isinstance(regression, dict) or regression.get("status") not in ("unreviewed", "passed"):
        raise ValueError("Dialogue policy regression report records a failed or missing assessment")
    seeds = report.get("seeds")
    if (not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("Dialogue policy regression needs at least two fixed generation seeds")
    outputs = report.get("cases")
    expected = {(case["id"], seed):case for case in cases for seed in seeds}
    if not isinstance(outputs, list) or len(outputs) != len(expected):
        raise ValueError("Dialogue policy regression is missing full case/seed outputs")
    prompt = effective_system_prompt(config, load_policy(policy_path))
    seen = set()
    for output in outputs:
        if not isinstance(output, dict):
            raise ValueError("Invalid dialogue policy regression output")
        key = (output.get("id"), output.get("seed"))
        if key not in expected or key in seen:
            raise ValueError("Dialogue policy regression has missing or duplicate case/seed outputs")
        seen.add(key)
        current = expected[key]
        recorded_prompt = output.get("system_prompt", output.get("effective_system_prompt"))
        calls = output.get("generate_calls", output.get("generation_calls"))
        if (output.get("messages") != current["messages"] or recorded_prompt != prompt
                or type(calls) is not int or calls != 1 or output.get("error")
                or not isinstance(output.get("reply"), str) or not output["reply"].strip()
                or output.get("binding") != expected_binding or output.get("full_history_retained") is not True
                or type(output.get("prompt_tokens")) is not int or output["prompt_tokens"] < 1
                or not isinstance(output.get("prompt"), str) or not output["prompt"].strip()):
            raise ValueError("Dialogue policy evidence must retain real history, new prompt and one successful generation")
        position = output["prompt"].find(prompt)
        if position < 0:
            raise ValueError("Recorded tokenized dialogue prompt omitted the effective system instructions")
        position += len(prompt)
        for message in current["messages"]:
            index = output["prompt"].find(message["content"], position)
            if index < 0:
                raise ValueError("Recorded tokenized dialogue prompt omitted or reordered real history")
            position = index+len(message["content"])
    return report, cases, expected_binding


def _common_binding(binding):
    return {key:value for key,value in binding.items() if key != "cases_file_sha256"}


def build_policy_release(config, config_path, policy_path, validation_reports):
    """Prepare a reviewable release record; caller writes/activates it later."""
    if not isinstance(validation_reports, list) or len(validation_reports) != 2:
        raise ValueError("Dialogue policy release needs development and independent holdout reports")
    entries, phases, suite_ids, contexts, common = [], set(), set(), set(), None
    for item in validation_reports:
        phase = item.get("phase")
        if phase not in ("dev", "holdout") or phase in phases:
            raise ValueError("Dialogue policy release needs one dev and one holdout review")
        phases.add(phase)
        cases_path, report_path = _artifact_path(item["cases_path"]), _artifact_path(item["report_path"])
        report, cases, binding = _validate_evidence(config, config_path, policy_path, cases_path, report_path, phase)
        review = item.get("review")
        if (not isinstance(review, dict) or review.get("status") != "passed"
                or any(not isinstance(review.get(key), str) or not review[key].strip() for key in ("reviewer", "notes"))):
            raise ValueError("Dialogue policy release requires recorded semantic review of every real output")
        ids = {case["id"] for case in cases}
        current_contexts = {_json_sha256(case["messages"]) for case in cases}
        if suite_ids & ids or contexts & current_contexts:
            raise ValueError("Dialogue policy holdout must be independent from its development suite")
        suite_ids.update(ids)
        contexts.update(current_contexts)
        current_common = _common_binding(binding)
        if common is not None and common != current_common:
            raise ValueError("Development and holdout must evaluate the same frozen dialogue policy")
        common = current_common
        reviewed = {**copy.deepcopy(review), "reviewed_case_ids":sorted(ids), "scope":RELEASE_SCOPE}
        entry = dict(phase=phase, cases_path=_reference(cases_path), report_path=_reference(report_path),
                     report_sha256=_sha256(report_path), binding=binding, review=reviewed)
        if phase == "holdout":
            attempt_path = _artifact_path(str(report_path)+".attempt.json")
            attempt = _read_json(attempt_path)
            if (not isinstance(attempt, dict) or attempt.get("status") != "complete"
                    or attempt.get("binding") != binding or attempt.get("seeds") != report["seeds"]
                    or attempt.get("report_sha256") != entry["report_sha256"]):
                raise ValueError("Dialogue policy holdout must have one completed frozen assessment")
            entry.update(attempt_path=_reference(attempt_path), attempt_sha256=_sha256(attempt_path))
        entries.append(entry)
    quality = _read_json(Path(config["run_dir"])/"generator/manifest.json").get("quality_status")
    return dict(format_version=1, scope=RELEASE_SCOPE, status="passed", binding=common,
                version="dialogue-policy-"+common["policy_file_sha256"][:12],
                base_quality_status=quality, validation_reports=entries,
                limitations="This release covers the recorded dialogue regressions only; original model quality status is unchanged.")


def resolve_policy(config, config_path, *, policy_path=None, release_path=None):
    """Return disabled state or verify a policy against its signed evidence."""
    selected = Path(policy_path or DEFAULT_POLICY_PATH).resolve()
    original = config.get("system_prompt", "")
    if not selected.exists():
        if policy_path is not None:
            raise FileNotFoundError(f"Dialogue policy file is missing: {selected}")
        return dict(enabled=False, status="disabled", system_prompt=original, version=None,
                    policy_path=None, scope=RELEASE_SCOPE)
    policy = load_policy(selected)
    if not policy["enabled"]:
        return dict(enabled=False, status="disabled", system_prompt=original, version=None,
                    policy_path=str(selected), policy_file_sha256=_sha256(selected), scope=RELEASE_SCOPE)
    release_file = Path(release_path or DEFAULT_RELEASE_PATH).resolve()
    if not release_file.is_file():
        raise ValueError("Enabled dialogue policy has no reviewed release report")
    release = _read_json(release_file)
    if (not isinstance(release, dict) or release.get("format_version") != 1
            or release.get("scope") != RELEASE_SCOPE or release.get("status") != "passed"):
        raise ValueError("Dialogue policy release has not passed its own regression review")
    validated = build_policy_release(config, config_path, selected, release.get("validation_reports"))
    if any(release.get(key) != validated[key] for key in ("binding", "version", "base_quality_status", "validation_reports")):
        raise ValueError("Dialogue policy release is stale or its reviewed report has changed")
    return dict(enabled=True, status="regression_passed", system_prompt=effective_system_prompt(config, policy),
                version=release["version"], mode=policy["mode"], scope=RELEASE_SCOPE,
                policy_path=str(selected), policy_file_sha256=_sha256(selected), release_path=str(release_file),
                release_report_sha256=_sha256(release_file), base_quality_status=release["base_quality_status"])
