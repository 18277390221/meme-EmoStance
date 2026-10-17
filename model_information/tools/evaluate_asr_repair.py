"""Evaluate text repair on frozen, constructed cases; never alter chat config.

These are controlled text examples, not an estimate of a microphone/STT
pipeline's field accuracy. Role-name rules are scored separately from the LLM.
Optional reply comparisons use the same seed and the same loaded model.
Independent LLM repair is temporarily enabled in this process regardless of
the service default; neither chat_config.json nor a running server is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

from model_information.model import PROJECT_ROOT, load_chat_engine
from model_information.normalization import NormalizedInput
from model_information.role_names import canonicalize_role_name


def comparable(text):
    """Ignore whitespace/trailing Chinese periods, retaining numeric symbols."""
    return "".join(c for c in text if not c.isspace()).rstrip("。")


def distance(left, right):
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def expected_texts(case):
    values = case["expected"]
    return [values] if isinstance(values, str) else values


def prompt_seen(case):
    """Mark explicit example sentences/error pairs in the existing ASR prompt."""
    examples = ("我今天不开心，鸭力很大。", "我今天不开心，压力很大。",
                "我不是不想见她，只是太累了。", "帮我去葱一下。",
                "他又那样了，我真烦。", "嗯", "谢谢", "么么哒")
    return comparable(case["input"]) in {comparable(text) for text in examples} or (
        "鸭力" in case["input"] and any("压力" in ref for ref in expected_texts(case)))


def load_cases(path):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for case in cases:
        if case["id"] in ids:
            raise ValueError("Duplicate case id: " + case["id"])
        ids.add(case["id"])
        if case["group"] not in {"correction", "clean", "role_alias", "essential_ambiguity"}:
            raise ValueError("Unknown case group")
        if not expected_texts(case) or any(not isinstance(v, str) or not v for v in expected_texts(case)):
            raise ValueError("Each case needs explicit nonempty reference text")
        if not isinstance(case["expect_clarification"], bool):
            raise ValueError("expect_clarification must be boolean")
        history = case.get("history", [])
        if len(history) % 2 or any(m["role"] != ("user" if i % 2 == 0 else "assistant") for i, m in enumerate(history)):
            raise ValueError("Case history must contain complete user/assistant pairs")
    return cases


def gpu_snapshot():
    executable = Path("/usr/lib/wsl/lib/nvidia-smi")
    if not executable.is_file():
        return None
    result = subprocess.run([str(executable), "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def timing_summary(values):
    ordered = sorted(values)
    return {"mean": round(statistics.mean(values), 4), "median": round(statistics.median(values), 4),
            "p95": round(ordered[min(len(ordered) - 1, int(.95 * len(ordered)))], 4)}


def summarize(rows):
    groups = {}
    for group in sorted({row["group"] for row in rows}):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {"n": len(selected), "success": sum(row["success"] for row in selected),
            "exact_text_match": sum(row["exact_text_match"] for row in selected),
            "content_text_match": sum(row["content_text_match"] for row in selected),
            "unexpected_clarification": sum(row["unexpected_clarification"] for row in selected),
            "status_counts": dict(Counter(row["result"]["status"] for row in selected)),
            "latency_seconds": timing_summary([row["latency_seconds"] for row in selected])}
    edits = [row for row in rows if row["llm_content_edit"]]
    typos = [row for row in rows if row["group"] == "correction"]
    clean = [row for row in rows if row["group"] == "clean"]
    aliases = [row for row in rows if row["group"] == "role_alias"]
    heldout_typos = [row for row in typos if not row["prompt_seen"]]
    ref_length = sum(row["reference_characters"] for row in typos)
    return {"groups": groups,
        "llm_accepted_content_edits": len(edits),
        "llm_correct_accepted_content_edits": sum(row["content_text_match"] for row in edits),
        "clean_content_changed": sum(comparable(row["input"]) != comparable(row["result"]["text"]) for row in clean),
        "rule_only_role_alias_success": sum(row["rule_only_success"] for row in aliases),
        "rule_only_role_alias_n": len(aliases),
        "correction_prompt_seen": {"n": len(typos) - len(heldout_typos),
            "success": sum(row["success"] for row in typos if row["prompt_seen"])},
        "correction_not_prompt_seen": {"n": len(heldout_typos),
            "success": sum(row["success"] for row in heldout_typos)},
        "raw_json_schema_valid": sum(row["raw_schema_valid"] for row in rows),
        "fallback_reasons": dict(Counter(row["result"]["reason"] for row in rows if row["result"]["status"] == "fallback")),
        "correct_raw_suggestions_rejected": sum(row["raw_candidate_matches_reference"] and not row["content_text_match"] for row in typos),
        "typo_character_error_rate": {
            "before": round(sum(row["before_distance"] for row in typos) / ref_length, 6) if ref_length else None,
            "after": round(sum(row["after_distance"] for row in typos) / ref_length, 6) if ref_length else None},
        "latency_seconds": timing_summary([row["latency_seconds"] for row in rows])}


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description="临时启用独立校对的受控文本评估，不修改线上配置")
    parser.add_argument("--cases", type=Path, default=PROJECT_ROOT / "model_information/evaluation/asr_repair_cases.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "model_information/runs/asr_repair_evaluation.json")
    parser.add_argument("--compare-replies", action="store_true", help="对预先选定的14条文本比较原文、仅名字规则、完整校对三条回复")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    # Fix the comparison selection before observing any model output.
    comparison_ids = set()
    if args.compare_replies:
        for group, count in (("correction", 8), ("clean", 4), ("role_alias", 2)):
            comparison_ids.update(case["id"] for case in [c for c in cases if c["group"] == group][:count])
    report = {"status": "running", "dataset": str(args.cases.resolve()),
        "dataset_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "source": "Agent-constructed controlled text examples with explicit reference texts; not independently human-labeled microphone/STT recordings.",
        "scoring": "Explicit reference text; content matching ignores whitespace and trailing Chinese periods, retaining other punctuation including numeric signs/decimal points; it is not a proof of semantic preservation. Clean-input success requires exact text preservation. Exact match is also reported for all groups. For ambiguity, success only scores preservation and clarification triggering, not question relevance; inspect questions separately. Role aliases and explicit prompt-seen examples/error pairs are reported separately from held-out LLM repairs.",
        "asr_prompt_sha256": hashlib.sha256((PROJECT_ROOT / "model_information/prompts/asr_repair_zh.txt").read_bytes()).hexdigest(),
        "comparison_ids": sorted(comparison_ids), "seed": args.seed,
        "gpu_before": gpu_snapshot(), "timing_note": "Evaluation may share the GPU with other processes; no queue, network or Pi timing included. Reply comparisons reuse cached repair results, so their reply_pipeline_seconds exclude the separately recorded repair latency.",
        "cases": [], "reply_comparisons": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    engine = load_chat_engine()
    raw_outputs = []
    original_complete = engine.generator.complete
    original_repair_enabled = engine.normalizer.enabled
    report["repair_mode_override"] = {
        "configured_asr_enabled": engine.settings["asr_enabled"],
        "normalizer_enabled_before": original_repair_enabled,
        "normalizer_enabled_for_evaluation": True,
        "scope": "Independent repair evaluation in this process only; cached reply comparisons do not invoke repair.",
        "chat_config_modified": False,
    }

    def capture(*a, **kw):
        output = original_complete(*a, **kw)
        raw_outputs.append(output)
        return output

    engine.normalizer.enabled = True
    engine.generator.complete = capture
    try:
        for position, case in enumerate(cases, 1):
            history = case.get("history", [])
            messages = history + [{"role": "user", "content": case["input"]}]
            references = expected_texts(case)
            contents = [comparable(value) for value in references]
            before = time.perf_counter()
            canonical = canonicalize_role_name(case["input"], history)
            rule_seconds = time.perf_counter() - before
            raw_outputs.clear()
            sync()
            before = time.perf_counter()
            result = engine.normalizer.normalize(messages)
            sync()
            elapsed = time.perf_counter() - before
            actual = comparable(result.text)
            text_match = actual in contents
            clarification = result.status == "uncertain"
            raw = raw_outputs[-1] if raw_outputs else ""
            try:
                payload = json.loads(raw)
                valid = isinstance(payload, dict) and set(payload) == {"status", "text", "clarification"} and payload["status"] in {"unchanged", "corrected", "uncertain"} and isinstance(payload["text"], str) and bool(payload["text"].strip()) and isinstance(payload["clarification"], str)
            except (ValueError, TypeError):
                payload, valid = {}, False
            row = {**case, "prompt_seen": prompt_seen(case), "rule_only_text": canonical, "rule_only_seconds": round(rule_seconds, 6),
                "rule_only_success": comparable(canonical) in contents and not case["expect_clarification"],
                "result": asdict(result), "raw_output": raw, "raw_schema_valid": valid,
                "raw_candidate_matches_reference": valid and comparable(payload["text"]) in contents,
                "latency_seconds": round(elapsed, 6), "exact_text_match": result.text in references,
                "content_text_match": text_match, "unexpected_clarification": clarification and not case["expect_clarification"],
                "success": (result.text in references if case["group"] == "clean" else text_match) and clarification == case["expect_clarification"],
                "llm_content_edit": actual != comparable(canonical),
                "reference_characters": len(contents[0]),
                "before_distance": min(distance(comparable(case["input"]), ref) for ref in contents),
                "after_distance": min(distance(actual, ref) for ref in contents)}
            report["cases"].append(row)
            if position % 5 == 0 or position == len(cases):
                save()
                print(json.dumps({"done": position, "total": len(cases), "statuses": dict(Counter(r["result"]["status"] for r in report["cases"]))}, ensure_ascii=False), flush=True)
    finally:
        engine.generator.complete = original_complete
        engine.normalizer.enabled = original_repair_enabled

    report["summary"] = summarize(report["cases"])
    save()
    normalize = engine.normalizer.normalize
    try:
        for case in report["cases"]:
            if case["id"] not in comparison_ids:
                continue
            variants = {"raw": NormalizedInput(case["input"]),
                        "role_rules_only": NormalizedInput(case["rule_only_text"]),
                        "full_repair": NormalizedInput(**case["result"])}
            comparison = {"id": case["id"], "input": case["input"], "expected": case["expected"], "variants": {}}
            for variant, normalized in variants.items():
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
                engine.normalizer.normalize = lambda messages, value=normalized: value
                sync()
                started = time.perf_counter()
                turn = engine.reply_turn(case.get("history", []) + [{"role": "user", "content": case["input"]}])
                sync()
                comparison["variants"][variant] = {"effective_input": turn.user_text, **turn.as_reply(),
                    "reply_pipeline_seconds": round(time.perf_counter() - started, 6)}
            report["reply_comparisons"].append(comparison)
            save()
            print(json.dumps({"reply_comparison_done": case["id"]}, ensure_ascii=False), flush=True)
    finally:
        engine.normalizer.normalize = normalize
    report["status"] = "complete"
    report["gpu_after"] = gpu_snapshot()
    save()
    print(json.dumps({"report": str(args.output.resolve()), "summary": report["summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
