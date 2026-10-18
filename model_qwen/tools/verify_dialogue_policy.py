"""Replay real dialogue with an explicit system policy and the selected prefix.

This is an independent diagnostic, not a holdout run or a model release. It
never edits the original configuration, weights, quality reports or server.
Each case/seed makes exactly one completion, without candidates or retries.
Policy JSON supports mode=append or mode=replace; plain text appends by default.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parent))


def load_cases(path):
    """Accept a JSON array, a suite object with cases, or JSONL case records."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        suite = {"cases": [json.loads(line) for line in text.splitlines() if line.strip()]}
    else:
        payload = json.loads(text)
        suite = {"cases": payload} if isinstance(payload, list) else payload
    if not isinstance(suite, dict) or not isinstance(suite.get("cases"), list) or not suite["cases"]:
        raise ValueError("Cases must be a nonempty JSON array, JSONL, or suite object with cases")
    seen = set()
    for case in suite["cases"]:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
            raise ValueError("Every case requires a nonempty string id")
        if case["id"] in seen:
            raise ValueError(f"Duplicate case id: {case['id']}")
        seen.add(case["id"])
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
            raise ValueError(f"Case {case['id']} needs a user/assistant history ending with user")
        for index, message in enumerate(messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if (not isinstance(message, dict) or message.get("role") != expected
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                raise ValueError(f"Case {case['id']} requires alternating user/assistant messages with text")
    return suite


def _seed(value):
    seed = int(value)
    if not 0 <= seed < 2**32:
        raise argparse.ArgumentTypeError("Seeds must be in [0, 2**32)")
    return seed


def generation_options(config):
    """Keep all sampling settings from the original evaluated deployment."""
    return {"max_new_tokens": config.get("chat_quality_max_new_tokens", config.get("max_new_tokens", 128)),
            "do_sample": config.get("do_sample", True),
            "temperature": config.get("chat_quality_temperature", config.get("temperature", .7)),
            "top_p": config.get("chat_quality_top_p", config.get("top_p", .8)),
            "top_k": config.get("chat_quality_top_k", config.get("top_k", 20)),
            "no_repeat_ngram_size": config.get("chat_quality_no_repeat_ngram_size", config.get("no_repeat_ngram_size", 4)),
            "max_prompt_tokens": config.get("chat_quality_max_prompt_tokens", config.get("max_prompt_tokens", 4096))}


def holdout_attempt(output, binding, seeds, file_sha256):
    """Reserve a frozen holdout once, or verify its existing complete report."""
    attempt = Path(str(output) + ".attempt.json")
    if attempt.exists():
        record = json.loads(attempt.read_text(encoding="utf-8"))
        if (record.get("status") != "complete" or record.get("binding") != binding
                or record.get("seeds") != seeds or not output.is_file()
                or record.get("report_sha256") != file_sha256(output)):
            raise ValueError("Holdout was already attempted; incomplete, changed or stale evidence cannot be regenerated")
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") != "complete" or report.get("binding") != binding or report.get("seeds") != seeds:
            raise ValueError("The existing holdout report does not match the frozen evaluation")
        return attempt, report
    if output.exists():
        raise FileExistsError("Holdout output exists without a matching once-only attempt record")
    attempt.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(attempt, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"status": "running", "binding": binding, "seeds": seeds,
                   "started_at": time.time()}, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return attempt, None


def evaluate(generator, planner, suite, seeds, system_prompt, options, binding, save,
             seed_all):
    """Collect each actual output, preserving failed attempts without retrying."""
    report = {"format_version": 1, "kind": "qwen_dialogue_policy_diagnostic", "status": "running",
              "binding": binding, "seeds": list(seeds), "cases": [],
              "generation_options": {**options, "enable_thinking": False,
                                     "num_beams": 1, "num_return_sequences": 1},
              "suite_metadata": {key: value for key, value in suite.items() if key != "cases"},
              "weights_modified": False, "config_modified": False, "server_state_modified": False,
              "original_quality_reports_modified": False, "original_holdout_rerun": False,
              "regression": {"status": "unreviewed", "scope": "dialogue_repetition_regression_only"}}
    save(report)
    for case in suite["cases"]:
        messages = copy.deepcopy(case["messages"])
        history = [{"role": "A" if message["role"] == "user" else "B", "text": message["content"]}
                   for message in messages]
        vector = planner.predict(history, "B")["target_vector"]
        for seed in seeds:
            entry = {"id": case["id"], "seed": seed, "messages": copy.deepcopy(messages),
                     "system_prompt": system_prompt, "binding": binding,
                     "generate_calls": 0, "reply": "", "elapsed_seconds": 0.0}
            if "category" in case:
                entry["category"] = case["category"]
            started = time.monotonic()
            original_generate = generator.model.generate
            def counted_generate(*args, **kwargs):
                entry["generate_calls"] += 1
                return original_generate(*args, **kwargs)
            try:
                seed_all(seed)
                # Keep the entire supplied history. A diagnostic cannot claim
                # to test a context that was silently trimmed by the tokenizer.
                budget = min(int(options["max_prompt_tokens"]),
                    generator.model.config.max_position_embeddings - options["max_new_tokens"]
                    - generator.projector.prefix_length)
                encoded = generator.tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, *messages],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False,
                    return_dict=True, return_tensors="pt")
                ids = encoded["input_ids"]
                if ids.shape[1] > budget:
                    raise ValueError(f"Full policy and history require {ids.shape[1]} tokens, exceeding {budget}; no history was trimmed")
                entry["prompt_tokens"] = int(ids.shape[1])
                entry["prompt"] = generator.tokenizer.decode(ids[0], skip_special_tokens=False)
                entry["full_history_retained"] = True
                generator.model.generate = counted_generate
                reply = generator.complete(system_prompt, messages, stance_vector=vector, **options)
                if entry["generate_calls"] != 1:
                    raise RuntimeError("Each diagnostic output requires exactly one actual model.generate call")
                if not isinstance(reply, str) or not reply.strip():
                    raise ValueError("Qwen returned an empty or nontext completion")
                entry["reply"] = reply
            except Exception as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
            finally:
                generator.model.generate = original_generate
            entry["elapsed_seconds"] = round(time.monotonic() - started, 6)
            report["cases"].append(entry)
            save(report)
            print(json.dumps({key: entry[key] for key in ("id", "seed", "reply", "generate_calls", "elapsed_seconds")}
                             | ({"error": entry["error"]} if "error" in entry else {}),
                             ensure_ascii=False), flush=True)
    report["status"] = "failed" if any(entry.get("error") for entry in report["cases"]) else "complete"
    save(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "inference_config.json")
    parser.add_argument("--seeds", type=_seed, nargs="+", default=[1729, 2718])
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must not contain duplicates")
    output = args.output.expanduser().resolve()
    if output in {path.expanduser().resolve() for path in (args.policy_file, args.cases, args.config)}:
        parser.error("--output must differ from the policy, cases and original configuration")
    suite = load_cases(args.cases)
    from emostance_qwen.paper_pipeline import (
        deployment_selection_binding, file_sha256, load_config, require_inference_release, write_json)
    from model_information_qwen.dialogue_policy import load_policy, effective_system_prompt, policy_binding
    config = load_config(args.config)
    require_inference_release(config)
    policy = load_policy(args.policy_file)
    system_prompt = effective_system_prompt(config, policy)
    binding = policy_binding(config, config_path=args.config,
                             policy_path=args.policy_file, cases_path=args.cases)
    attempt = None
    if suite.get("split") in ("holdout", "diagnostic_holdout"):
        attempt, existing = holdout_attempt(output, binding, args.seeds, file_sha256)
        if existing is not None:
            print(json.dumps({"status": "reused_complete_holdout", "output": str(output)}, ensure_ascii=False))
            return 0
    elif output.exists():
        parser.error("--output already exists; use a new path to preserve earlier evidence")
    origin = deployment_selection_binding(config)
    run = Path(config["run_dir"])
    checkpoint = run / "generator/best.pt"
    original_status = json.loads((run / "generator/manifest.json").read_text(encoding="utf-8")).get("quality_status")
    identity = dict(model_dir=str(run), original_quality_status=original_status,
        original_prefix_sha256=origin["prefix_sha256"],
        original_config_binding_sha256=origin["configuration_sha256"],
        policy_file_sha256=file_sha256(args.policy_file), cases_file_sha256=file_sha256(args.cases),
        policy_sha256=hashlib.sha256(policy["text"].encode("utf-8")).hexdigest(),
        policy_mode=policy["mode"],
        effective_system_prompt_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        config_path=str(args.config.expanduser().resolve()),
        policy_file=str(args.policy_file.expanduser().resolve()),
        cases_file=str(args.cases.expanduser().resolve()))

    def save(report):
        report.update(identity)
        for entry in report["cases"]:
            for key in ("original_prefix_sha256", "original_config_binding_sha256", "policy_sha256",
                        "policy_mode", "effective_system_prompt_sha256"):
                entry[key] = identity[key]
        write_json(output, report)

    try:
        import numpy as np
        import torch
        from emostance_qwen.paper_generator import PaperGenerator
        from emostance_qwen.paper_planner import load_planner
        torch.set_num_threads(8)
        generator = PaperGenerator(config, checkpoint=checkpoint)
        planner = load_planner(run, config)

        def seed_all(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        report = evaluate(generator, planner, suite, args.seeds, system_prompt,
                          generation_options(config), binding, save, seed_all)
        if policy_binding(config, config_path=args.config, policy_path=args.policy_file, cases_path=args.cases) != binding:
            report["status"] = "failed"
            report["binding_changed_during_evaluation"] = True
            save(report)
        if attempt is not None:
            write_json(attempt, {"status": report["status"], "binding": binding, "seeds": args.seeds,
                                "finished_at": time.time(), "report_sha256": file_sha256(output)})
    except BaseException as exc:
        if attempt is not None:
            write_json(attempt, {"status": "failed", "binding": binding, "seeds": args.seeds,
                                "finished_at": time.time(), "error_type": type(exc).__name__})
        raise
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
