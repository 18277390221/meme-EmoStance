#!/usr/bin/env python3
"""Generate verified run diagnostics and export a completed model version.

Training, indexing, evaluation, runtime verification, and unit tests must already
have succeeded. Tokenizer diagnostics read only the configured training split;
this tool never runs training or loads a language model onto the GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sys


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / "tools"))
sys.path.insert(0, str(MODEL_ROOT / "src"))

from export_model_version import digest, export
from summarize_paper_run import build_report, write_report


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def parse_unit_test_log(path):
    """Use the final unittest run, rejecting incomplete or failed trailing runs."""
    path = Path(path).resolve()
    log = path.read_text(encoding="utf-8")
    summaries = list(re.finditer(
        r"^Ran (\d+) tests? in ([0-9]+(?:\.[0-9]+)?)\s*s\s*$", log, re.MULTILINE))
    if not summaries:
        raise ValueError(f"No unittest run summary found: {path}")
    summary = summaries[-1]
    tail = [line.strip() for line in log[summary.end():].splitlines() if line.strip()]
    if len(tail) != 1 or not re.fullmatch(r"OK(?: \([^\r\n]*\))?", tail[0]):
        raise ValueError(f"The final unittest run did not finish with OK: {path}")
    count, seconds = int(summary[1]), float(summary[2])
    if count < 1 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid unittest count or duration: {path}")
    return dict(unit_tests=count, unit_test_seconds=seconds, unit_test_log=str(path),
                unit_test_log_sha256=digest(path), unit_test_result=tail[0])


def _statistics(total_tokens, unk_tokens, samples_with_unk, lengths):
    count = len(lengths)
    return dict(total_tokens=total_tokens, unk_tokens=unk_tokens,
                unk_token_ratio=unk_tokens / total_tokens,
                unk_token_percent=100 * unk_tokens / total_tokens,
                samples_containing_unk=samples_with_unk,
                samples_containing_unk_ratio=samples_with_unk / count,
                samples_containing_unk_percent=100 * samples_with_unk / count,
                mean_tokens=total_tokens / count, max_tokens=max(lengths), min_tokens=min(lengths))


def tokenizer_diagnostics(config, *, tokenizer=None, serializer=None, batch_size=256):
    """Measure exact planner windows in batches without reading held-out rows."""
    if batch_size < 1:
        raise ValueError("Tokenizer batch size must be positive")
    run = Path(config["run_dir"]).resolve()
    training_file = Path(config["data_dir"]).resolve() / "train.jsonl"
    tokenizer_dir = run / "planner/tokenizer"
    tokenizer_files = sorted(path for path in tokenizer_dir.rglob("*") if path.is_file())
    if not tokenizer_files:
        raise ValueError(f"Saved planner tokenizer is missing: {tokenizer_dir}")
    file_hashes = {str(path.relative_to(tokenizer_dir)): digest(path) for path in tokenizer_files}
    source_hash = digest(training_file)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True,
                                                 trust_remote_code=False, use_fast=True)
    if serializer is None:
        from emostance_meme.paper_planner import serialize_row
        serializer = serialize_row
    tokenizer.truncation_side = "left"
    max_length = int(config.get("planner_max_length", 320))
    if max_length < 2:
        raise ValueError("Planner maximum token length must be at least two")
    unk_id = tokenizer.unk_token_id
    lengths, full_lengths = [], []
    tokens = full_tokens = unknown = full_unknown = unknown_samples = full_unknown_samples = 0
    special_tokens = truncated_samples = tokens_removed = 0

    def consume(texts):
        nonlocal tokens, full_tokens, unknown, full_unknown, unknown_samples, full_unknown_samples
        nonlocal special_tokens, truncated_samples, tokens_removed
        options = dict(add_special_tokens=True, padding=False, return_token_type_ids=False)
        window = tokenizer(texts, **options, truncation=True, max_length=max_length,
                           return_special_tokens_mask=True)
        complete = tokenizer(texts, **options, truncation=False)
        if len(window["input_ids"]) != len(texts) or len(complete["input_ids"]) != len(texts):
            raise ValueError("Tokenizer returned a different batch size")
        for ids, full_ids, mask in zip(window["input_ids"], complete["input_ids"],
                                       window["special_tokens_mask"]):
            size, full_size = len(ids), len(full_ids)
            if not 0 < size <= max_length or full_size < size or len(mask) != size:
                raise ValueError("Tokenizer returned invalid training window lengths")
            nunk = ids.count(unk_id) if unk_id is not None else 0
            full_nunk = full_ids.count(unk_id) if unk_id is not None else 0
            lengths.append(size)
            full_lengths.append(full_size)
            tokens += size
            full_tokens += full_size
            unknown += nunk
            full_unknown += full_nunk
            unknown_samples += int(nunk > 0)
            full_unknown_samples += int(full_nunk > 0)
            special_tokens += sum(mask)
            truncated_samples += int(full_size > size)
            tokens_removed += full_size - size

    batch = []
    with training_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("split", row.get("source_split")) != "train":
                raise ValueError(f"Diagnostics require train-only samples: {training_file}:{line_number}")
            batch.append(serializer(row))
            if len(batch) == batch_size:
                consume(batch)
                batch = []
    if batch:
        consume(batch)
    if not lengths:
        raise ValueError(f"Training split is empty: {training_file}")
    if digest(training_file) != source_hash or any(digest(path) != file_hashes[str(path.relative_to(tokenizer_dir))]
                                                   for path in tokenizer_files):
        raise ValueError("Training data or saved tokenizer changed during diagnostics")
    window = _statistics(tokens, unknown, unknown_samples, lengths)
    non_special = tokens - special_tokens
    window.update(inserted_special_tokens=special_tokens, non_special_tokens=non_special,
                  unk_ratio_excluding_inserted_special_tokens=unknown / non_special if non_special else 0.0)
    return dict(format_version=1,
        scope="Only configured training adjacent-turn samples; full repeated histories are counted per sample exactly as supplied to the planner.",
        source=dict(path=str(training_file), sha256=source_hash, sample_count=len(lengths)),
        tokenizer=dict(path=str(tokenizer_dir), files_sha256=file_hashes,
            tokenizer_json_sha256=file_hashes.get("tokenizer.json"),
            is_fast=bool(tokenizer.is_fast), **{"class": tokenizer.__class__.__name__},
            unk_token=tokenizer.unk_token, unk_token_id=unk_id, vocab_size=tokenizer.vocab_size,
            truncation_side="left", max_length=max_length, padding=False,
            special_tokens_included_in_primary_token_denominator=True),
        training_window=window,
        before_truncation=_statistics(full_tokens, full_unknown, full_unknown_samples, full_lengths),
        truncation=dict(samples_truncated=truncated_samples,
            samples_truncated_ratio=truncated_samples / len(lengths),
            samples_truncated_percent=100 * truncated_samples / len(lengths), tokens_removed=tokens_removed),
        interpretation="UNK replaces out-of-vocabulary spans and can erase lexical distinctions. Coverage and truncation statistics do not measure reply quality or imply accuracy.")


def completed_run_evidence(config, unit_test_log):
    run = Path(config["run_dir"]).resolve()
    tests = parse_unit_test_log(unit_test_log)
    manifest = read_json(run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("Training, indexing, and evaluation must be complete before exporting")
    for key in ("run_dir", "data_dir"):
        if Path(manifest["config"][key]).resolve() != Path(config[key]).resolve():
            raise ValueError(f"Completed run configuration differs in {key}")
    for name in ("planner", "generator"):
        if read_json(run / name / "manifest.json").get("status") != "complete":
            raise ValueError(f"{name} training is incomplete")
    smoke = read_json(run / "final_smoke.json")
    if smoke.get("status") != "passed" or Path(smoke.get("run_dir", "")).resolve() != run:
        raise ValueError("Runtime verification has not passed for this run")
    generator = read_json(run / "generator/manifest.json")
    if smoke.get("generator_checkpoint_global_step") != generator["global_step"]:
        raise ValueError("Runtime verification belongs to a different generator checkpoint")
    if smoke.get("actual_history_lengths") != [1, 3, 1] or smoke.get("total_real_generation_candidates") != 16:
        raise ValueError("Runtime verification lacks expected session and generation checks")
    index = read_json(run / "response_index_manifest.json")
    for path, expected in (
        (Path(config["data_dir"]) / "train.jsonl", index["train_file_sha256"]),
        (run / "planner/best.pt", index["planner_checkpoint_sha256"]),
        (run / "space.json", index["space_sha256"]),
    ):
        if digest(path) != expected:
            raise ValueError(f"Training/index provenance mismatch: {path}")
    return dict(status="passed", **tests, runtime_smoke="final_smoke.json",
                runtime_smoke_sha256=digest(run / "final_smoke.json"),
                history_lengths=smoke["actual_history_lengths"],
                true_generation_candidates=smoke["total_real_generation_candidates"])


def verify_archive(destination, source):
    destination = Path(destination).resolve()
    manifest = read_json(destination / "version_manifest.json")
    if manifest.get("status") != "complete" or manifest.get("version") != destination.name:
        raise ValueError("Exported version metadata is incomplete or names a different version")
    if Path(manifest["source_run"]).resolve() != Path(source).resolve():
        raise ValueError("Exported version belongs to a different training run")
    if manifest.get("prefix_tensors_verified_equal") is not True:
        raise ValueError("Exported prefix equality verification is missing")
    total = 0
    for relative, info in manifest["files"].items():
        path = (destination / relative).resolve()
        if not path.is_relative_to(destination) or not path.is_file():
            raise ValueError(f"Invalid or missing archive file: {relative}")
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Archive integrity verification failed: {relative}")
        total += info["bytes"]
    actual = {str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()}
    if actual != set(manifest["files"]) | {"version_manifest.json"} or total != manifest["total_bytes"]:
        raise ValueError("Exported file inventory or total bytes differs from version manifest")
    return dict(status="complete", destination=str(destination), version=destination.name,
                archive_integrity_verified=True, files=len(actual), bytes=total)


def finalize_version(config_path, destination, unit_test_log):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Version already exists; refusing to overwrite: {destination}")
    from emostance_meme.paper_pipeline import finalize_run, load_config, write_json
    config = load_config(config_path)
    run = Path(config["run_dir"])
    if destination == run or run in destination.parents:
        raise ValueError("Archive must be separate from the training run")
    verification = completed_run_evidence(config, unit_test_log)
    # Check held-out evaluation/count consistency before writing final evidence.
    _, complete = build_report(run, Path(config["data_dir"]) / "metadata/split_manifest.json")
    if not complete:
        raise ValueError("Training/evaluation evidence is incomplete")
    diagnostics = tokenizer_diagnostics(config)
    if diagnostics["source"]["sample_count"] != read_json(run / "planner/manifest.json")["train_examples"]:
        raise ValueError("Tokenizer diagnostic count differs from trained planner examples")
    write_json(run / "tokenizer_diagnostics.json", diagnostics)
    write_json(run / "verification_summary.json", verification)
    finalize_run(config)
    report, complete = build_report(run, Path(config["data_dir"]) / "metadata/split_manifest.json")
    if not complete:
        raise ValueError("Final report cannot be generated from incomplete evidence")
    write_report(run / "training_report.md", report)
    export(run, destination)
    result = verify_archive(destination, run)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--unit-test-log", type=Path, required=True)
    args = parser.parse_args()
    try:
        finalize_version(args.config, args.destination, args.unit_test_log)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Cannot finalize model version: {exc}\n")


if __name__ == "__main__":
    main()
