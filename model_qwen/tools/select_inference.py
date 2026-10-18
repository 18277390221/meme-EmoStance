"""Prepare the completed final prefix for an explicitly requested deployment.

The original training run and its rejected reports remain unchanged. This tool
does not relabel conversational quality as passed or resume model training.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def prepare(source, destination, config_path):
    import torch
    from emostance_qwen.paper_pipeline import file_sha256, load_config, write_json
    source, destination = source.resolve(), destination.resolve()
    if source == destination or not destination.is_relative_to(ROOT):
        raise ValueError("A separate deployment directory below model_qwen is required")
    if destination.exists():
        raise FileExistsError(f"Deployment already exists; inspect it before selecting again: {destination}")
    generator_manifest = json.loads((source / "generator/manifest.json").read_text(encoding="utf-8"))
    expected_steps = generator_manifest["train_examples"] * generator_manifest["epochs_requested"]
    if (generator_manifest["status"] != "complete" or generator_manifest["global_step"] != expected_steps
            or generator_manifest["epochs_complete"] < generator_manifest["epochs_requested"]):
        raise ValueError("Only a completed, fully trained final checkpoint can be selected")
    checkpoint = source / "generator/last.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (state["global_step"] != expected_steps or state["next_offset"] != 0
            or state["next_epoch"] != generator_manifest["epochs_complete"]
            or state["training_signature"] != generator_manifest["training_signature"]):
        raise ValueError("The final checkpoint does not match the completed training manifest")
    destination.mkdir(parents=True)
    shutil.copytree(source / "planner", destination / "planner",
                    ignore=shutil.ignore_patterns("last.pt", "*.lock"))
    shutil.copytree(source / "graph", destination / "graph")
    shutil.copy2(source / "space.json", destination / "space.json")
    (destination / "generator").mkdir()
    # Copy only inference parameters and metadata; optimizer/RNG state stays in
    # the original resumable checkpoint. Tensor equality is checked below.
    excluded = {"optimizer", "torch_rng", "python_rng", "numpy_rng", "cuda_rng"}
    slim = {key: value for key, value in state.items() if key not in excluded}
    slim["inference_only"] = True
    slim_path = destination / "generator/best.pt"
    torch.save(slim, slim_path)
    restored = torch.load(slim_path, map_location="cpu", weights_only=False, mmap=True)
    if set(restored["projector"]) != set(state["projector"]) or any(
            not torch.equal(value, restored["projector"][name]) for name, value in state["projector"].items()):
        raise ValueError("Exported prefix parameters differ from the selected final checkpoint")
    raw_config = json.loads((ROOT / "paper_config.json").read_text(encoding="utf-8"))
    raw_config["run_dir"] = str(destination.relative_to(ROOT))
    raw_config["deployment_release_mode"] = "user_selected_with_known_quality_failures"
    write_json(config_path, raw_config)
    config = load_config(config_path)
    planner_manifest_path = destination / "planner/manifest.json"
    planner_manifest = json.loads(planner_manifest_path.read_text(encoding="utf-8"))
    planner_manifest.update(checkpoint=str(destination / "planner/best.pt"), deployment_source=str(source / "planner"))
    write_json(planner_manifest_path, planner_manifest)
    generator_manifest.update(selection_criterion="user_selected_with_known_quality_failures",
        source_selection_criterion=generator_manifest["selection_criterion"],
        best_checkpoint=str(slim_path), last_checkpoint=None,
        best_step=expected_steps, selected_step=expected_steps, config=config,
        checkpoint_sha256=file_sha256(slim_path), quality_report=None,
        quality_report_sha256=None, source_training_run=str(source))
    write_json(destination / "generator/manifest.json", generator_manifest)
    shutil.copy2(source / f"generator/quality_step_{expected_steps:06d}.json", destination / "generator/source_quality_dev.json")
    if (ROOT / "logs/checkpoint_selection_review.json").exists():
        shutil.copy2(ROOT / "logs/checkpoint_selection_review.json", destination / "selection_review.json")
    preparation = dict(status="prepared", source_run=str(source), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=file_sha256(checkpoint),
        source_training_manifest_sha256=file_sha256(source / "generator/manifest.json"),
        selected_checkpoint_sha256=file_sha256(slim_path), global_step=expected_steps,
        tensor_parameters_identical=True, optimizer_and_rng_exported=False,
        source_development_report_is_reference_only=True)
    write_json(destination / "preparation.json", preparation)
    del state, slim, restored
    gc.collect()
    return config, preparation


def assess(config, preparation, authorization_text, reason):
    import torch
    from emostance_qwen.paper_generator import PaperGenerator
    from emostance_qwen.paper_planner import load_planner
    from emostance_qwen.paper_pipeline import (assess_selected_quality, build_quality_report,
        deployment_selection_binding, file_sha256, write_json)
    root = Path(config["run_dir"])
    generator = PaperGenerator(config, checkpoint=root / "generator/best.pt")
    planner = load_planner(root, config)
    # This is a new real assessment of the exact exported parameters. A prior
    # dev report is not silently rebound after checkpoint metadata changed.
    cache = Path(preparation["source_run"]) / "generator/quality_baseline_dev.json"
    baseline = json.loads(cache.read_text(encoding="utf-8"))["report"]
    report = build_quality_report(config, generator, planner, phase="dev",
                                  checkpoint_path=root / "generator/best.pt", baseline=baseline)
    report["global_step"] = preparation["global_step"]
    write_json(root / "generator/quality_dev.json", report)
    manifest_path = root / "generator/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(quality_status=report["status"], quality_report="generator/quality_dev.json",
                    quality_report_sha256=file_sha256(root / "generator/quality_dev.json"))
    write_json(manifest_path, manifest)
    selection = dict(format_version=1, release_mode="user_selected_with_known_quality_failures",
        status="pending_quality", user_selection=dict(authorized=True, source="user", accept_known_quality_failures=True,
            authorization_text=authorization_text, selection_reason=reason),
        selected_checkpoint={key: preparation[key] for key in ("source_run", "source_checkpoint",
            "source_checkpoint_sha256", "source_training_manifest_sha256", "global_step")},
        binding=deployment_selection_binding(config),
        quality_reports={"dev":dict(path="generator/quality_dev.json",
                                    sha256=file_sha256(root / "generator/quality_dev.json"))},
        quality_status=report["status"])
    write_json(root / "deployment_selection.json", selection)
    del generator, planner
    gc.collect()
    torch.cuda.empty_cache()
    result = assess_selected_quality(config)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/qwen3_8b_v2_quality")
    parser.add_argument("--destination", type=Path, default=ROOT / "runs/qwen3_8b_v2_selected")
    parser.add_argument("--config", type=Path, default=ROOT / "inference_config.json")
    parser.add_argument("--authorization-text", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-prepared", action="store_true")
    args = parser.parse_args()
    from emostance_qwen.paper_pipeline import load_config, write_json
    if args.resume_prepared:
        config = load_config(args.config)
        if Path(config["run_dir"]).resolve() != args.destination.resolve():
            raise ValueError("Resume configuration and deployment destination differ")
        preparation = json.loads((args.destination / "preparation.json").read_text(encoding="utf-8"))
    else:
        config, preparation = prepare(args.source, args.destination, args.config)
    if args.prepare_only:
        print(json.dumps(preparation, ensure_ascii=False), flush=True)
        return
    completed, started = [], time.time()
    state_path = args.destination / "deployment_status.json"

    def record(stage, status="running", **extra):
        write_json(state_path, dict(pid=os.getpid(), status=status, stage=stage,
            started_at=started, updated_at=time.time(), completed_stages=completed, **extra))
        print(json.dumps(dict(stage=stage, status=status, **extra), ensure_ascii=False), flush=True)

    try:
        record("chat_assessment")
        selection_path = args.destination / "deployment_selection.json"
        existing = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None
        if existing is not None and existing.get("status") == "ready":
            from emostance_qwen.paper_pipeline import require_inference_release
            require_inference_release(config)
        else:
            assess(config, preparation, args.authorization_text, args.reason)
        completed.append("chat_assessment")
        for stage, command in (
            ("response_index", [str(ROOT / "run.sh"), "index", "--config", str(args.config)]),
            ("evaluation", [str(ROOT / "run.sh"), "evaluate", "--config", str(args.config)]),
            ("real_inference", [sys.executable, str(ROOT / "tools/verify_runtime.py"),
                                "--model-dir", str(args.destination), "--config", str(args.config)])):
            record(stage)
            if stage == "response_index" and (args.destination / "response_index_manifest.json").is_file():
                from emostance_qwen.paper_engine import PaperResponseIndex
                # A completed, matching training-only index can be reused when
                # resuming deployment; this never reuses an incomplete file.
                PaperResponseIndex(config)
                print(json.dumps(dict(stage=stage, reused_verified_index=True)), flush=True)
            else:
                subprocess.run(command, check=True, cwd=ROOT.parent)
            completed.append(stage)
        record("complete", status="complete", selected_step=preparation["global_step"])
    except BaseException as exc:
        record("failed", status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
