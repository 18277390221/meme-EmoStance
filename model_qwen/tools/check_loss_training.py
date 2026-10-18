"""Check two real updates in a new run, then resume them in the full workflow.

The checks are the first updates of the requested training version. They do
not create another model or use any previous Qwen prefix.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from emostance_qwen.paper_pipeline import load_config, train, write_json, file_sha256
    config = load_config(args.config)
    run = Path(config["run_dir"])
    if config.get("generator_selection_mode") != "dev_response_ce":
        raise ValueError("This check is only for the explicit loss-only training mode")
    if any((run / "generator" / name).exists() for name in ("best.pt", "last.pt", "progress.jsonl")):
        raise FileExistsError("Initial check requires a fresh prefix; use workflow to resume an existing run")
    config["generator_max_steps"] = 2
    result = train(config, "generator")
    if result["status"] != "incomplete":
        raise RuntimeError("Two updates unexpectedly completed the full training run")
    generator = json.loads((run / "generator/manifest.json").read_text(encoding="utf-8"))
    progress = [json.loads(line) for line in (run / "generator/progress.jsonl").read_text(encoding="utf-8").splitlines()]
    if generator["global_step"] != 2 or generator["epochs_complete"] != 0 or len(progress) != 2:
        raise RuntimeError("Initial check did not perform exactly the first two updates")
    for row in progress:
        if not math.isfinite(row["loss"]) or not math.isfinite(row["prefix_grad_norm"]) or row["prefix_grad_norm"] <= 0:
            raise RuntimeError("Real prefix training has nonfinite loss or missing gradients")
        if row["loss"] != row["reply_ce"] or row["reply_ce_weight"] != 1.0:
            raise RuntimeError("Training objective differs from pure reply CE")
        if any(row[key] != 0.0 for key in ("teacher_kl", "anchor_teacher_kl", "anchor_ce")):
            raise RuntimeError("Initial check unexpectedly enabled teacher or preservation loss")
    if generator["quality_status"] != "not_evaluated" or generator["selection_criterion"] != "dev_response_ce":
        raise RuntimeError("Loss-only training used an unexpected selection method")
    report = dict(status="passed", version=config["version"], model_version=config.get("model_version"),
        data_version=config.get("data_version"), full_training_complete=False, real_optimizer_updates=2,
        fresh_prefix=True, next_training_step=3, training_rows=generator["train_examples"],
        epochs_requested=generator["epochs_requested"], best_dev_response_ce=generator["best_dev_response_ce"],
        best_step=generator["best_step"], development_subset=generator["config"]["prefix_dev_subset_size"],
        best_checkpoint_sha256=generator["checkpoint_sha256"],
        last_checkpoint_sha256=file_sha256(run / "generator/last.pt"),
        chat_quality_evaluated=False, progress=progress,
        note="Continue the full workflow with the original configuration; it resumes these same first two updates.")
    write_json(run / "initial_training_check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
