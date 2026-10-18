"""Prepare a new version with the verified V2 planner and a fresh Qwen prefix.

Only the planner, emotion space and graph are reused. Existing runs are never
overwritten, and neither generator weights nor optimizer state are copied.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from emostance_qwen.paper_pipeline import load_config, read_rows, file_sha256, write_json
    from emostance_qwen.paper_planner import _options, _signature, training_statistics
    from emostance_qwen.space import EmotionSpace
    config = load_config(args.config)
    if config.get("generator_selection_mode") != "dev_response_ce" or config.get("chat_quality_enabled") is not False:
        raise ValueError("Preparation requires an explicitly loss-only configuration")
    if not config.get("model_version") or not config.get("version"):
        raise ValueError("A new model version and run version are required")
    run, source = Path(config["run_dir"]), Path(config["planner_reuse_source"])
    if run.exists() or run == source:
        raise FileExistsError("Choose a new run_dir; preparation never overwrites existing runs")
    source_manifest = json.loads((source / "planner/manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete":
        raise ValueError("Source planner must have completed training")
    train = read_rows(Path(config["data_dir"]) / "train.jsonl")
    dev = read_rows(Path(config["data_dir"]) / "dev.jsonl")
    space = EmotionSpace.load(source / "space.json")
    options = _options(config)
    stats = training_statistics(train, space, options["transition_alpha"], options["class_weight_power"])
    verified_space = copy.deepcopy(space)
    verified_space.prototypes = stats["target_prototypes"].astype("float64")
    signature = _signature(train, dev, verified_space, options)
    planner_config = json.loads((source / "planner/config.json").read_text(encoding="utf-8"))
    if signature != source_manifest.get("training_signature") or signature != planner_config.get("training_signature"):
        raise ValueError("Source planner does not match the new version's data, space or planner hyperparameters")
    source_hashes = {name: file_sha256(source / name) for name in
                     ("planner/best.pt", "planner/config.json", "space.json", "graph/manifest.json")}
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source / "planner", run / "planner", ignore=shutil.ignore_patterns("last.pt", "*.tmp"))
    shutil.copytree(source / "graph", run / "graph")
    shutil.copy2(source / "space.json", run / "space.json")
    updated = {**source_manifest, "checkpoint": str(run / "planner/best.pt"),
               "resume_checkpoint": None, "reused_from": str(source / "planner"),
               "source_checkpoint_sha256": source_hashes["planner/best.pt"]}
    write_json(run / "planner/manifest.json", updated)
    for name, expected in source_hashes.items():
        if file_sha256(run / name) != expected:
            raise RuntimeError(f"Reused planner artifact changed while copying: {name}")
    record = dict(status="complete", version=config["version"], model_version=config["model_version"],
                  data_version=config.get("data_version"), source_run=str(source),
                  training_signature=signature, train_examples=len(train), dev_examples=len(dev),
                  artifact_sha256=source_hashes, optimizer_copied=False,
                  generator_prefix_copied=False, backbone_copied=False)
    write_json(run / "planner_reuse.json", record)
    write_json(run / "version.json", dict(version=config["version"], model_version=config["model_version"],
               data_version=config.get("data_version"), config_path=str(args.config.resolve()),
               config_sha256=file_sha256(args.config), run_dir=str(run),
               training_objective="reply_ce", selection_criterion="dev_response_ce",
               previous_generator_reused=False, automatic_inference_cutover=False))
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
