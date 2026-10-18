"""Read download/training progress without loading any neural model."""
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def read(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def last_record(path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 16384))
        for line in reversed(handle.read().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_status(config_path=None):
    config_path = Path(config_path or ROOT / "paper_config.json").expanduser().resolve()
    config = read(config_path)
    if not isinstance(config, dict) or not config.get("run_dir"):
        raise ValueError(f"Missing training configuration: {config_path}")
    run_path = Path(config["run_dir"]).expanduser()
    run = (run_path if run_path.is_absolute() else config_path.parent / run_path).resolve()
    workflow = read(run / "workflow_status.json", {"status": "not_started"})
    try:
        os.kill(workflow["pid"], 0)
        workflow["process_alive"] = True
    except (OSError, KeyError):
        workflow["process_alive"] = False
    loss_only = config.get("generator_selection_mode") == "dev_response_ce"
    final = read(run / "manifest.json", {})
    report = {"config": str(config_path), "run_dir": str(run),
              "version": config.get("version", run.name), "model_version": config.get("model_version"),
              "data_version": config.get("data_version"),
              "generator_selection_mode": config.get("generator_selection_mode", "chat_quality"),
              "workflow": workflow,
              "download": read(ROOT / "logs/download_state.json", {"status": "not_started"}),
              "planner": read(run / "planner/manifest.json") or last_record(run / "planner/metrics.jsonl") or {"status": "not_started"},
              "generator": read(run / "generator/manifest.json") or last_record(run / "generator/progress.jsonl") or {"status": "not_started"},
              "generator_progress": last_record(run / "generator/progress.jsonl"),
              "chat_quality": {"status": "not_evaluated", "reason": "loss_only_selection"} if loss_only else
                  last_record(run / "generator/quality_metrics.jsonl") or {"status": "not_evaluated"},
              "chat_holdout": {"status": "not_evaluated", "reason": "loss_only_selection"} if loss_only else
                  read(run / "generator/quality_holdout_attempt.json", {"status": "not_attempted"}),
              "final_model": final.get("status", "not_complete"),
              "training_complete": final.get("training_status") == "complete" or final.get("status") == "complete",
              "ready_for_review": final.get("status") == "ready_for_review",
              "inference_release_status": final.get("inference_release_status"),
              "real_inference": "not_authorized" if loss_only else read(run / "final_smoke.json", {"status": "not_run"}).get("status")}
    if loss_only:
        latest = last_record(run / "generator/loss_metrics.jsonl")
        selection = read(run / "generator/loss_selection.json")
        use_live = workflow.get("status") == "running" and latest is not None
        report["loss_selection"] = latest if use_live else selection or latest
        report["loss_selection_source"] = "live_dev_response_ce" if use_live else "final_selection" if selection else "live_dev_response_ce"
    inference_config = read(ROOT / "inference_config.json")
    if inference_config:
        inference_run = (ROOT / inference_config["run_dir"]).resolve()
        selected = read(inference_run / "deployment_selection.json", {})
        active_manifest = read(inference_run / "manifest.json", {})
        report["active_inference"] = {
            "config": str(ROOT / "inference_config.json"), "run_dir": str(inference_run),
            "deployment": read(inference_run / "deployment_status.json", {"status": "not_started"}),
            "model_status": active_manifest.get("status", "not_ready"),
            "quality_status": active_manifest.get("quality_status", selected.get("quality_status")),
            "release_mode": selected.get("release_mode"),
            "selected_step": selected.get("selected_checkpoint", {}).get("global_step"),
            "real_inference": read(inference_run / "final_smoke.json", {"status": "not_run"}).get("status")}
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper_config.json")
    args = parser.parse_args()
    print(json.dumps(collect_status(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
