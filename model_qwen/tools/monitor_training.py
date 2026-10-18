"""Observe a training workflow until it completes; never alter model state."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def records_tail(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 262144))
            data = stream.read()
    except FileNotFoundError:
        return []
    rows = []
    for line in data.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except ValueError:
            pass
    return rows


def alive(pid):
    try:
        state = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state not in ("Z", "X")
    except (OSError, ValueError, TypeError):
        return False


def snapshot(run, config):
    workflow = read_json(run / "workflow_status.json", {})
    stage = workflow.get("stage", "unknown")
    is_alive = alive(workflow.get("pid"))
    result = {"checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "monitor_pid": os.getpid(), "workflow_pid": workflow.get("pid"),
              "workflow_alive": is_alive, "status": workflow.get("status", "unknown"),
              "stage": stage, "completed_stages": workflow.get("completed_stages", []),
              "disk_free_gib": round(shutil.disk_usage(run).free / 2**30, 2), "alerts": []}
    result.update(version=config.get("version", run.name), model_version=config.get("model_version"),
                  data_version=config.get("data_version"),
                  generator_selection_mode=config.get("generator_selection_mode", "chat_quality"))
    if result["status"] == "running" and not is_alive:
        result["status"] = "process_exited_without_final_status"
        result["alerts"].append("Workflow process exited before recording a terminal state")
    if workflow.get("error"):
        result["alerts"].append(workflow["error"])
    if stage in ("planner", "generator"):
        path = run / stage / ("metrics.jsonl" if stage == "planner" else "progress.jsonl")
        rows = records_tail(path)
        steps = [r for r in rows if ("global_step" if stage == "planner" else "step") in r and "elapsed_seconds" in r]
        if steps:
            latest = steps[-1]
            key = "global_step" if stage == "planner" else "step"
            total = (latest.get("batches", 0) * config["planner_epochs"] if stage == "planner"
                     else latest.get("epoch_total", 0) * config["prefix_epochs"])
            result.update(epoch=latest.get("epoch"), step=latest[key], total_steps=total,
                          percent=round(100 * latest[key] / total, 2) if total else None,
                          latest_loss=latest.get("loss"), log_age_seconds=round(time.time() - path.stat().st_mtime, 1))
            for metric in ("reply_ce", "teacher_kl", "anchor_teacher_kl"):
                if metric in latest:
                    result[metric] = latest[metric]
            # Keep a two-minute window and reject elapsed-time resets on resume.
            recent = [r for r in steps if 0 <= latest["elapsed_seconds"] - r["elapsed_seconds"] <= 120 and r[key] < latest[key]]
            if recent:
                first = recent[0]
                elapsed = latest["elapsed_seconds"] - first["elapsed_seconds"]
                if elapsed >= 20:
                    speed = (latest[key] - first[key]) / elapsed
                    result.update(recent_steps_per_second=round(speed, 3),
                                  training_steps_eta_minutes=round((total - latest[key]) / speed / 60, 1))
            if result["log_age_seconds"] > 600:
                result["alerts"].append("Training progress log has not updated for ten minutes; check the active process")
            loss = latest.get("loss")
            if isinstance(loss, (int, float)) and not math.isfinite(loss):
                result["alerts"].append("Nonfinite training loss")
        epochs = [r for r in rows if r.get("event") == "epoch"]
        if epochs:
            result["last_completed_epoch"] = epochs[-1]
    final = read_json(run / "manifest.json", {})
    result["final_model_complete"] = final.get("status") == "complete"
    result["training_complete"] = final.get("training_status") == "complete" or final.get("status") == "complete"
    result["ready_for_review"] = final.get("status") == "ready_for_review"
    result["inference_release_status"] = final.get("inference_release_status")
    result["real_inference_complete"] = read_json(run / "final_smoke.json", {}).get("status") == "complete"
    if config.get("generator_selection_mode") == "dev_response_ce":
        rows = records_tail(run/"generator/loss_metrics.jsonl")
        if rows:
            result["latest_dev_response_ce"] = rows[-1]
        selection = read_json(run/"generator/loss_selection.json")
        use_live = workflow.get("status") == "running" and bool(rows)
        if use_live or selection:
            result["loss_selection"] = rows[-1] if use_live else selection
            result["loss_selection_source"] = "live_dev_response_ce" if use_live else "final_selection"
        result.update(selection_criterion="dev_response_ce", chat_quality_status="not_evaluated",
                      chat_release_status="not_evaluated", real_inference_complete=False)
    elif config.get("chat_quality_enabled"):
        baseline = read_json(run / "generator/quality_baseline_dev.json", {}).get("report")
        if baseline:
            result["chat_baseline"] = {key: baseline.get(key) for key in ("score", "hard_failures", "pass_rate")}
        quality_rows = records_tail(run / "generator/quality_metrics.jsonl")
        if quality_rows:
            result["latest_chat_comparison"] = quality_rows[-1]
        manifest = read_json(run / "generator/manifest.json", {})
        result["chat_release_status"] = manifest.get("quality_status", "awaiting_training_and_acceptance")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper_config.json")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("--interval must be at least five seconds")
    config = read_json(args.config)
    run = (args.config.resolve().parent / config["run_dir"]).resolve()
    if not run.is_relative_to(ROOT):
        raise ValueError("Monitoring output must remain below model_qwen/")
    if args.once:
        print(json.dumps(snapshot(run, config), ensure_ascii=False, indent=2))
        return
    with (run / "monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("A training monitor is already running", flush=True)
            return
        (run / "monitor.pid").write_text(str(os.getpid()) + "\n")
        while True:
            result = snapshot(run, config)
            target = run / "monitor_status.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(target)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] in ("complete", "ready_for_review", "failed", "process_exited_without_final_status"):
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
