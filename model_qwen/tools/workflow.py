"""Run all stages in order; persist honest progress for detached training."""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def build_stages(config, config_path):
    """Loss-only versions finish as review artifacts, without chat evaluation."""
    run = Path(config["run_dir"])
    planner = ("planner", [str(ROOT/"run.sh"), "train", "--stage", "planner", "--config", str(config_path)])
    generator = ("generator", [str(ROOT/"run.sh"), "train", "--stage", "generator", "--config", str(config_path)])
    final = [("index", [str(ROOT/"run.sh"), "index", "--config", str(config_path)]),
             ("evaluate", [str(ROOT/"run.sh"), "evaluate", "--config", str(config_path)])]
    if config.get("generator_selection_mode") == "dev_response_ce":
        # The downloaded backbone is shared and verified by the local loader.
        # Do not rewrite its download manifest or reuse a previous smoke run.
        return [planner, generator, *final]
    return [planner,
            ("download", [str(ROOT/"run.sh"), "download", "--output-dir", config["base_model_path"]]),
            ("generator_smoke", [str(ROOT/"run.sh"), "smoke", "--config", str(config_path)]),
            generator, ("quality", [str(ROOT/"run.sh"), "quality", "--config", str(config_path)]),
            *final,
            ("inference_smoke", [sys.executable, str(ROOT/"tools/verify_runtime.py"), "--model-dir", str(run), "--config", str(config_path)])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "paper_config.json"))
    args = parser.parse_args()
    args.config = str(Path(args.config).expanduser().resolve())
    sys.path.insert(0, str(ROOT / "src"))
    from emostance_qwen.paper_pipeline import load_config
    config = load_config(args.config)
    run = Path(config["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    with (run / "workflow.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "status": "running", "completed_stages": [],
                 "config": str(Path(args.config).resolve()),
                 "version":config.get("version", run.name), "model_version":config.get("model_version"),
                 "data_version":config.get("data_version"),
                 "generator_selection_mode":config.get("generator_selection_mode", "chat_quality"),
                 "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}

        def status(stage, outcome="running", **extra):
            state.update(stage=stage, status=outcome, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
            path = run / "workflow_status.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
            tmp.replace(path)
            print(json.dumps(state, ensure_ascii=False), flush=True)

        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT.parent)
        env["PYTHONUNBUFFERED"] = "1"
        stages = build_stages(config, args.config)
        try:
            for name, command in stages:
                status(name)
                subprocess.run(command, cwd=ROOT.parent, env=env, check=True)
                state["completed_stages"].append(name)
            if config.get("generator_selection_mode") == "dev_response_ce":
                status("ready_for_review", "ready_for_review", training_status="complete",
                       selection_criterion="dev_response_ce", chat_quality_status="not_evaluated",
                       inference_release_status="not_authorized")
            else:
                status("complete", "complete")
        except BaseException as exc:
            status(state.get("stage", "startup"), "failed", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise


if __name__ == "__main__":
    main()
