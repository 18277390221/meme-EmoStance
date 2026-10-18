"""Read-only replay of incident contexts with the same Qwen, with/without prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "inference_config.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729,2718])
    args = parser.parse_args()
    import numpy as np
    import torch
    from emostance_qwen.paper_pipeline import load_config, require_inference_release, file_sha256, write_json
    from emostance_qwen.paper_generator import PaperGenerator
    from emostance_qwen.paper_planner import load_planner
    torch.set_num_threads(8)
    config = load_config(args.config)
    require_inference_release(config)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    run = Path(config["run_dir"])
    checkpoint = run / "generator/best.pt"
    generator = PaperGenerator(config, checkpoint=checkpoint)
    planner = load_planner(run, config)
    report = {"status":"running", "model_dir":str(run), "prefix_sha256":file_sha256(checkpoint),
        "input_file_sha256":file_sha256(args.cases), "seeds":args.seeds,
        "server_state_modified":False, "weights_modified":False,
        "context_scope":"Replayed incident excerpt; online full prompts and session boundaries were not recorded.",
        "cases":[]}
    for case in cases:
        messages = case["messages"]
        history = [{"role":"A" if m["role"]=="user" else "B", "text":m["content"]} for m in messages]
        vector = planner.predict(history,"B")["target_vector"]
        token_ids = generator._prompt_ids(messages, config["system_prompt"],
                                          config.get("chat_quality_max_prompt_tokens",4096))
        entry = {"id":case["id"], "messages":messages, "prompt_tokens":int(token_ids.shape[1]),
                 "full_excerpt_retained":True, "outputs":[]}
        for seed in args.seeds:
            for mode in ("no_prefix","trained_prefix"):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                started = time.monotonic()
                reply = generator.complete(config["system_prompt"],messages,
                    max_new_tokens=config["max_new_tokens"],
                    stance_vector=vector if mode=="trained_prefix" else None,
                    do_sample=config["do_sample"], temperature=config["temperature"],
                    top_p=config["top_p"],top_k=config.get("top_k",20),
                    no_repeat_ngram_size=config["no_repeat_ngram_size"],
                    max_prompt_tokens=config.get("chat_quality_max_prompt_tokens",4096))
                output={"seed":seed,"mode":mode,"reply":reply,
                        "elapsed_seconds":round(time.monotonic()-started,3)}
                entry["outputs"].append(output)
                print(json.dumps({"case":case["id"],**output},ensure_ascii=False),flush=True)
        report["cases"].append(entry)
        write_json(args.output,report)
    report["status"]="complete"
    write_json(args.output,report)


if __name__ == "__main__":
    main()
