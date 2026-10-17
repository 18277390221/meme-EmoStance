"""Offline EmoStance training pipeline for the four-annotator Chinese dataset."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from .space import EmotionSpace

MODEL_DIR = Path(__file__).resolve().parents[2]


def read_rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_config(path=None):
    path = Path(path or MODEL_DIR/"paper_config.json").resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("run_dir", "data_dir"):
        value = Path(config[key])
        config[key] = str((path.parent/value).resolve() if not value.is_absolute() else value)
    run = Path(config["run_dir"])
    if not run.is_relative_to(MODEL_DIR):
        raise ValueError("Training outputs must be below model/")
    return config


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def finalize_run(config):
    root = Path(config["run_dir"])
    stages = {name:json.loads((root/name/"manifest.json").read_text()) for name in ("planner","generator")}
    if any(stage.get("status") != "complete" for stage in stages.values()):
        raise ValueError("Both planner and generator must complete before finalizing")
    evaluation = json.loads((root/"evaluation.json").read_text())
    index = json.loads((root/"response_index_manifest.json").read_text())
    graph = json.loads((root/"graph/manifest.json").read_text())
    manifest = dict(format_version=2,status="complete",method="EmoStance Chinese meme adaptation",
        config=config,seed=config.get("seed",42),planner=stages["planner"],generator=stages["generator"],
        graph=graph,index=index,evaluation_file=str(root/"evaluation.json"),
        evaluation_summary={split:{key:val for key,val in evaluation[split].items() if key != "generator"} for split in ("dev","test")},
        inference=dict(num_candidates=config.get("num_candidates",4),reranking="minimum predicted-control soft cross entropy",
            no_repeat_ngram_size=config.get("no_repeat_ngram_size",4),
            meme_selection="nearest training response to generated reply source-distribution @ source-prototypes vector",
            idle_timeout_seconds=config.get("idle_timeout",10)),
        original_emostance_modified=False,base_model_weights_copied=False,
        adaptations=["Chinese character/Latin word TF-IDF instead of English-only regex",
            "user-selected grouped 80/10/10 split; seed42",
            "source-entropy gate from paper, rather than target-entropy gate in released code",
            "mean MSE vector auxiliary loss follows released implementation rather than paper L2 sum notation",
            f"Leiden discovers {graph['num_clusters']} regions for this dataset",
            "training-response meme retrieval and ten-second conversation reset added for application"])
    for name in ("final_smoke","verification_summary"):
        if (root/f"{name}.json").exists():
            manifest[name] = json.loads((root/f"{name}.json").read_text())
    write_json(root/"manifest.json",manifest)
    return manifest


def prepare_space(config):
    from .paper_space import build_space
    run = Path(config["run_dir"])
    run.mkdir(parents=True,exist_ok=True)
    data = Path(config["data_dir"])
    manifest_bytes = (data/"metadata/split_manifest.json").read_bytes()
    execution_keys = {"generator_runtime_gradient_checkpointing","generator_runtime_checkpoint_interval",
        "do_sample","temperature","top_p","max_new_tokens","retrieval_metric","idle_timeout",
        "num_candidates","no_repeat_ngram_size","repetition_penalty"}
    method_config = {key:value for key,value in config.items() if key not in execution_keys}
    provenance = dict(split_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        split_manifest=str(data/"metadata/split_manifest.json"),config=method_config,
        original_emostance_modified=False)
    existing = run/"provenance.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        previous["config"] = {key:value for key,value in previous["config"].items() if key not in execution_keys}
        if previous != provenance:
            raise ValueError("Run configuration or dataset changed; choose a new run_dir")
    write_json(existing,provenance)
    if (run/"space.json").exists():
        return EmotionSpace.load(run/"space.json")
    space, report = build_space(read_rows(data/"annotation_rows/train.jsonl"),
        read_rows(data/"train.jsonl"),run,config)
    print(json.dumps({"stage":"graph","status":"complete",**report},ensure_ascii=False),flush=True)
    return space


def train(config,stage="all"):
    import torch
    torch.set_num_threads(min(8,os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started = time.time()
    space = prepare_space(config)
    if stage == "space":
        return
    data,run = Path(config["data_dir"]),Path(config["run_dir"])
    train_rows, dev_rows = read_rows(data/"train.jsonl"),read_rows(data/"dev.jsonl")
    if stage in ("all","planner"):
        from .paper_planner import train_planner
        planner = train_planner(train_rows,dev_rows,space,run,config)
        write_json(run/"planner_result.json",planner)
        gc.collect()
        torch.cuda.empty_cache()
    if stage in ("all","generator"):
        from .paper_generator import train_generator
        generator = train_generator(train_rows,space,run,config)
        write_json(run/"generator_result.json",generator)
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"stage":stage,"elapsed_seconds":time.time()-started,"status":"complete"}),flush=True)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["train","index","evaluate","chat","serve","smoke"])
    parser.add_argument("--config",default=None)
    parser.add_argument("--stage",choices=["all","space","planner","generator"],default="all")
    parser.add_argument("--text",default="今天有点累，想休息一下。")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8790)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "train":
        train(config,args.stage)
    elif args.command == "index":
        from .paper_engine import build_response_index
        build_response_index(config)
    elif args.command == "evaluate":
        from .paper_engine import evaluate
        print(json.dumps(evaluate(config),ensure_ascii=False,indent=2))
    elif args.command == "smoke":
        from .paper_generator import smoke_train_step
        space = prepare_space(config)
        rows = read_rows(Path(config["data_dir"])/"train.jsonl")
        print(smoke_train_step(rows[0],space,MODEL_DIR/"runs/smoke_v1",config))
    else:
        from .paper_engine import PaperMemeEngine
        engine = PaperMemeEngine(config)
        if args.command == "serve":
            from .http_server import serve
            serve(engine,host=args.host,port=args.port,idle_timeout=config.get("idle_timeout",10))
        else:
            from .service import ChatService
            with ChatService(engine,idle_timeout=config.get("idle_timeout",10)) as service:
                if args.text:
                    print(json.dumps(service.reply(args.text),ensure_ascii=False))
                else:
                    for line in sys.stdin:
                        if line.strip():
                            print(json.dumps(service.reply(line.strip()),ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
