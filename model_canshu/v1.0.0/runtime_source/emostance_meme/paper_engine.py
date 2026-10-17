"""Generate a reply, then retrieve its nearest training response's meme id."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .paper_pipeline import read_rows,write_json
from .space import EmotionSpace


def dominant_meme(distribution):
    return min(distribution,key=lambda meme:(-distribution[meme],meme))


def build_response_index(config):
    import torch
    from .paper_planner import load_planner
    root,data = Path(config["run_dir"]),Path(config["data_dir"])
    space = EmotionSpace.load(root/"space.json")
    rows = read_rows(data/"train.jsonl")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Response bank must contain training examples only")
    torch.set_num_threads(8)
    planner = load_planner(root,config)
    reply_rows = [{"history":row["history"]+[{"role":row["next_role"],"text":row["response"]}],
        "source_role":row["next_role"],"next_role":"A" if row["next_role"]=="B" else "B"} for row in rows]
    vectors = planner.predict_rows(reply_rows)["source_realized_vector"]
    anchors = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Response bank must contain training examples only")
        anchors.append(dict(sample_id=row["sample_id"],dialogue_id=row["dialogue_id"],
            turn_id=row["target_turn_id"],reply=row["response"],split="train",
            meme_id=dominant_meme(row["target_meme_distribution"]),
            meme_distribution=row["target_meme_distribution"]))
    np.savez_compressed(root/"response_vectors.npz",vectors=np.asarray(vectors,dtype=np.float32))
    with (root/"response_anchors.jsonl").open("w",encoding="utf-8") as f:
        for anchor in anchors:
            f.write(json.dumps(anchor,ensure_ascii=False)+"\n")
    report = dict(split="train",responses=len(rows),vector_dim=space.vector_dim,
        vectors="full training history plus reply -> source soft distribution @ training source prototypes; same encoder as generated queries",
        selection="meme id with highest confidence-weighted annotator vote; lexical id breaks ties",
        train_file_sha256=hashlib.sha256((data/"train.jsonl").read_bytes()).hexdigest(),
        planner_checkpoint_sha256=hashlib.sha256((root/"planner/best.pt").read_bytes()).hexdigest(),
        space_sha256=hashlib.sha256((root/"space.json").read_bytes()).hexdigest())
    write_json(root/"response_index_manifest.json",report)
    del planner
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"stage":"response_index",**report},ensure_ascii=False),flush=True)
    return report


class PaperResponseIndex:
    def __init__(self,config):
        root = Path(config["run_dir"])
        report = json.loads((root/"response_index_manifest.json").read_text())
        if report["split"] != "train" or report["space_sha256"] != hashlib.sha256((root/"space.json").read_bytes()).hexdigest():
            raise ValueError("Response index provenance does not match the training space")
        if report["train_file_sha256"] != hashlib.sha256((Path(config["data_dir"])/"train.jsonl").read_bytes()).hexdigest():
            raise ValueError("Response index provenance does not match the training file")
        if report.get("planner_checkpoint_sha256") != hashlib.sha256((root/"planner/best.pt").read_bytes()).hexdigest():
            raise ValueError("Response index provenance does not match the planner checkpoint")
        self.anchors = read_rows(root/"response_anchors.jsonl")
        self.vectors = np.load(root/"response_vectors.npz",allow_pickle=False)["vectors"]
        if self.vectors.ndim != 2 or len(self.vectors) != len(self.anchors) or len(self.anchors) != report["responses"] or self.vectors.shape[1] != report["vector_dim"] or any(r["split"] != "train" for r in self.anchors):
            raise ValueError("Invalid response index")
        self.norms = np.linalg.norm(self.vectors,axis=1)
        if not np.isfinite(self.vectors).all() or np.any(self.norms == 0):
            raise ValueError("Invalid training response vectors")
        self.normalized = self.vectors/self.norms[:,None]
        self.metric = config.get("retrieval_metric","cosine")
        if self.metric not in ("cosine","euclidean"):
            raise ValueError("Unknown emotion vector distance")

    def nearest(self,vector):
        vector = np.asarray(vector,dtype=np.float32)
        if vector.shape != self.vectors.shape[1:] or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise ValueError("Expected a finite, nonzero emotion vector")
        if self.metric == "cosine":
            distances = 1-np.clip(self.normalized @ (vector/np.linalg.norm(vector)),-1,1)
        else:
            distances = np.linalg.norm(self.vectors-vector,axis=1)
        best = float(distances.min())
        # Stable tie order independent of file ordering, including equal labels.
        tied = np.flatnonzero(distances == best)
        index = min(tied,key=lambda i:(self.anchors[i]["meme_id"],self.anchors[i]["reply"],self.anchors[i]["sample_id"]))
        return {**self.anchors[index],"distance":best}


class PaperMemeEngine:
    def __init__(self,config):
        import torch
        from .paper_planner import load_planner
        from .paper_generator import PaperGenerator
        root = Path(config["run_dir"])
        for stage in ("planner","generator"):
            manifest = json.loads((root/stage/"manifest.json").read_text())
            if manifest.get("status") != "complete":
                raise ValueError(f"{stage} training has not completed")
        torch.set_num_threads(8)
        self.config = config
        self.planner = load_planner(root,config)
        self.generator = PaperGenerator(config)
        self.index = PaperResponseIndex(config)
        self._lock = threading.Lock()

    def reply(self,messages):
        history = [{"role":"A" if m["role"]=="user" else "B","text":m["content"]} for m in messages]
        with self._lock:
            plan = self.planner.predict(history,"B")
            candidates = []
            count = int(self.config.get("num_candidates",4)) if self.config.get("do_sample",True) else 1
            if count < 1:
                raise ValueError("num_candidates must be positive")
            for _ in range(count):
                text = self.generator.generate(history,"B",plan["target_vector"],self.config.get("max_new_tokens",64))
                realized = self.planner.predict(history+[{"role":"B","text":text}],"A")
                # Reference rerank_control: CE against the predicted orientation;
                # no oracle target annotation is visible at inference.
                score = -float(np.sum(plan["target_distribution"] * np.log(np.maximum(realized["source_distribution"],1e-12))))
                candidates.append((score,text,realized))
            _,reply,realized = min(candidates,key=lambda item:item[0])
            # Reconstruct the selected reply from its source soft distribution
            # and training source prototypes, matching the paper's orientation
            # representation. The auxiliary direct regression head can drift
            # outside this space and is not used as a retrieval query.
            nearest = self.index.nearest(realized["source_realized_vector"])
            return {"reply":reply,"meme_id":nearest["meme_id"]}


def distribution_metrics(actual,predicted):
    actual,predicted = np.asarray(actual),np.asarray(predicted)
    midpoint = (actual+predicted)/2
    log = lambda x:np.log(np.maximum(x,1e-12))
    return dict(soft_ce=float(-(actual*log(predicted)).sum(1).mean()),
        jsd=float((.5*(actual*(log(actual)-log(midpoint))+predicted*(log(predicted)-log(midpoint))).sum(1)).mean()),
        overlap=float(np.minimum(actual,predicted).sum(1).mean()))


def evaluate(config):
    """Once-only held-out planner metrics; never select checkpoints using test."""
    import torch
    from .paper_planner import load_planner
    root,data = Path(config["run_dir"]),Path(config["data_dir"])
    torch.set_num_threads(8)
    planner = load_planner(root,config)
    space = planner.space
    train_rows = read_rows(data/"train.jsonl")
    mean_target = np.mean([space.targets(r["target_meme_distribution"])[0] for r in train_rows],axis=0)
    index = PaperResponseIndex(config)
    result = dict(method="EmoStance Chinese meme adaptation",seed=config.get("seed",42),
        train_examples=len(train_rows),vector_dim=space.vector_dim,num_clusters=space.num_clusters,
        checkpoint_selection="best development target soft CE across three planner epochs; final one-epoch prefix",
        test_used_for_training=False)
    evaluation_cache = {}
    for split in ("dev","test"):
        rows = read_rows(data/f"{split}.jsonl")
        predicted = planner.predict_rows(rows)
        actual = np.asarray([space.targets(r["target_meme_distribution"])[0] for r in rows])
        goldvectors = np.asarray([space.targets(r["target_meme_distribution"])[1] for r in rows])
        metrics = distribution_metrics(actual,predicted["target_distribution"])
        metrics["examples"] = len(rows)
        metrics["target_vector_mse"] = float(np.square(goldvectors-predicted["target_vector"]).mean())
        metrics["majority_cluster_accuracy"] = float((actual.argmax(1)==predicted["target_distribution"].argmax(1)).mean())
        metrics["train_prior_baseline"] = distribution_metrics(actual,np.broadcast_to(mean_target,actual.shape))
        result[split] = metrics
        evaluation_cache[split] = (rows,predicted["target_vector"])
    result["metric_notes"] = "Planner latent-region metrics are not generated-reply or meme-selection quality. Four annotators can disagree. No automatic score establishes conversational quality."
    del planner
    gc.collect()
    torch.cuda.empty_cache()
    if (root/"generator/manifest.json").exists():
        from .paper_generator import PaperGenerator
        generator = PaperGenerator(config)
        generator.model.eval()
        subset_size = int(config.get("evaluation_subset_per_split",512))
        rng = np.random.default_rng(config.get("seed",42))
        with torch.inference_mode():
            for split,(rows,predicted_vectors) in evaluation_cache.items():
                selected = sorted(rng.choice(len(rows),size=min(subset_size,len(rows)),replace=False).tolist())
                ce = {"gold_control":0.,"predicted_control":0.}
                tokens = 0
                for i in selected:
                    row = rows[i]
                    _,gold_vector = space.targets(row["target_meme_distribution"])
                    count = int(generator.response_ids(row["response"]).numel())
                    tokens += count
                    ce["gold_control"] += float(generator.loss(row,gold_vector))*count
                    ce["predicted_control"] += float(generator.loss(row,predicted_vectors[i]))*count
                result[split]["generator"] = dict(examples=len(selected),reply_eos_tokens=tokens,
                    token_ce={key:value/tokens for key,value in ce.items()},
                    selection="seeded random fixed subset; no checkpoint tuning",sample_ids=[rows[i]["sample_id"] for i in selected])
    write_json(root/"evaluation.json",result)
    from .paper_pipeline import finalize_run
    finalize_run(config)
    return result
