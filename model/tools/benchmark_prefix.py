#!/usr/bin/env python3
"""Measure one frozen-Mistral prefix forward/backward without optimizer or files.

Reads the real training split and fitted space. Selects the longest actually
tokenized training example after configured truncation. Does not alter config,
checkpoints, data, or the running training implementation. Results go to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def longest_training_example(rows, tokenizer, config):
    """Use exact training tokenization, including prefix and appended EOS."""
    from emostance_meme.paper_generator import generation_prompt
    max_prompt = int(config.get("max_prompt_tokens", 384))
    max_response = int(config.get("max_training_response_tokens", 128))
    prefix = int(config.get("prefix_length", 8))
    best = None
    best_lengths = None
    maximum = -1
    for row in rows:
        if row.get("split", row.get("source_split")) != "train":
            raise ValueError("Benchmark requires real train-only rows")
        prompt = generation_prompt(row["history"], row["next_role"], row.get("situation", ""))
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=max_prompt)["input_ids"]
        response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"][:max_response]
        lengths = {"prompt_tokens": len(prompt_ids), "response_content_tokens": len(response_ids),
                   "eos_tokens": 1, "prefix_tokens": prefix,
                   "sequence_tokens": len(prompt_ids) + len(response_ids) + 1 + prefix}
        if lengths["sequence_tokens"] > maximum:
            maximum = lengths["sequence_tokens"]
            best, best_lengths = row, lengths
    if best is None:
        raise ValueError("Training split is empty")
    return best, best_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "paper_config.json")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable GC for a comparison; default is disabled")
    parser.add_argument("--repeats", type=int, default=1, help="Repeated forward/backward measurements, without parameter updates")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    import torch
    from transformers import AutoTokenizer
    from emostance_meme.config import local_backbone
    from emostance_meme.paper_generator import PaperGenerator
    from emostance_meme.paper_pipeline import load_config, read_rows
    from emostance_meme.space import EmotionSpace

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = load_config(args.config)
    if not torch.cuda.is_available() or config.get("device", "cuda") == "cpu":
        raise RuntimeError("CUDA is required for this GPU memory benchmark")
    torch.manual_seed(int(config.get("seed", 42)))
    space = EmotionSpace.load(Path(config["run_dir"]) / "space.json")
    rows = read_rows(Path(config["data_dir"]) / "train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(str(local_backbone(config["base_model_path"])),
                                             local_files_only=True, trust_remote_code=False, use_fast=True)
    tokenizer.truncation_side = "left"
    selected, lengths = longest_training_example(rows, tokenizer, config)
    # Removing run_dir disables automatic checkpoint loading. No adapter or
    # optimizer state is needed to measure activation shapes and throughput.
    benchmark_config = {key: value for key, value in config.items() if key != "run_dir"}
    benchmark_config["vector_dim"] = space.vector_dim
    benchmark_config["gradient_checkpointing"] = args.gradient_checkpointing
    requested_device = config.get("device", "cuda")
    device = torch.device("cuda" if requested_device == "auto" else requested_device)
    free_before, total = torch.cuda.mem_get_info(device)
    print(json.dumps({"stage": "benchmark_selection", "sample_id": selected.get("sample_id"),
                      "train_rows_scanned": len(rows), **lengths,
                      "gradient_checkpointing": args.gradient_checkpointing,
                      "gpu_free_before_load_bytes": free_before, "gpu_total_bytes": total}), flush=True)
    generator = PaperGenerator(benchmark_config)
    generator.model.train()
    generator.projector.train()
    generator.model.config.use_cache = False
    if args.gradient_checkpointing:
        generator.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        generator.model.gradient_checkpointing_disable()
    _, vector = space.targets(selected["target_meme_distribution"])
    parameter_bytes = sum(p.numel() * p.element_size() for p in generator.projector.parameters())
    estimated_adam_states = 2 * parameter_bytes
    estimated_adam_foreach_workspace = max(p.numel() * p.element_size() for p in generator.projector.parameters())
    measurements = []
    for repeat in range(args.repeats):
        generator.projector.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before_allocated = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        loss = generator.loss(selected, vector)
        loss.backward()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p.grad.detach().float())
                                                        for p in generator.projector.parameters() if p.grad is not None]))
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError("Nonfinite loss/gradient or zero prefix gradient during benchmark")
        if any(p.grad is not None for p in generator.model.parameters()):
            raise RuntimeError("Frozen LM unexpectedly received parameter gradients")
        free_after, _ = torch.cuda.mem_get_info(device)
        external_used = max(0, total - free_after - torch.cuda.memory_reserved(device))
        measurement = {"repeat": repeat + 1, "seconds_forward_backward": elapsed,
                       "loss": float(loss.detach()), "prefix_gradient_norm": float(grad_norm),
                       "baseline_allocated_bytes": before_allocated, "peak_allocated_bytes": peak_allocated,
                       "peak_reserved_bytes": peak_reserved,
                       "activation_and_gradient_increment_bytes": peak_allocated - before_allocated,
                       "estimated_adam_extra_state_bytes": estimated_adam_states,
                       "estimated_adam_foreach_workspace_bytes": estimated_adam_foreach_workspace,
                       "estimated_external_gpu_used_bytes": external_used,
                       "estimated_training_peak_plus_external_bytes": peak_reserved + estimated_adam_states + estimated_adam_foreach_workspace + external_used,
                       "gpu_free_after_bytes": free_after}
        measurements.append(measurement)
        print(json.dumps({"stage": "benchmark_measurement", **measurement}), flush=True)
        del loss
    print(json.dumps({"status": "complete", "gradient_checkpointing": args.gradient_checkpointing,
                      "sample_id": selected.get("sample_id"), **lengths, "measurements": measurements,
                      "parameter_updates": 0, "checkpoints_written": False, "config_modified": False,
                      "optimizer_estimate_is_not_measured": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
