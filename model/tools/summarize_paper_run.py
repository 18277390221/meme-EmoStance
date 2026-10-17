#!/usr/bin/env python3
"""Write a Chinese training report from completed run evidence, using stdlib.

No models are imported or loaded. Missing/incomplete evidence produces an
explicit incomplete-status report and exit code 2, never final metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


MODEL_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def number(value: Any, label: str, *, integer: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or nonfinite metric {label}")
    if integer:
        if value != int(value) or value < 0:
            raise ValueError(f"Invalid count {label}")
        return f"{int(value):,}"
    return f"{value:.6f}"


def last_progress(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    last = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently written trailing line is incomplete.
            if isinstance(value, dict):
                last = value
    return last


def _link(path: Path, root: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def baseline_comparison(model_ce: float, baseline_ce: float) -> str:
    number(model_ce, "test.soft_ce")
    number(baseline_ce, "test.baseline_ce")
    if min(model_ce, baseline_ce) < 0:
        raise ValueError("Cross entropy cannot be negative")
    if model_ce == baseline_ce:
        comparison = "测试软标签 CE 与训练频率先验基线持平。"
    elif baseline_ce == 0:
        comparison = "测试软标签 CE 高于训练频率先验基线，该指标表现变差；基线为零，无法计算相对百分比变化。"
    else:
        change = (model_ce - baseline_ce) / baseline_ce * 100
        if change < 0:
            comparison = f"测试软标签 CE 相对训练频率先验基线降低约 **{abs(change):.2f}%**。"
        else:
            comparison = f"测试软标签 CE 相对训练频率先验基线上升约 **{change:.2f}%**，该指标表现变差。"
    return comparison + "这些指标不能单独证明生成回复与表情匹配的质量。"


def build_report(run_dir: Path, split_manifest: Path | None = None) -> tuple[str, bool]:
    run_dir = Path(run_dir).resolve()
    manifest = read_json(run_dir / "manifest.json")
    evaluation = read_json(run_dir / "evaluation.json")
    config = (manifest or {}).get("config", {})
    if split_manifest is None:
        data = Path(config.get("data_dir", MODEL_ROOT.parent / "data_spilt/v1.0.0"))
        split_manifest = data / "metadata/split_manifest.json"
    split_manifest = Path(split_manifest).resolve()
    splits = read_json(split_manifest)
    planner = (manifest or {}).get("planner") or read_json(run_dir / "planner/manifest.json")
    generator = (manifest or {}).get("generator") or read_json(run_dir / "generator/manifest.json")
    issues = []
    if not manifest or manifest.get("status") != "complete":
        issues.append("总运行清单尚未记录 status=complete")
    for name, stage in (("取向规划器", planner), ("回复生成器", generator)):
        if not stage or stage.get("status") != "complete":
            issues.append(f"{name}尚未记录训练完成")
    if evaluation is None:
        issues.append("最终 evaluation.json 尚未生成")
    if splits is None:
        issues.append("缺少数据切分清单")
    if issues:
        lines = ["# EmoStance 训练状态", "", "**训练与评估尚未完整完成，暂不输出正式结果。**", ""]
        lines += [f"- {issue}。" for issue in issues]
        lines += ["", "以下为最近记录的进度，不能作为最终测试成绩。", ""]
        for label, stage, path in (("规划器", planner, run_dir / "planner/metrics.jsonl"),
                                   ("生成器", generator, run_dir / "generator/progress.jsonl")):
            record = last_progress(path)
            status = (stage or {}).get("status", "未完成或未生成清单")
            if record:
                step = record.get("global_step", record.get("step", "未记录"))
                epoch = record.get("epoch", "未记录")
                lines.append(f"- {label}：{status}；最近记录 epoch={epoch}，step={step}。")
            else:
                lines.append(f"- {label}：{status}；尚无进度记录。")
        lines += ["", "完成后重新运行本工具生成正式报告。", ""]
        return "\n".join(lines), False

    # A complete flag alone does not justify fabricating absent or inconsistent
    # results. Validate the source counts and essential actual training fields.
    counts = splits["splits"]
    train_count = counts["train"]["adjacent_samples"]
    planner_epochs = planner["epochs"]
    generator_epochs = generator["epochs_complete"]
    if generator_epochs != generator["epochs_requested"]:
        raise ValueError("Generator complete flag contradicts unfinished epochs")
    for stage_name, stage in (("planner", planner), ("generator", generator)):
        if stage.get("train_examples") != train_count:
            raise ValueError(f"{stage_name} training count differs from split manifest")
    if generator["global_step"] != train_count * generator_epochs:
        raise ValueError("Generator update count differs from completed epochs")
    if evaluation["train_examples"] != train_count:
        raise ValueError("Evaluation training count differs from split manifest")
    for split in ("dev", "test"):
        if evaluation[split]["examples"] != counts[split]["adjacent_samples"]:
            raise ValueError(f"{split} evaluation count differs from split manifest")
        subset = evaluation[split].get("generator")
        if not subset or not 0 < subset.get("examples", 0) <= evaluation[split]["examples"]:
            raise ValueError(f"Missing or invalid {split} generator subset evaluation")
    graph = manifest["graph"]
    inference = manifest.get("inference", {})
    candidates = inference.get("num_candidates", config.get("num_candidates", 4))
    if config.get("do_sample") is False:
        candidates = 1
    timeout = inference.get("idle_timeout_seconds", config.get("idle_timeout", 10))
    index_count = manifest["index"]["responses"]
    if index_count != train_count:
        raise ValueError("Response index count differs from training responses")

    lines = ["# EmoStance 中文对话训练结果", "",
             "**取向规划器、连续 prefix 生成器、训练回复检索库及评估均已完成。**",
             "结果由本运行目录的完成清单、评估文件和数据切分清单生成；不使用中间训练日志推断测试成绩。", "",
             "## 数据与实际训练", "",
             "| 集合 | 完整对话 | 原始发言轮次 | 相邻轮次切片 |", "|---|---:|---:|---:|"]
    for split in ("train", "dev", "test"):
        record = counts[split]
        lines.append(f"| {split} | {number(record['dialogues'], split+'.dialogues', integer=True)} | "
                     f"{number(record['real_turns'], split+'.real_turns', integer=True)} | "
                     f"{number(record['adjacent_samples'], split+'.adjacent_samples', integer=True)} |")
    total_dialogues = sum(counts[split]["dialogues"] for split in ("train", "dev", "test"))
    total_turns = sum(counts[split]["real_turns"] for split in ("train", "dev", "test"))
    lines += ["", f"合计 {number(total_dialogues, 'total_dialogues', integer=True)} 段完整对话、"
              f"{number(total_turns, 'total_turns', integer=True)} 轮。按完整对话约 8:1:1 划分，随机种子为 {manifest.get('seed', splits.get('seed'))}；"
              "四份标注及重复、前缀相关对话绑定同一集合，先划分再切片。",
              "每段 N 轮对话产生 N−1 个相邻样本，包含 A→B、B→A 两个方向及截至来源轮的全部历史。"
              "原始来源集合和原对话 ID 保留用于追溯。四份置信度按 meme_id 累加、归一化成软标签。", "",
              "| 训练组件 | 实际 epoch | 实际更新步数 | 最佳检查点口径 |", "|---|---:|---:|---|",
              f"| DeBERTa 取向规划器 | {number(planner_epochs, 'planner.epochs', integer=True)} | "
              f"{number(planner['global_steps'], 'planner.global_steps', integer=True)} | 验证目标软标签交叉熵 |",
              f"| 冻结 Mistral 连续 prefix | {number(generator_epochs, 'generator.epochs_complete', integer=True)} | "
              f"{number(generator['global_step'], 'generator.global_step', integer=True)} | 训练 epoch 回复及 EOS 交叉熵 |", "",
              f"规划器最佳 epoch：{number(planner['best_epoch'], 'planner.best_epoch', integer=True)}；"
              f"最佳验证目标软标签 CE：**{number(planner['best_dev_target_soft_ce'], 'planner.best_dev_target_soft_ce')}**。", "",
              f"训练图实际发现 {number(graph['num_clusters'], 'graph.num_clusters', integer=True)} 个无名情绪取向区域，"
              f"向量维度 {number(graph['vector_dim'], 'graph.vector_dim', integer=True)}，覆盖 "
              f"{number(graph['meme_count'], 'graph.meme_count', integer=True)} 个表情。区域编号不等于人工命名的情绪类别。", "",
              "## 验证与测试结果", "",
              "| 取向规划指标 | dev | test |", "|---|---:|---:|"]
    for label, key in (("目标软标签 CE ↓", "soft_ce"), ("JSD ↓", "jsd"),
                       ("软分布重叠度 ↑", "overlap"), ("目标向量 MSE ↓", "target_vector_mse")):
        lines.append(f"| {label} | {number(evaluation['dev'][key], 'dev.'+key)} | {number(evaluation['test'][key], 'test.'+key)} |")
    lines.append(f"| 训练频率先验基线 CE ↓ | {number(evaluation['dev']['train_prior_baseline']['soft_ce'],'dev.baseline_ce')} | {number(evaluation['test']['train_prior_baseline']['soft_ce'],'test.baseline_ce')} |")
    baseline = evaluation['test']['train_prior_baseline']['soft_ce']
    lines += ["", baseline_comparison(evaluation['test']['soft_ce'], baseline)]
    lines += ["", "生成器在每个集合的固定随机子集上计算按回复及 EOS token 数加权的交叉熵，未以测试结果选择检查点。", "",
              "| 集合 | 子集样本数 | 回复及 EOS token 数 | 真实向量控制 token CE ↓ | 预测向量控制 token CE ↓ |",
              "|---|---:|---:|---:|---:|"]
    for split in ("dev", "test"):
        subset = evaluation[split]["generator"]
        lines.append(f"| {split} | {number(subset['examples'], split+'.generator.examples', integer=True)} | "
                     f"{number(subset['reply_eos_tokens'], split+'.generator.reply_eos_tokens', integer=True)} | "
                     f"{number(subset['token_ce']['gold_control'], split+'.gold_control')} | "
                     f"{number(subset['token_ce']['predicted_control'], split+'.predicted_control')} |")
    lines += ["", "真实控制向量由标注软标签投影得到；预测控制向量由规划器根据历史输出。"
              "上述 CE、JSD、重叠度和 MSE 衡量取向预测或参考回复的条件拟合，不能单独证明开放生成回复的自然度、相关性或对话质量。", "",
              "## 推理与适配范围", "",
              f"推理按实际配置生成 {number(candidates, 'num_candidates', integer=True)} 个候选，"
              "以预测目标取向分布对候选实现取向分布的软交叉熵最小值重排，保留新生成文字。"
              "将生成回复追加历史，使用来源软分布乘训练来源原型重构回复向量，再检索情绪向量最近的训练回复。",
              f"检索库只含 {number(index_count, 'index.responses', integer=True)} 个训练目标回复，覆盖 A、B 两个方向；"
              "库和生成回复均追加到对应历史，使用同一来源软分布及训练原型编码。返回 `reply` 与最近训练回复对应的 `meme_id`，"
              "该表情取四份标注置信度累计最高票，同票按表情 ID 稳定选择。", "",
              f"按 session_id 保存用户与助手历史，从最后一次用户消息到达开始，连续 {timeout:g} 秒没有新问题则重置对话。", "",
              "图空间将原英文词正则适配为中文单字及相邻双字、英文词及相邻词的哈希 TF-IDF；"
              "IDF、图、转移先验及类别统计只在训练集拟合。",
              "本地 DeBERTa 为英文基座，部分中文标点及少数汉字仍可能映射为 [UNK]，不能称为中文专用模型；"
              "中文图特征适配不会改变该词表。Mistral 全部基座权重冻结，仅训练连续 prefix 投影器，原 EmoStance 目录保持只读。", ""]
    runtime_gc = generator.get("gradient_checkpointing_enabled", config.get("generator_runtime_gradient_checkpointing"))
    if runtime_gc is not None:
        lines += [f"生成器最后运行阶段的梯度检查点设置：`{str(bool(runtime_gc)).lower()}`。"
                  "这是执行内存与重算设置，不改变数据、损失或优化目标。", ""]
    smoke = manifest.get("final_smoke",{})
    checks = manifest.get("verification_summary",{})
    if smoke.get("status") == "passed" and checks.get("status") == "passed":
        lines += ["## 实现验证", "",
            f"{checks['unit_tests']} 项代码测试通过。最终检查点实际生成 {smoke['total_real_generation_candidates']} 个候选；"
            f"上下文输入长度为 {smoke['actual_history_lengths']}，恰好空闲 10 秒时重置。HTTP health/chat/reset 均通过，服务仅返回 reply 与合法 meme_id。",
            "实际输出保存在 [final_smoke.json](final_smoke.json)。这些工程检查不替代回复与表情匹配的人工评估。", ""]
    lines += ["## 结果来源", "",
              "- [完整运行清单](manifest.json)", "- [完整评估及子集 sample_id](evaluation.json)",
              f"- [数据切分与泄漏审计]({_link(split_manifest, run_dir)})",
              "- [规划器检查点清单](planner/manifest.json)", "- [生成器检查点清单](generator/manifest.json)", ""]
    return "\n".join(lines), True


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=MODEL_ROOT / "runs/paper_v1")
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    try:
        report, complete = build_report(args.run_dir, args.split_manifest)
        output = args.run_dir.resolve() / "training_report.md"
        write_report(output, report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot produce a verified report: {exc}\n")
    print(json.dumps({"status": "complete" if complete else "incomplete", "report": str(output)}, ensure_ascii=False))
    if not complete:
        parser.exit(2, "Training/evaluation is incomplete; only a status report was written.\n")


if __name__ == "__main__":
    main()
