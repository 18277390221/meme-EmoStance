#!/usr/bin/env python3
"""Archive selected EmoStance weights and a runnable, versioned inference bundle."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024),b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def archive_readme(source, destination, manifest):
    config = manifest["config"]
    planner, generator = manifest["planner"], manifest["generator"]
    return f'''# 模型参数 {Path(destination).name}

本版本保存训练后选中的 DeBERTa 与连续 prefix 参数、情绪空间、训练回复检索库、配置和运行源码快照。

- 训练来源：`{source}`。
- 数据来源：`{config['data_dir']}`；训练样本数为 {planner['train_examples']:,}，数据划分记录见 `data_split_manifest.json`。
- DeBERTa：训练 {planner['epochs']:,} 轮，验证集选中第 {planner['best_epoch']:,} 轮；`planner/best.pt` 原样保存，含完整训练后的编码器、情绪原型和角色转移参数。
- Prefix：训练 {generator['epochs_complete']:,} 轮、{generator['global_step']:,} 次更新；`generator/best.pt` 保留全部推理参数并去掉优化器和随机数状态，导出前后每个权重张量完全一致。
- 原始完整训练检查点仍保留在上述训练来源目录，继续训练请使用原目录。
- 冻结 Mistral 基座及其 tokenizer 继续使用 `{config['base_model_path']}`，本参数包不复制其大文件。
- `data/train.jsonl` 用于回复库的训练来源哈希验证；推理不会对该文件继续训练。
- `runtime_source/` 为保存时的代码快照，保证后续拉取仓库后仍能使用同一加载代码。
- `version_manifest.json` 记录每个归档文件的 SHA256 与来源，可核验版本完整性。

在项目根目录测试：

```bash
bash model_canshu/{Path(destination).name}/run_chat.sh --text '你好'
```

逐行连续对话：

```bash
bash model_canshu/{Path(destination).name}/run_chat.sh --text ''
```

输出为 `reply` 与 `meme_id`。本版本的验证与测试结果见 `training_report.md` 和 `evaluation.json`，回复与表情匹配效果仍需结合实际使用评估。
'''


def export(source,destination):
    import torch
    source,destination = Path(source).resolve(),Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Version already exists; refusing to overwrite: {destination}")
    if source==destination or source in destination.parents:
        raise ValueError("Archive must be separate from the training run")
    manifest=json.loads((source/"manifest.json").read_text())
    if manifest.get("status")!="complete":
        raise ValueError("Training run is not complete")
    for component in ("planner","generator"):
        if json.loads((source/component/"manifest.json").read_text()).get("status")!="complete":
            raise ValueError(f"{component} has not completed")
    config=dict(manifest["config"])
    data=Path(config["data_dir"])/"train.jsonl"
    index=json.loads((source/"response_index_manifest.json").read_text())
    for path,expected in ((data,index["train_file_sha256"]),
        (source/"planner/best.pt",index["planner_checkpoint_sha256"]),
        (source/"space.json",index["space_sha256"])):
        if digest(path)!=expected:
            raise ValueError(f"Training/index provenance mismatch: {path}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{destination.name}-export-",dir=destination.parent))
    source_hashes={}

    def copy_file(relative,target=None):
        src=source/relative
        dst=staging/(target or relative)
        dst.parent.mkdir(parents=True,exist_ok=True)
        expected=digest(src)
        shutil.copy2(src,dst)
        if digest(dst)!=expected:
            raise RuntimeError(f"Copy verification failed: {relative}")
        source_hashes[relative]=expected

    try:
        for component in ("planner/config.json","planner/best.pt","planner/emotion_space.json",
            "planner/manifest.json","generator/manifest.json","space.json","response_vectors.npz",
            "response_anchors.jsonl","response_index_manifest.json","evaluation.json","final_smoke.json",
            "verification_summary.json","tokenizer_diagnostics.json","training_report.md",
            "provenance.json","graph/manifest.json","graph/graph_arrays.npz"):
            copy_file(component)
        copy_file("manifest.json","source_manifest.json")
        for folder in ("planner/tokenizer","planner/encoder_config"):
            for path in sorted((source/folder).rglob("*")):
                if path.is_file():
                    copy_file(str(path.relative_to(source)))
        (staging/"generator").mkdir(exist_ok=True)
        original=source/"generator/best.pt"
        original_hash=digest(original)
        state=torch.load(original,map_location="cpu",weights_only=False)
        # Keep model identity/progress for the existing inference loader; resume
        # optimizer/RNG state remains in the untouched original training run.
        removed={"optimizer","torch_rng","cuda_rng"}
        inference_state={key:value for key,value in state.items() if key not in removed}
        inference_state["inference_only"]=True
        torch.save(inference_state,staging/"generator/best.pt")
        exported=torch.load(staging/"generator/best.pt",map_location="cpu",weights_only=False)
        if set(exported["projector"])!=set(state["projector"]):
            raise RuntimeError("Exported prefix tensor keys differ")
        if any(not torch.equal(value,exported["projector"][key]) for key,value in state["projector"].items()):
            raise RuntimeError("Exported prefix tensors differ")
        if exported["architecture"]!=state["architecture"] or any(key in exported for key in removed):
            raise RuntimeError("Prefix export metadata is invalid")
        source_hashes["generator/best.pt"]=original_hash
        del exported,inference_state,state
        if digest(original)!=original_hash:
            raise RuntimeError("Original prefix checkpoint changed during export")
        (staging/"data").mkdir()
        shutil.copy2(data,staging/"data/train.jsonl")
        if digest(staging/"data/train.jsonl")!=index["train_file_sha256"]:
            raise RuntimeError("Copied training-index validation file differs")
        split_manifest=Path(config["data_dir"])/"metadata/split_manifest.json"
        if split_manifest.exists():
            shutil.copy2(split_manifest,staging/"data_split_manifest.json")
            report=staging/"training_report.md"
            report.write_text(report.read_text(encoding="utf-8").replace(
                os.path.relpath(split_manifest,source),"data_split_manifest.json"),encoding="utf-8")
        shutil.copytree(PROJECT/"model/src/emostance_meme",staging/"runtime_source/emostance_meme",
            ignore=shutil.ignore_patterns("__pycache__","*.pyc","*.pyo"))
        shutil.copy2(PROJECT/"model/pyproject.toml",staging/"runtime_dependencies.toml")
        shutil.copytree(PROJECT/"model/THIRD_PARTY_LICENSES",staging/"THIRD_PARTY_LICENSES")
        runtime=dict(config,run_dir=".",data_dir="data")
        write_json(staging/"inference_config.json",runtime)
        archive_manifest=dict(manifest)
        archive_manifest["config"]=dict(config,run_dir=str(destination),data_dir=str(destination/"data"))
        archive_manifest["archive_version"]=destination.name
        archive_manifest["inference_only"]=True
        write_json(staging/"manifest.json",archive_manifest)
        script='''#!/usr/bin/env bash
set -euo pipefail
version_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$version_dir/runtime_source${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python_executable="${MEME_PYTHON:-/home/aida102/emoji/.venv/bin/python}"
exec "$python_executable" -m emostance_meme.paper_pipeline chat --config "$version_dir/inference_config.json" "$@"
'''
        (staging/"run_chat.sh").write_text(script,encoding="utf-8")
        (staging/"run_chat.sh").chmod(0o755)
        readme=archive_readme(source,destination,manifest)
        (staging/"README.md").write_text(readme,encoding="utf-8")
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=PROJECT,text=True).strip()
        files={str(p.relative_to(staging)):{"sha256":digest(p),"bytes":p.stat().st_size}
            for p in sorted(staging.rglob("*")) if p.is_file()}
        version_manifest=dict(version=destination.name,status="complete",
            saved_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),source_run=str(source),
            selected_planner_epoch=manifest["planner"]["best_epoch"],prefix_global_step=manifest["generator"]["global_step"],
            git_commit_before_pull=git_commit,inference_only=True,prefix_tensors_verified_equal=True,
            base_model_path=config["base_model_path"],base_model_weights_copied=False,
            environment={"python":sys.version.split()[0],**{name:version(name) for name in ("torch","transformers","numpy","igraph","leidenalg")}},
            source_sha256=source_hashes,files=files,total_bytes=sum(info["bytes"] for info in files.values()))
        write_json(staging/"version_manifest.json",version_manifest)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during export: {destination}")
        staging.rename(destination)
        print(json.dumps({"status":"complete","destination":str(destination),"files":len(files)+1,
            "bytes":version_manifest["total_bytes"],"prefix_tensors_verified_equal":True},ensure_ascii=False))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=PROJECT/"model/runs/paper_v1")
    parser.add_argument("--destination",type=Path,default=PROJECT/"model_canshu/v1.0.0")
    arguments=parser.parse_args()
    export(arguments.source,arguments.destination)
