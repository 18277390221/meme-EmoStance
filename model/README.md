# EmoStance 中文回复与 meme 选择

当前训练入口为 `paper_run.sh`、配置为 `paper_config.json`。使用本地 DeBERTa 训练带角色的情绪取向预测器，再冻结本地 Mistral，训练连续 prefix。推理返回新生成的 `reply`，并按该回复的情绪向量检索最近训练回复，返回其 `meme_id`。连续 10 秒无用户消息时清空会话。

完整方法、数据格式及操作见 [PAPER_TRAINING.md](PAPER_TRAINING.md)。原 `/home/aida102/emoji/EmoStance/` 只作参考，不修改其内容，也不下载模型权重。

## Git 数据恢复

克隆仓库后先在项目根目录运行 `python3 model/tools/restore_git_assets.py`。该工具按 [git_upload_manifest.json](git_upload_manifest.json) 校验并恢复超过 Git 单文件限制的 V2 数据包，以及两版归档的重复训练数据。压缩包位于 `model/git_assets/`，恢复后原数据路径和 SHA256 不变。

两版归档的 `planner/best.pt` 和 `generator/best.pt` 超过 100 MiB，未上传 Git，须从原服务器另行补齐；冻结 Mistral 基座也需本地准备。`model/runs/` 中的训练日志、优化器状态和临时结果不在仓库中。完整上传范围与大文件列表见项目根目录 README。

## 数据

- `dataset/v1.0.0/cleaned_dialogues.jsonl`：11,092 段完整对话，每轮保留四位标注者。
- `data_spilt/v1.0.0/dialogues/`：按整段对话约 80%/10%/10% 划分。
- `data_spilt/v1.0.0/{train,dev,test}.jsonl`：所有相邻轮切片，含完整已知历史。训练 15,134 条、验证 1,900 条、测试 1,869 条。
- `data_spilt/v1.0.0/annotation_rows/`：逐真实轮标签，训练情绪图时只读 train。

上列路径均相对项目根目录。重复及前缀关联的完整对话绑定同一集合；原始 `source_split` 作为来源另存。训练图覆盖 166 个 meme，Leiden 自动得到 10 个无名区域，向量为 256 维。

## 启动

在项目根目录执行：

```bash
# 已有检查点会恢复训练；完整训练结束后再次运行会复用结果。
bash model/paper_run.sh train
bash model/paper_run.sh index
bash model/paper_run.sh evaluate

# 单句输入，输出 reply + meme_id。
bash model/paper_run.sh chat --text '今天有点累，想休息一下。'

# 多轮终端交互。
bash model/paper_run.sh chat --text ''

# HTTP：POST /chat，JSON 包含 text 和可选 session_id。
bash model/paper_run.sh serve --host 127.0.0.1 --port 8790
```

默认使用 `/home/aida102/emoji/.venv/bin/python`。更换解释器可设置 `MEME_PYTHON`。训练权重、日志、统计和评估均在 `model/runs/paper_v1/`，本地大文件已由 `model/.gitignore` 排除。

## 实现

`paper_space.py` 建训练集情绪图；`paper_planner.py` 训练/加载 DeBERTa；`paper_generator.py` 训练/加载冻结 Mistral 的 prefix；`paper_engine.py` 重排生成候选并检索 meme；`paper_pipeline.py` 连接训练、评估和服务。`session.py`、`service.py`、`http_server.py` 管理会话和接口。

原先的冻结 Mistral 特征与谱聚类适配代码保留，说明移至 [LEGACY_ADAPTATION.md](LEGACY_ADAPTATION.md)；本轮训练使用上述论文方法入口。
