# Mistral 模型版本归档

本目录保存本项目两版 EmoStance 中文适配的 Mistral 推理归档。每版包含独立的运行源码快照、配置、情绪空间、训练回复索引和来源清单；训练代码与方法说明见 [`model/`](../model/README.md)。这些归档不是 EmoStance 作者发布的官方模型，训练损失或归档完整性也不代表聊天质量通过验收。

| 版本 | 数据版本 | Prefix 训练 | 情绪空间 | 版本说明 |
| --- | --- | --- | --- | --- |
| `v1.0.0/` | `data_spilt/v1.0.0` | 1 轮、15,134 次更新 | 256 维、10 个无名区域 | [README](v1.0.0/README.md) |
| `V2.0.0/` | `data_spilt/V2.0.0` | 1 轮、52,379 次更新 | 256 维、8 个无名区域 | [README](V2.0.0/README.md) |

两版均冻结 `Mistral-7B-Instruct-v0.3`，只训练回复生成器的连续 prefix 投影器；角色感知 DeBERTa planner 另行训练 3 轮，按验证集目标软标签 CE 选择检查点。目录名大小写保留原样，Linux 上不可混用。

每个版本的核心文件如下：

- `runtime_source/emostance_meme/`：保存时的图空间、规划器、prefix 生成器、回复检索和运行入口源码。
- `planner/`、`generator/`：模型配置和训练清单；对应的本地 `best.pt` 权重另行保存。
- `space.json`、`response_vectors.npz`、`response_anchors.jsonl`：情绪空间及训练回复库。检索返回最近训练回复关联的 `meme_id`，文字仍为模型新生成的回复。
- `version_manifest.json`、`source_manifest.json`、`provenance.json` 等：归档文件 SHA-256、训练来源和配置；历史评估结果见各版 `training_report.md`、`evaluation.json`。

历史 Git 资产已包含部分源码、配置、清单、空间和检索索引。**大型 planner/prefix 权重保留在本地，不上传到普通 Git；冻结 Mistral 基座及其 tokenizer 也不包含在归档内。** 新增的忽略规则不会删除历史提交已有资产。仅克隆仓库不能直接推理，需另外取得与版本清单匹配的权重和完整依赖，并准备本地基座。不要把 Git LFS 指针当作实际权重文件。

在项目根目录执行数据恢复：

```bash
python3 model/tools/restore_git_assets.py
```

恢复器按 [`model/git_upload_manifest.json`](../model/git_upload_manifest.json) 校验并解压已打包的数据，将训练文件复制到各版本的 `data/train.jsonl`，供回复索引来源核验。它会跳过内容已匹配的文件，拒绝覆盖冲突文件，并报告缺失权重；**不会下载、重建或恢复任何模型权重**。

准备好匹配权重、本地 Mistral 基座和 Python 环境后，可在项目根目录调用原归档入口：

```bash
# 使用已安装 model/pyproject.toml 所需依赖的环境。
export MEME_PYTHON=/path/to/venv/bin/python

# 单句推理；两版按需选择。
bash model_canshu/v1.0.0/run_chat.sh --text '你好'
bash model_canshu/V2.0.0/run_chat.sh --text '你好'

# 连续逐行输入；Ctrl+D 结束。
bash model_canshu/V2.0.0/run_chat.sh --text ''
```

脚本使用各版 `inference_config.json` 及本版源码，禁止在线下载。现有配置中的外部基座路径为 `/home/aida102/emoji/Mistral-7B-Instruct-v0.3`，迁移环境须准备相应路径或重新导出并核对配置与版本清单；严格服务加载器会拒绝与清单不一致的归档。已保存的 planner 使用本版 encoder 配置、tokenizer 和完整训练权重，推理无需重新训练；从头训练 planner 才需要原始 DeBERTa 基座。

原归档入口在采样时默认生成 4 个候选并按预测取向重排，不能用于证明当前服务的单次生成行为。需要 Mistral TCP 与树莓派服务时，使用独立的 [`model_information/`](../model_information/README.md) 入口；它有自己的会话窗口、通信协议和单次回复生成实现。Qwen 训练与服务位于 [`model_qwen/`](../model_qwen/README.md) 及相应推理目录，不在本归档内。
