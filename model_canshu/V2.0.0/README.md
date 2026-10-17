# 模型参数 V2.0.0

本版本保存训练后选中的 DeBERTa 与连续 prefix 参数、情绪空间、训练回复检索库、配置和运行源码快照。

- 训练来源：`/home/aida102/hks/model/runs/paper_v2`。
- 数据来源：`/home/aida102/hks/data_spilt/V2.0.0`；训练样本数为 52,379，数据划分记录见 `data_split_manifest.json`。
- DeBERTa：训练 3 轮，验证集选中第 2 轮；`planner/best.pt` 原样保存，含完整训练后的编码器、情绪原型和角色转移参数。
- Prefix：训练 1 轮、52,379 次更新；`generator/best.pt` 保留全部推理参数并去掉优化器和随机数状态，导出前后每个权重张量完全一致。
- 原始完整训练检查点仍保留在上述训练来源目录，继续训练请使用原目录。
- 冻结 Mistral 基座及其 tokenizer 继续使用 `/home/aida102/emoji/Mistral-7B-Instruct-v0.3`，本参数包不复制其大文件。
- `data/train.jsonl` 用于回复库的训练来源哈希验证；推理不会对该文件继续训练。
- `runtime_source/` 为保存时的代码快照，保证后续拉取仓库后仍能使用同一加载代码。
- `version_manifest.json` 记录每个归档文件的 SHA256 与来源，可核验版本完整性。

在项目根目录测试：

```bash
bash model_canshu/V2.0.0/run_chat.sh --text '你好'
```

逐行连续对话：

```bash
bash model_canshu/V2.0.0/run_chat.sh --text ''
```

输出为 `reply` 与 `meme_id`。本版本的验证与测试结果见 `training_report.md` 和 `evaluation.json`，回复与表情匹配效果仍需结合实际使用评估。
