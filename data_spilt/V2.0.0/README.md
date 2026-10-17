# V2.0.0 数据划分

采用 V1.0.0 的清洗、文本归一化、四标注合并及划分规则；版本目录为 `data_spilt/V2.0.0/`。

原始数据包含 34,502 段对话、100,000 轮发言。剔除 27 段标注异常对话及 42 段原文逐轮完全一致的较短严格前缀，最终保留 34,433 段对话、99,745 轮发言及 398,980 条标注。

| 集合 | 对话数 | 发言轮数 | 相邻轮次样本数 |
| --- | ---: | ---: | ---: |
| 训练集 `train` | 27,547 | 79,926 | 52,379 |
| 验证集 `dev` | 3,443 | 9,842 | 6,399 |
| 测试集 `test` | 3,443 | 9,977 | 6,534 |
| 合计 | 34,433 | 99,745 | 65,312 |

按整段对话约 80%/10%/10% 划分，随机种子为 42。四位标注者始终跟随同一段对话；归一化后相同或存在完整轮次前缀关系的对话绑定到同一集合。原始 `source_split` 保存在 `original_source_split` 中用于追溯。

每段 N 轮对话生成 N−1 条相邻轮次样本，覆盖 A→B 与 B→A，样本历史保留截至源发言的完整前文。每轮表情分布按四位标注者的置信度求和归一化，可靠性权重为置信度总和除以 20。

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：模型训练与评估使用的相邻轮次样本。
- `dialogues/`：划分后的完整对话与四份标注。
- `annotation_rows/`：划分后的逐轮发言、四份标注、软标签与权重。
- [metadata/split_manifest.json](metadata/split_manifest.json)：划分统计、分组关系、防泄露检查与输出文件 SHA256。
- [metadata/processing_report.json](metadata/processing_report.json)：来源版本、清洗统计、备份路径、脚本版本与处理命令。
- [metadata/validation_report.json](metadata/validation_report.json)：最终数据的独立完整性核验。

清洗后的合并数据位于 [dataset/V2.0.0/cleaned_dialogues.jsonl](../../dataset/V2.0.0/cleaned_dialogues.jsonl)，清洗记录位于 [cleaning_report.json](../../dataset/V2.0.0/cleaning_report.json)，被剔除记录位于 [deleted_dialogues.jsonl](../../dataset/V2.0.0/deleted_dialogues.jsonl)。清洗前的六份原始文件已完整备份，具体路径见处理报告。原始压缩包与 V1 数据已核对 SHA256 并保持原样。

新版压缩包仅包含对话级标注，清洗脚本已兼容缺失旧逐轮标注文件的情况；仍按 V1 规则校验每段对话的四份完整标注。
