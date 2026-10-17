# EmoStance 方法训练与回复表情检索

本流程使用清洗后的四标注者中文对话，先按完整对话划分训练、验证、测试集，再生成包含全部历史的相邻轮次样本。模型根据历史规划回复情绪取向，用连续 prefix 控制本地 Mistral 生成新回复，再检索情绪向量最近的训练回复，返回新回复及对应的 `meme_id`。

方法参考 [EmoStance 论文](https://arxiv.org/abs/2609.02133)及[官方代码](https://github.com/18277390221/EmoStance)，许可保留于 [EmoStance.txt](THIRD_PARTY_LICENSES/EmoStance.txt)。这是中文 meme 数据适配：切分比例由用户指定，Leiden 区域数由中文数据重新发现，新增回复表情检索及会话管理。门控采用论文 C.5 的来源预测熵公式；官方发布代码的目标预测熵门控与论文不同。历史文本格式沿用官方代码，下一发言角色通过角色嵌入输入。

代码、训练产物和运行入口均在 `model/` 下。原 `/home/aida102/emoji/EmoStance/`、本地基座目录及 `dataset/v.1.0.0/data/` 作为只读输入保留；不修改原 EmoStance 文件，不另外下载基座模型。

## 数据文件与切分

清洗后的完整对话单独保存在 [cleaned_dialogues.jsonl](../dataset/v1.0.0/cleaned_dialogues.jsonl)，共 **11,092 段、29,995 轮、119,980 条标注**。每轮均有 `paper1` 至 `paper4` 四份表情包与置信度标注，包含原始 ID 和来源追溯；中文分词空格已合并。[导出清单](../dataset/v1.0.0/cleaned_dialogues.manifest.json) 保存统计与输入、输出文件 SHA256。

采用随机种子 `42`，按完整对话约 `8:1:1` 重新划分。同一段的全部轮次和四份标注进入同一集合；剩余的空白归一化重复及前缀相关对话绑定分组，避免跨集合泄漏。数据中的原 `train/dev/test` 仅作为 `original_source_split` 保留，新划分由 `split` 和 `source_split` 表示。

| 集合 | 完整对话数 | 原始发言轮次 | 相邻轮次切片数 |
|---|---:|---:|---:|
| train | 8,874 | 24,008 | 15,134 |
| dev | 1,109 | 3,009 | 1,900 |
| test | 1,109 | 2,978 | 1,869 |
| 合计 | 11,092 | 29,995 | 18,903 |

划分产物在 `data_spilt/v1.0.0/`：

- `train.jsonl`、`dev.jsonl`、`test.jsonl`：训练及评估使用的切片，带历史、回复、角色和上下文两端的软标签。
- `dialogues/`：三个集合的完整对话。
- `annotation_rows/`：三个集合的真实发言与四份标注，供情绪空间构建和审计。
- [metadata/split_manifest.json](../data_spilt/v1.0.0/metadata/split_manifest.json)：划分规则、数量、表情覆盖、分组关系和防泄漏检查结果。

切片遵循 EmoStance 的全部相邻轮次规则，包括 `A→B` 和 `B→A`。一段 `A1 → B1 → A2 → B2` 产生三个样本：

| 输入历史 | 目标回复 |
|---|---|
| `A1` | `B1` |
| `A1 → B1` | `A2` |
| `A1 → B1 → A2` | `B2` |

因此，`N` 轮完整对话生成 `N−1` 个样本。切片保留原始历史，编码时再按长度左侧截断，以保留最近发言。

每轮先将四份标注中相同 `meme_id` 的置信度相加，再除以四份置信度总和，得到 `meme_distribution`。例如两个标注各给某表情 `4` 分，另两个分别给其他表情 `3`、`3` 分，该表情的软标签概率为 `8/14`。原四份标注仍保留，不把四个标注者拆成四份独立对话。另保存平均置信度除以 `5` 的权重，供图构建与追溯；训练损失不在置信度软标签之外重复加权。

## 情绪空间与训练模型

图、词频统计、转移先验、类别权重及检索库均只由训练集构建。验证集用于取向规划器选优；测试集留作评估。

情绪空间按 EmoStance 的无名区域方法，以文本上下文相似度和标注混淆关系建立表情图，采用 Leiden 社区划分及边界软归属。当前实际得到 **166 个表情、256 维向量、10 个情绪取向区域**；区域为算法发现的编号，没有人为赋予“高兴”“悲伤”等类别名。见 [graph/manifest.json](runs/paper_v1/graph/manifest.json)。

原实现的英文词正则不能完整表示中文文本，因此本项目将文本特征适配为中文单字及相邻双字、英文词及相邻词的有符号哈希 TF-IDF。IDF 仅在训练集拟合。这里使用的是文本与标注产生的统计特征；不读取表情说明或标注理由作为模型输入。

取向规划器使用本地 `/home/aida102/emoji/deberta-v3-base`，学习上下文、当前与下一发言角色、来源取向，以及角色相关转移先验。转移先验通过来源预测熵控制的门加入目标预测，目标分布再投影到训练得到的区域原型向量。本轮设置为 **3 个 epoch**，学习率 `1.5e-5`，批量大小 `8`，最大输入长度 `320`；以验证集目标软标签交叉熵选择最佳检查点。

向量辅助损失沿用官方训练实现的逐维均方误差（mean MSE），与论文写作中的 L2 平方和存在维数因子差异；本轮不据此更改训练超参。应用检索使用软分布经训练原型重构的向量，避免依赖直接回归头的尺度。

该本地 DeBERTa 基座为英文模型，不能称为中文专用模型。训练集分词诊断中，未知 token 占 **6.304%**，主要来自中文标点，其次为“啥、哎、嗯”等汉字；81.333% 的样本至少含一个未知 token，这不意味着这些样本整体丢失。仅两条训练样本超过 320 tokens。详见 [tokenizer_diagnostics.json](runs/paper_v1/tokenizer_diagnostics.json)。本轮继续使用现有基座，实际中文取向效果以验证与测试结果为准。中文 TF-IDF 适配针对图空间特征，不会改变 DeBERTa 的词表。

回复生成器使用本地 `/home/aida102/emoji/Mistral-7B-Instruct-v0.3`，冻结全部基座权重，只训练 `256 → 4096 → 8×4096` 的两层 MLP，将目标情绪向量转为 8 个连续 prefix token。本轮设置为 **1 个 epoch**、批量大小 `1`、学习率 `1e-4`、CUDA BF16。初始启用梯度检查点；最长真实样本显存测试通过后，使用 `generator_runtime_gradient_checkpointing=false` 关闭重计算以加速，并将恢复检查点间隔调整为 2,000 步，保留此前优化器与训练进度。运行设置不改变模型或训练样本，最终阶段 manifest 记录实际设置。官方 `Dialogue:`、`A:`、`B:` 文本格式用于编码全部历史，历史最多保留最近 `384` token；目标回复内容最多 `128` token，随后追加 EOS。交叉熵仅监督回复及 EOS，历史与 prefix 位置均屏蔽。

生成器的 `best.pt` 按训练 epoch 的回复交叉熵保存；取向规划器的 `best.pt` 按验证损失选择，二者选优口径不同。检查点包含训练进度与优化器状态以便恢复；生成器检查点不复制 Mistral 基座权重。

## 训练与运行

以下命令均从项目根目录 `/home/aida102/hks` 执行。入口 [paper_run.sh](paper_run.sh) 默认使用 `/home/aida102/emoji/.venv/bin/python`；可以用环境变量 `MEME_PYTHON` 指定其他已安装依赖的 Python。配置见 [paper_config.json](paper_config.json)，产物目录为 `model/runs/paper_v1/`。

导出和切分可按以下命令复现：

```bash
python3 model/tools/export_cleaned_dialogues.py
python3 model/tools/build_paper_splits.py
```

完整训练入口依次构建图空间、训练规划器和训练回复 prefix。若同一运行目录已有可恢复检查点，入口从记录的位置继续；数据或关键配置改变时应更换 `run_dir`。

```bash
bash model/paper_run.sh train
```

也可分别运行各阶段：

```bash
bash model/paper_run.sh train --stage space
bash model/paper_run.sh train --stage planner
bash model/paper_run.sh train --stage generator
```

本轮已完成规划器 **3 个 epoch / 5,676 次更新**及生成 prefix **1 个 epoch / 15,134 次更新**。规划器验证集选中第 2 轮，目标软标签交叉熵 **1.977678**；prefix 的训练回复及 EOS 交叉熵按样本平均为 **2.613928**。规划器日志在 `runs/paper_v1/planner/metrics.jsonl`，生成器日志在 `runs/paper_v1/generator/progress.jsonl`；各阶段 `manifest.json` 均记录 `complete`。

训练完成后构建训练回复检索库并运行评估：

```bash
bash model/paper_run.sh index
bash model/paper_run.sh evaluate
```

最终产物和结果查看 [manifest.json](runs/paper_v1/manifest.json) 与 [evaluation.json](runs/paper_v1/evaluation.json)；训练、建库和评估完成后才会生成对应文件，不能将正在运行的中间损失当作测试成绩。评估包含规划器的定量指标，以及验证、测试子集上真实情绪向量控制和预测情绪向量控制下的回复交叉熵；这些指标不能单独保证开放生成回复的自然度或语义质量。

本轮测试集 1,869 条切片的取向软标签 CE 为 **1.959147**，训练频率先验基线为 **1.986888**，相对降低约 **1.40%**，收益有限。测试集固定随机 512 条参考回复的 token CE 为真实向量控制 **2.645259**、预测向量控制 **2.708820**。完整统计及限制见 [training_report.md](runs/paper_v1/training_report.md)。

最终真实模型验证通过：4 次请求共生成 16 个候选，保留历史的输入长度为 `1 → 3 → 1`，在恰好 10 秒空闲时重置；HTTP `/health`、`/chat`、`/reset` 均通过，`/chat` 仅返回 `reply` 与合法 `meme_id`。记录见 [final_smoke.json](runs/paper_v1/final_smoke.json)。116 项代码测试通过；这些检查证明实现链路可用，不能替代对回复和表情相关性的人工评价。

重新生成报告及运行真实模型接口检查：

```bash
python3 model/tools/summarize_paper_run.py
/home/aida102/emoji/.venv/bin/python model/tools/verify_paper_runtime.py
```

单句调用与连续终端会话：

```bash
bash model/paper_run.sh chat --text '今天有点累，想休息一下。'
bash model/paper_run.sh chat --text ''
```

连续会话中逐行输入，每行得到一个 `{"reply": "...", "meme_id": "..."}`。推理默认采样生成 4 个候选，通过规划器测量各候选的来源情绪分布，按与预测目标分布的软标签交叉熵由低到高重排，选出新回复；该步骤只使用模型预测，不使用真实目标标签。贪心解码时只生成一个候选。候选数及重排方法记录在最终 manifest 中，解码参数记录在运行配置中。

随后将生成回复追加到历史，用规划器的来源软分布乘训练来源原型，重构该回复的情绪向量。训练回复库也将每条真实回复追加到对应历史，通过同一规划器及同一来源原型编码，因此库与查询的计算路径完全一致，默认使用余弦距离。直接来源向量回归头仅作辅助训练及目标预测特征；工程审计发现其尺度偏离训练向量空间，因此不将其作为检索查询。审计只用三条人工输入及训练库，详见 [retrieval_vector_audit.json](runs/paper_v1/retrieval_vector_audit.json)；距离更近不能单独证明语义匹配更好。检索库包含训练集全部 **15,134** 个相邻目标回复，包括 A、B 两个方向，库中的 `meme_id` 取该真实回复四份标注置信度累计最高票；索引同时绑定规划器检查点的 SHA256。返回的 `reply` 保留 Mistral 新生成的文字，`meme_id` 来自最近训练回复，生成文字不会替换成检索到的旧回复。

部署解码额外设置 `no_repeat_ngram_size=4`，限制连续生成重复的四 token 片段。这是针对实测标点循环的应用解码设置，不参与训练或参考回复交叉熵计算。无名原型只有 10 个区域，表情相关性与多样性仍需实际使用评估。

启动 HTTP 接口：

```bash
bash model/paper_run.sh serve --host 127.0.0.1 --port 8790
```

调用示例：

```bash
curl http://127.0.0.1:8790/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"raspberry-pi-1","text":"今天有点累，想休息一下。"}'
```

同一 `session_id` 自动保存 `用户问题 → 助手回复 → 用户问题 → …` 的历史，用户每次只需发送一句话。从最后一次用户消息到达开始计算，连续 **10 秒**没有新消息则重置对话；不同 `session_id` 的历史独立。可调用 `POST /reset` 并提交 `session_id` 手动重置，`GET /health` 检查服务状态。
