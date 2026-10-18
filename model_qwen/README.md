# Qwen3-8B + EmoStance（V2.0.0）

训练代码、下载基座、配置、日志和训练产物均位于 `model_qwen/`；最终推理位于 [`model_information_qwen`](../model_information_qwen/README.md)。数据读取 `data_spilt/V2.0.0`，不重新划分，不改写旧模型目录。

## V3.0.0：仅按 loss 训练与选优

模型版本 **V3.0.0** 的独立运行名为 `qwen3_8b_v3_loss_ce`，配置为 [`paper_config_v3_loss.json`](paper_config_v3_loss.json)。模型版本与数据版本分别记录：训练数据仍是 **V2.0.0**。旧 V2 训练目录、质量报告和默认推理配置均保留。

本版冻结同一 Qwen3-8B 基座，重新初始化 EmoStance prefix，完整训练 3 个 epoch，共 157,137 次更新，学习率 `3e-5`。所有 52,379 条训练切片均计算真实参考回复及 EOS 的 CE；关闭 teacher KL、语言保持样例和基于角色/回复长度的参考过滤。情绪规划器复用已完成的 V2 规划器；[`tools/prepare_loss_run.py`](tools/prepare_loss_run.py) 在复制前重新核对训练/验证数据、空间与规划器超参数签名，不复制旧 prefix、优化器或基座。

每 1,000 步及每轮结束，使用固定的 512 条开发集样本，以规划器预测的控制向量计算按回复 token 加权的 CE，只按最低开发集 CE 保存 `generator/best.pt`。`generator/last.pt` 用于断点续训，`generator/loss_metrics.jsonl` 记录选优；测试集只在训练完成后作 loss 评估。训练和选优均不生成聊天回复、不使用上一版聊天评分或独立聊天验收集。

本机这版 V3 先用 `tools/prepare_loss_run.py` 复制并核验已完成的 V2 planner、空间和图，再运行工作流。仅设置配置中的 `planner_reuse_source` 不会自动复制这些产物。首次复用时，先准备好该字段指向的源运行目录，且目标 `run_dir` 必须不存在；已有 V3 运行目录直接执行 `workflow` 恢复，不重复执行准备命令。脚本会验证源 planner 完成状态、训练/验证数据和规划器配置签名，不复制旧 prefix 或优化器。

```bash
# 仅新运行目录首次准备；先在配置中填写实际 V2 源目录和新的 run_dir。
# 使用已安装本包依赖的 Python 环境；已有目标目录会明确拒绝。
python model_qwen/tools/prepare_loss_run.py --config model_qwen/paper_config_v3_loss.json

# 完整后台流程使用该版本专属配置；中断后同一命令从 last.pt 续训。
bash model_qwen/run.sh workflow --config model_qwen/paper_config_v3_loss.json

# 查看 V3 进度，避免把旧 V2 的完成状态当成 V3。
bash model_qwen/run.sh status --config model_qwen/paper_config_v3_loss.json
tail -f model_qwen/runs/qwen3_8b_v3_loss_ce/workflow.log
```

产物位于 `runs/qwen3_8b_v3_loss_ce/`，包含模型/数据版本、规划器复用证明、固定开发子集、最佳 loss 选择依据和开发/测试 CE。流程完成时标记 `training_status=complete`、`status=ready_for_review`、`quality_status=not_evaluated`，不会自动替换 `inference_config.json` 或启动推理服务。下面关于 KL 和聊天选优的说明对应保留的 V2 质量保护版本。

工作流的 planner 阶段会调用图空间准备逻辑：如果目标目录没有空间或已完成的 planner，它会从已准备的数据构图并训练规划器，不能将这种新训练称为复用原 V2 权重。无论选择哪条路径，都需事先准备本地 DeBERTa、Qwen 基座及数据；loss 工作流不自动下载基座。

## 模型与方法

下载官方 [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)，固定 commit `b968826d9c46dd6066d109eabc6255188de91218`，五个原始 BF16 safetensors 分片约 16.4 GB。下载器校验官方文件大小、SHA256/Git blob SHA、模型类型、分片索引和 tensor 边界；支持续传。基座位于 `backbones/Qwen3-8B/`，下载证明位于该目录的 `download_manifest.json`。重复执行保持完成清单不变。

沿用本项目对 [EmoStance](https://github.com/18277390221/EmoStance) 的中文数据适配，并保留[许可证](THIRD_PARTY_LICENSES/EmoStance.txt)：

1. 仅由训练集构建 256 维中文 TF-IDF、表情混淆图、Leiden 情绪区域和软归属。
2. 角色感知情绪规划器按 3 个 epoch、batch 8 训练，以验证集目标软标签 CE 选优。当前质量保护版本复用已经完成的同一 V2 规划器（最佳第 2 轮、验证 CE 1.71698），记录 `planner_reuse_source`；不重复训练它，也不复用旧 prefix。
3. 冻结 Qwen3-8B 全部参数，重新初始化 `256 → 4096 → 8×4096` 的连续 prefix 投影器，1 个 epoch、batch 1，学习率降为 `3e-5`。目标为相容参考回复 CE 加 `KL(无 prefix 基座 || 有 prefix 模型)`，KL 权重为 1；teacher 和 student 共用同一冻结基座，不加载第二份 8B。对齐回复及 EOS 的预测位置，历史和 prefix 不参与 CE/KL。
4. 训练与推理统一使用 Qwen3 原生聊天模板，显式 `enable_thinking=False`。训练 `B→A` 时把待预测的 A 映射为 assistant，保留两个回复方向。规划器仍使用原 EmoStance 的 `Dialogue:/A:/B:` 序列化。
5. 每轮只生成一条回复，固定 `num_return_sequences=1`、`num_beams=1`；无候选池或重排。对实际返回的回复计算来源情绪向量，在训练回复库中检索对应 `meme_id`。

训练数据有 52,379 个切片，验证 6,399 个，测试 6,534 个。图、频率、转移先验和检索库只用训练集；测试不参与训练或检查点选择。英文 DeBERTa 基座和少量无名情绪区域的限制仍然存在，评估指标不能替代人工回复质量检查。

## 保留语言能力与聊天选优

冻结基座权重仍允许 prefix 改变模型的实际行为，训练回复 loss 不能证明理解、角色遵守或流畅度提高。本版本采用训练约束、真实聊天选优和发布门槛三项措施：

- 全部 V2 训练切片保留并计算基座 teacher KL。超过角色 40 字限制的回复，以及保守规则识别的明显身份冲突回复，不施加硬标签 CE，减少语料和角色设定互相冲突。实际统计写入 `generator/training_target_policy.json`；身份规则是启发式，统计不代表人工语义标注。
- 每 8 次更新增加一条独立的[语言保持训练提示](evaluation/language_preservation_train.jsonl)，施加准确参考回复 CE 和 teacher KL；20 条提示涵盖事实跟踪、否定、更正、算术、身份和不虚构动作，与聊天 dev/holdout 的用户话语不重合。
- 第 100、500、1,000 步，此后每 2,000 步及 epoch 结束，用[聊天开发集](evaluation/chat_dev.jsonl)的 16 例、2 个固定 seed 实际生成回复，与相同提示词、相同采样设置下的无 prefix 基座比较。包含上下文、否定、身份、情绪、虚构行为、系统提示保护、简单推理和中文表达。
- 除受控事实规则外，使用同一冻结的无 prefix Qwen 做离线语义评分，分别评估理解、角色、语言自然性和真实性，保存全部历史、实际回复、评审原文及解析结果。解析失败按不合格处理；离线评审不会成为线上候选生成或重排。
- 仅接纳硬错误为零，且总分、每类得分及四项语义维度都不低于无 prefix 基座的检查点。主要按聊天得分选 `best.pt`，平分时按固定 128 条 dev 参考回复 CE 选优；训练 loss 仅用于诊断。周期评估恢复模型模式及 Python/NumPy/Torch/CUDA RNG。
- 选优结束后在独立[聊天验收集](evaluation/chat_holdout.jsonl)进行一次验收。报告绑定 prefix、规划器、空间、提示词、套件及生成设置哈希。普通自动发布必须通过 dev 与 holdout；`verify=False` 不能绕过门槛。用户获知质量失败后明确要求选用完整训练版本时，使用下述独立的显式选择记录，保留真实质量状态。

这是对已定义场景的回归保护，不是对所有语言能力的保证。同基座自动评审存在偏差。首次真实无 prefix 基线已暴露角色改名与虚构共同经历，训练约束和验收也针对这些问题；完整原始聊天记录保留供人工审阅；不能把自动通过等同于全面理解能力提高。独立验收失败后，不能反复用同一验收集调参。

## 执行

以下命令从项目根目录运行。当前机器直接使用已有的 `/home/aida102/emoji/.venv/bin/python`，支持 CUDA、Torch 和 Transformers；可通过 `QWEN_PYTHON` 指定其他环境（依赖见 `pyproject.toml`）。不调用外部模型 API。

```bash
# 下载（可续传）
bash model_qwen/run.sh download

# 全流程：规划器复用 → 权重校验 → 蒸馏单步验证 → prefix训练及周期聊天选优
#       → 独立聊天验收 → 训练回复建库 → 验证/测试评估 → 真实无候选推理验证
bash model_qwen/run.sh workflow

# 只读查看当前进度
bash model_qwen/run.sh status
tail -f model_qwen/runs/qwen3_8b_v2_quality/workflow.log
```

当前配置见 [`paper_config.json`](paper_config.json)。新训练写入 `runs/qwen3_8b_v2_quality/`，旧 CE 训练的检查点保留在 `runs/qwen3_8b_v2/`，原配置保留为 `paper_config_before_quality.json`。后台流程以 `workflow_status.json` 的 `pid/status/stage` 和实际阶段文件为准；完成单步验证不表示全量训练或聊天验收完成。中断后重新执行 `workflow` 会恢复优化器及训练位置；改变数据或核心训练配置应选择新的 `run_dir`。需要在更新边界安全停止时，可创建 `generator/STOP_REQUESTED` 文件；移除该文件后才可继续。

各阶段也可单独执行：

```bash
bash model_qwen/run.sh train --stage space
bash model_qwen/run.sh train --stage planner
bash model_qwen/run.sh smoke
bash model_qwen/run.sh train --stage generator
bash model_qwen/run.sh quality
bash model_qwen/run.sh index
bash model_qwen/run.sh evaluate
```

当前 `smoke` 的独立产物位于 `runs/smoke_qwen3_v2_quality/`，只证明真实基座上 CE/KL 和 prefix 梯度可用，绝不替代聊天验收。主模型 `manifest.json` 的 `status=complete` 表示选定的发布路径、索引和评估已完成；聊天质量单独记录为 `quality_status`，显式选择不会将 `rejected` 改为 `passed`。

## 产物与最终推理

`planner/best.pt` 是验证集选中的情绪规划器；普通训练路径的 `generator/best.pt` 是通过聊天 dev 的 prefix，不包含基座权重。`generator/last.pt` 保存最新训练位置及优化器。`quality_baseline_dev.json`、`quality_step_*.json`、`quality_dev.json` 和 `quality_holdout.json` 保留基线、周期比较、选中结果和独立验收的全部聊天记录；`quality_metrics.jsonl` 记录选优过程。原训练路径没有合格检查点时不会生成 `best.pt`。

本轮已完成 52,379 步、1/1 轮 prefix 训练，原路径 `runs/qwen3_8b_v2_quality/` 的 30 次自动聊天选优均未通过。用户在获知结果后要求选择充分训练版本作推理，因此另建 `runs/qwen3_8b_v2_selected/`：使用唯一完整训练的最终 prefix，去掉优化器和随机状态后保存为该目录的 `generator/best.pt`，逐项验证参数与原 `last.pt` 完全一致。历史最高分的中途版本只有报告、没有保存权重，不能据此加载一个不存在的模型。

默认推理配置为本机 `inference_config.json`，公开字段示例见 [`inference_config.example.json`](inference_config.example.json)，训练配置仍为 `paper_config.json`。实际模型运行目录、配置和发布记录不随核心源码上传，示例不能替代有效发布检查。显式选择记录 `deployment_selection.json` 绑定用户原始指令、来源完整训练检查点、当前前缀、规划器、空间、配置和真实 dev/holdout 报告。导出的精确参数重新运行开发集，并在独立留出集评估一次；行为失败如实保留，缺失报告、评估技术异常、哈希或配置不一致仍拒绝加载。`release_mode=user_selected_with_known_quality_failures` 表示按用户要求选用，不能解释为原聊天门槛已通过。

所选版本已完成全量训练回复建库、开发/测试指标、真实三轮 TCP 联调与默认 CLI 推理；205 项代码测试通过。TCP 实测每轮调用一次生成，历史长度为 `[1,3,1]`，验证了连续对话及 10 秒空闲重置。导出参数重新评估的聊天开发集得分为 0.94308（基座 0.93345），独立留出集为 0.92104（基座 0.92936），质量状态仍为 `rejected`。详见所选运行的 `verification_summary.json`、`generator/quality_dev.json`、`generator/quality_holdout.json` 与 `final_smoke.json`。

`response_vectors.npz` 和 `response_anchors.jsonl` 保存训练回复索引；`evaluation.json` 保存开发/测试规划器指标和固定随机 512 条/集合的参考回复 token CE；`final_smoke.json` 记录真实多轮生成与每轮单次调用验证。普通流程完成标志为 `workflow_status.json.status == "complete"`；本次显式选择流程由 `runs/qwen3_8b_v2_selected/deployment_status.json` 单独记录。

训练完成后：

```bash
bash model_information_qwen/run.sh chat
bash model_information_qwen/start_server.sh
```

协议、设备配置和启动参数见 [`model_information_qwen/README.md`](../model_information_qwen/README.md)。推理返回 `{"reply":"...","meme_id":"..."}`，保留多轮会话和 10 秒空闲重置。

代码验证：

```bash
/home/aida102/emoji/.venv/bin/python -m unittest discover -s model_qwen/tests -v
/home/aida102/emoji/.venv/bin/python -m unittest discover -s model_information_qwen/tests -v
```
