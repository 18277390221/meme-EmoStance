# meme-EmoStance

基于 EmoStance 方法的中文 meme 猫对话项目：树莓派负责离线语音识别、触屏和素材播放，服务器负责情绪取向规划、语言模型回复及表情匹配。支持 Mistral、Qwen3-8B 两种基座，以及独立的本地点歌、语音待机与唤醒。

仓库提供核心源码、启动脚本、配置示例和测试。**模型权重不随源码发布，克隆后需另外准备匹配的模型产物与本机配置。** 历史提交已包含部分整理数据、检索索引和归档元数据，本次源码更新不新增训练数据、模型权重、运行日志或本机发布记录。

## 工作流程

```text
树莓派麦克风 → 本地离线 ASR → 完整文字请求 → 认证 TCP 连接
                                                    ↓
                        历史上下文 → EmoStance 情绪取向规划
                                                    ↓
                               连续 prefix + 冻结的语言模型
                                                    ↓
                      生成一条回复 → 编码回复 → 匹配 meme_id
                                                    ↓
树莓派字幕、GIF、配套 WAV ← reply + meme_id + 对话/背景版本
```

三个 `model_information*` 服务都只生成一次，没有候选回复池或候选重排。表情检索决定 `meme_id`，不会用检索库中的句子替换生成的文字。声音来自素材已有的配套 WAV，**不是把任意生成回复实时转成语音的 TTS**。旧 `model/`、`model_canshu/` 的归档实验入口另有自己的生成设置，不能据此认定所有入口都已关闭候选重排。

## 目录与版本

| 目录 | 用途 |
| --- | --- |
| [`communication/`](communication/README.md) | TCP JSON 帧、共享口令、客户端与通信检查。 |
| `dataset/`、`data_spilt/` | 清洗数据及既定 train/dev/test 划分；`data_spilt` 保留原目录拼写。 |
| [`model/`](model/README.md) | Mistral 训练、EmoStance 方法适配、数据恢复工具。 |
| `model_canshu/v1.0.0/`、`model_canshu/V2.0.0/` | Mistral 版本归档、运行源码、清单及必要索引；权重单独保存。 |
| [`model_qwen/`](model_qwen/README.md) | Qwen3-8B 下载、训练、评估、检查点选择及共享推理后端。 |
| [`model_information/`](model_information/README.md) | Mistral V2.0.0 推理服务。 |
| [`model_information_qwen/`](model_information_qwen/README.md) | Qwen V2 选定版本服务，默认运行名 `qwen3_8b_v2_selected`。 |
| [`model_information_qwen_3.0/`](model_information_qwen_3.0/README.md) | Qwen V3.0.0 服务，默认运行名 `qwen3_8b_v3_loss_ce`。 |
| [`meme_software/`](meme_software/README.md) | 树莓派播放器、离线 ASR、播放回执、断线恢复及背景切换。 |
| [`meme_huanxing/`](meme_huanxing/README.md) | 本地待机、睡觉/起床命令与动画状态管理。 |
| [`meme_sing/`](meme_sing/README.md) | 本地点歌、按聊天情绪匹配音乐及语音停歌。 |
| `sucai_meme/` | 本地 GIF、背景 GIF、配套 WAV 与音乐素材。 |
| `zerotier_chat/` | 一键启动所需的网络配置示例与 WSL 网络辅助代码。 |

Qwen 的**模型版本**与**数据版本**分开记录，V3 模型仍使用 `data_spilt/V2.0.0`。V2 质量保护训练使用回复 CE、基座 teacher KL 和聊天回归选优；当前显式选择的 V2 检查点保留 `quality_status=rejected`。V3 只用参考回复及 EOS 的 CE 训练，按固定开发子集的最低 token 加权 CE 选优：训练共 3 轮、157,137 步，选中第 150,000 步，聊天质量状态为 `not_evaluated`。**完成训练或 loss 降低不能证明聊天理解、角色遵守或自然性已通过验收。** 详细训练、选优与发布检查见各版本 README。

## 服务器安装与启动

以下命令在仓库根目录运行。服务器需要 Python 3.10+；实际训练与 8B 推理还需要相容的 PyTorch/CUDA 环境和足够显存。树莓派只运行播放器及离线语音模块。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./model -e ./model_qwen
python -m pip install -r model_information/requirements.txt \
  -r model_information_qwen/requirements.txt \
  -r model_information_qwen_3.0/requirements.txt
export MEME_PYTHON="$PWD/.venv/bin/python"
export QWEN_PYTHON="$MEME_PYTHON"
```

脚本兼容原服务器的 Python 路径；在新环境中应设置上述变量。依赖以两个训练包的 `pyproject.toml` 和推理目录的 `requirements.txt` 为准。

启动前需准备：

1. **模型产物**：对应 Mistral 或 Qwen 基座、完整 planner（含编码器配置与 tokenizer）、prefix、情绪空间、训练回复索引及匹配清单。Qwen 服务还校验相应版本的完整评估/发布记录；这些本机记录不在本次源码提交中。
2. **本机模型配置**：参考 [`model_qwen/inference_config.example.json`](model_qwen/inference_config.example.json) 和 [`V3 配置示例`](model_information_qwen_3.0/inference_config.example.json)。示例仅说明字段与路径，不能直接替代已审核配置或生成有效发布授权。现有 V3 发布记录还绑定指定检查点、配置、提示词和绝对路径，迁移或重新训练后须重新核验，不能只改路径后沿用旧记录。
3. **网络与认证**：复制 `zerotier_chat/network.env.example` 为 `zerotier_chat/network.env`，填写实际地址及用户名。用 `python3 communication/init_config.py` 生成一次口令，将同一份口令安全配置到树莓派；已有口令直接复用。一键启动优先读取 `zerotier_chat/.env`，否则读取 `communication/.env`，也可用 `--token-file` 指定。
4. **树莓派安装**：配置 SSH、播放器、素材及离线 ASR。当前一键脚本针对 Windows ZeroTier + mirrored WSL 网络，使用已配置的 `~/.ssh/hks_zerotier_pi_ed25519` 和本地 known_hosts。其他环境可使用低层 `run.sh serve --host ... --port ...` 入口，并自行启动树莓派播放器；细节见各推理目录说明。

在准备好上述文件的部署机器上，按需选择一个服务：

```bash
# Mistral V2
bash model_information/start_server.sh

# Qwen V2
bash model_information_qwen/start_server.sh

# Qwen V3，仅更换模型，沿用同一套通信与播放器行为
bash model_information_qwen_3.0/start_server.sh
```

三个服务默认使用同一端口，切换模型前应先结束原服务。脚本检查并复用匹配的已加载服务，检查或启动树莓派现有播放器，并显示对话日志；不会自动覆盖树莓派代码、模型、素材或凭据。首次启动终端按 `Ctrl+C` 停止服务，复用服务的日志查看终端按 `Ctrl+C` 只结束查看。

```bash
# 检查已配置部署环境；不启动模型，也不启动树莓派播放器
bash model_information_qwen_3.0/start_server.sh --check

# 跳过树莓派 SSH，只运行服务器
bash model_information_qwen_3.0/start_server.sh --server-only

# 终端文字推理（仍需完整本地模型及有效发布记录）
bash model_information_qwen_3.0/run.sh chat --text '小猫，我今天很开心'
```

`model_information_qwen_3.0` 文件夹名称含点，其脚本通过 `bootstrap.py` 注册合法包名 `model_information_qwen_v3`；请使用提供的脚本入口。

## 对话窗口、背景与本地控制

三个服务器当前统一采用**最多 4 个已完成问答对的滚动窗口**。新请求使用这些历史与当前用户消息；成功回复后淘汰超出的最旧完整问答对，继续同一段对话。模型 token 预算仍可能进一步删去最旧的完整历史。失败请求、缓存重放、待机和点歌不推进窗口。

窗口每滚动 5 次切换一次背景：第 5 个成功问答首次淘汰最旧一对，因此第 **9、14、19…** 个成功回复切换背景；背景切换本身不重开对话。连续两次相同回复后，下次请求会重新开始；播放结束后真正空闲 10 秒、或显式重置也会新建对话。树莓派按 `conversation_id` 和 `background_revision` 保持对应背景，表情仍随 `meme_id` 改变。

背景素材为 `meme_gif_background/<原stem>-N.gif`，配套声音为 `meme_gif/<原stem>.wav`，编号组需完整可用。v2 协议支持就绪握手、完整句结束标志、接收回执、播放完成回执、断线结果查询与去重，避免重连造成重复推理或重复播放。

| 功能 | 示例与行为 |
| --- | --- |
| 待机 | 清醒时说“小猫，睡觉吧”“小猫，睡觉。”“小猫，睡觉了。”或“小me，睡觉吧。”；旧短句“小me，睡觉”继续支持。 |
| 唤醒 | 待机时说“小猫，起床吧”“小猫，起床。”“小猫，起床了。”或“小me，起床吧。”；“小me，起床”“小me，别睡了”继续支持。 |
| 状态区分 | 待机中的睡觉指令继续待机；清醒时的起床话语作为普通聊天。引用、否定、条件句和问句不触发控制。 |
| 自动待机 | 真正等待用户满 20 秒后待机；讲话、推理、表情和音乐播放不计入。起床动画完整结束后恢复聊天收音。 |
| 点歌 | “想听音乐”“放首歌”“来点音乐”；播放期间说“停止音乐”“别放了”等结束。按显式风格或最近聊天的本地情绪规则选曲。 |

睡觉、唤醒和音乐控制在树莓派本地执行，不调用语言模型。ASR 在树莓派离线运行，完整句断句默认等待 0.7 秒；实际识别速度还取决于设备处理能力、音频积压和 USB 状态。规则测试不替代现场麦克风、声音和动画验收。

## 数据恢复与训练

历史提交中的部分大型 JSONL 以 gzip 保存在 `model/git_assets/`。恢复器校验 SHA-256、跳过已正确恢复的文件，并拒绝覆盖内容冲突的本地文件：

```bash
python3 model/tools/restore_git_assets.py
```

该命令只恢复 [`git_upload_manifest.json`](model/git_upload_manifest.json) 中的数据及重复归档数据，**不会下载或恢复模型权重**。

```bash
# 下载固定版本的官方 Qwen3-8B 基座，可续传
bash model_qwen/run.sh download

# 复用已有 V2 planner/空间，首次准备新的 V3 run；目标目录必须尚不存在
python model_qwen/tools/prepare_loss_run.py --config model_qwen/paper_config_v3_loss.json

# 开始或恢复 V3 loss 训练（已有运行目录不要重复执行上面的准备命令）
bash model_qwen/run.sh workflow --config model_qwen/paper_config_v3_loss.json

# 只读查看该版本进度
bash model_qwen/run.sh status --config model_qwen/paper_config_v3_loss.json
```

下载入口校验基座分片；训练需准备 V2.0.0 数据、语言模型基座，并按环境调整配置中的路径。从头训练 planner 还需本地 DeBERTa。原服务器的 V3 在工作流前先用 `prepare_loss_run.py` 核验并复用了 V2 planner/空间；该命令只适用于目标目录尚不存在、源产物完整的新运行。已有运行继续执行 `workflow` 即可；直接对空运行目录执行 `workflow` 会建立空间并训练 planner，单独设置 `planner_reuse_source` 不会自动完成复制。流程完成后标记 `ready_for_review`，不会自动替换线上推理模型。Mistral、Qwen 各阶段和质量保护版本的用法见训练目录 README。

## 与 EmoStance 论文的关系

本项目参考 Ziyuan Jin、Yuxuan Ge、Zheng Tian 的 **EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision**，是面向中文 meme 猫对话的应用与方法适配。论文用 emoji 弱监督学习回复侧情绪取向，并通过连续前缀控制冻结的语言模型。[论文（arXiv:2609.02133）](https://arxiv.org/abs/2609.02133) · [作者官方代码](https://github.com/18277390221/EmoStance)

本项目把多标注者的 meme 选择与置信度合成为软分布，用训练集建立上下文/混淆图、情绪区域和软归属，再由角色感知规划器预测回复取向并构造 prefix。中文图特征采用单字、相邻双字等 TF-IDF；生成后增加 `meme_id` 检索和实体播放器。TCP 服务、滚动窗口、背景切换、待机与点歌属于本项目的工程扩展。

实现与论文实验存在差别，包括中文数据与 meme 标注、语言模型基座、单次生成设置、部分熵门控和辅助损失细节；当前 planner 仍使用英文 DeBERTa，存在中文词表限制。当前 V2/V3 空间有 166 个 meme、256 维向量和 8 个无名区域，这些区域不是预先定义的真实情绪类别。本仓库不代表作者官方实现，也不把论文报告的效果当成本项目的实测结果。

引用原论文可使用：

```bibtex
@article{jin2026emostance,
  title={EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision},
  author={Jin, Ziyuan and Ge, Yuxuan and Tian, Zheng},
  journal={arXiv preprint arXiv:2609.02133},
  year={2026},
  url={https://arxiv.org/abs/2609.02133}
}
```

## 验证与发布边界

不加载真实 8B 权重的基础验证示例：

```bash
python -m unittest discover -s meme_huanxing/tests -v
python -m unittest discover -s meme_sing/tests -v
python -m unittest discover -s meme_software/player/desktop/tests -v
python -m unittest model_information.tests.test_sessions \
  model_information_qwen.tests.test_sessions -v
python model_information_qwen_3.0/bootstrap.py unittest \
  model_information_qwen_v3.tests.test_sessions -v
```

完整测试及真实模型、TCP、树莓派联调入口见各目录 README；部分完整测试要求本地归档产物或有效发布记录，不能把源码测试通过当作真实模型聊天质量合格。

`.gitignore` 排除模型权重、基座、训练运行目录、日志、缓存、口令、SSH 记录和实际推理发布文件；配置示例和必要的受控评估用例保留。EmoStance 派生代码保留其 [MIT 许可证](model_qwen/THIRD_PARTY_LICENSES/EmoStance.txt)。基座模型、第三方数据、音频和图像素材各自遵循来源许可，代码许可不自动授予这些资产的使用权。
