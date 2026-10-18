# Qwen3-8B EmoStance 最终推理

本目录负责最终文字推理、TCP 服务和树莓派接入。模型下载与训练在 `model_qwen/`，默认读取 `model_qwen/inference_config.json`，选择运行目录 `model_qwen/runs/qwen3_8b_v2_selected`。全部生成调用共享训练包 `emostance_qwen.PaperGenerator.complete()`，使用同一个 Qwen 原生对话模板与训练前缀。未配置推理配置的环境可以回退 `paper_config.json`，此时会明确警告，所有加载检查仍然执行。

推理流程：真实 A/B 历史 → EmoStance 目标情绪规划 → 训练前缀引导 Qwen **生成一次回复** → 编码实际回复情绪 → 检索训练回复库对应的 `meme_id`。没有候选池或重排序，固定 `num_beams=1`、`num_return_sequences=1`、`enable_thinking=False`。表情检索不会替换生成文字。

加载时必须存在主运行清单、规划器清单、生成器清单和索引，并且全部为 `status=complete`。检查训练集、规划器和情绪空间与索引的 SHA-256 对应关系，以及主运行清单的训练产物哈希；拒绝缺失前缀、未完成训练、旧 Mistral 基座和 Git LFS 指针。推理只读本地文件，不会下载模型或临时训练随机前缀。

当前 prefix 使用基座 teacher KL 保留语言行为。普通发布必须通过开发集和独立聊天验收；用户明确授权选用已训练检查点时，另行验证 `deployment_selection.json` 中的授权、检查点及完整开发集和留出集报告绑定。显式选择不会把 `quality_status=rejected` 改为 `passed`，也不能省略报告、绕过哈希与生成配置验证。没有有效选择记录的运行仍必须通过原有全部质量检查。

`inspect_model_version()` 分别返回实际 `quality_status`、推理发布状态 `release_status`、发布方式 `release_mode` 和检查点 `selected_step`。显式选择完成后可以同时显示 `quality_status=rejected`、`release_status=ready`、`release_mode=user_selected_with_known_quality_failures`；普通验收通过时显示 `release_status=quality_passed`。自动语义评审使用无 prefix Qwen，存在自评偏差；固定用例不能保证所有场景的语言质量。原始报告和训练说明见 [`model_qwen/README.md`](../model_qwen/README.md)。

额外对话提示策略目前 **未启用**：候选提示没有完成并通过新的开发集与独立留出回归，`dialogue_policy.json` 和 `dialogue_policy_release.json` 均未创建，当前仍使用原提示和已选择权重，没有上线提示词复读修复。`inspect_model_version()` 的 `dialogue_policy.enabled` 会如实返回 `false`。可选策略框架支持追加或完整替换系统提示，但必须先通过原模型发布检查，再核验两个新回归的完整真实历史、逐例输出、单次生成、原生成参数、人工语义审阅以及独立留出集一次性记录；模型、原提示、策略、用例和报告全部绑定 SHA-256。补充策略的 `regression_passed` 只代表所记录的对话回归，不会改变原模型的 `quality_status=rejected`。

## 本地推理

```bash
# 单条输入，输出 reply 和 meme_id。
bash model_information_qwen/run.sh chat --text 'meme，我今天很开心'

# 终端连续对话；/quit 退出。
bash model_information_qwen/run.sh chat

# 选择其他已完成的 Qwen 训练运行。
bash model_information_qwen/run.sh chat \
  --config model_qwen/inference_config.json \
  --model-dir model_qwen/runs/qwen3_8b_v2_selected
```

脚本默认使用 `/home/aida102/emoji/.venv/bin/python`，也可设置 `QWEN_PYTHON=/环境/bin/python`；`MEME_PYTHON` 作为兼容后备。环境需要 `model_qwen` 的依赖及本目录 `requirements.txt`。

`chat_config.json` 默认关闭独立 ASR 大模型校对，保留轻量角色名字规范化。默认角色提示以训练配置的 `system_prompt_file` 解析结果为准，使训练和推理一致。`--chat-config` 指定的生成参数和角色提示必须与已验证的报告一致。历史超预算时仅移除最旧的完整对话对，当前用户输入过长会报错。

## TCP 与树莓派

服务器端以后使用同一条命令启动、复用服务和查看双方对话：

```bash
bash /home/aida102/hks/model_information_qwen/start_server.sh
```

脚本与当前 `model_information/start_server.sh` 使用相同的启动流程：读取原 `zerotier_chat/network.env` 和共享口令，监听 `10.205.84.185:9000`，连接树莓派 `10.205.84.95`，检查已有 WSL 路由与 SSH 配置。树莓派播放器已运行时保留，未运行时启动已安装的软件及离线识别模块；不需要到树莓派终端另开客户端。默认加载完成 52,379 步、1/1 轮训练的 `qwen3_8b_v2_selected` 及配套 planner、prefix 和表情索引。

两种服务器启动脚本启动树莓派播放器时都显式传入 `--end-silence 0.7`，说完停顿约0.7秒后提交整句。已运行的播放器保留原启动参数，阈值更新需要重启树莓派播放器后生效。

重复执行同一命令会认证检查并复用已加载的 Qwen，不重复加载权重；播放器关闭后可以用同一命令补启动。复用时当前终端显示最近 20 条双向对话，并继续实时查看。首次启动的终端按 Ctrl+C 停止服务；复用服务的查看终端按 Ctrl+C 只退出查看。私有对话日志保存于 `model_information_qwen/runs/conversation.log`，认证口令和心跳不写入对话日志。

```bash
# 仅检查配置、端口及树莓派状态。
bash model_information_qwen/start_server.sh --check

# 跳过树莓派 SSH 检查与自动启动。
bash model_information_qwen/start_server.sh --server-only

# 复用已有服务并检查/启动树莓派后返回，不持续显示对话。
bash model_information_qwen/start_server.sh --no-follow

# 低层服务入口，直接运行 TCP 服务。
bash model_information_qwen/run.sh serve \
  --host 10.205.84.185 --port 9000 \
  --token-file zerotier_chat/.env
```

一键启动读取既有 `zerotier_chat/network.env`、口令和 SSH 配置。`--config`、`--model-dir` 可选择训练配置与运行目录。9000 端口被其他服务占用时会退出；先在原服务终端停止旧服务，再启动 Qwen。每个服务只加载一份权重。

TCP 协议与最新 `model_information` 一致：UTF-8、每行一个 JSON，结果含 `reply`、`reply_text`、`meme_id` 和兼容旧客户端的 JSON `text`。树莓派现有播放器使用协议 v2，就绪握手明确返回 `model_ready=true`、所选 Qwen 版本和 `server_instance_id` 后才开麦。录音开始发送 `speech_start`，整句识别结束发送含 `end:true` 的 `request`；服务器先返回 `accepted`，再返回实际推理 `result`。播放器播完表情和声音后发送 `playback_done`，失败或取消使用 `playback_failed`。现有树莓派软件无需因换模型修改协议或口令。

录音、排队推理和播放期间暂停 10 秒用户空闲计时，播放完成后重新开始计时；录音和播放等待分别有 30 秒、120 秒租约。断线后以同一个 `session_id`、`request.id` 发送 `request_status` 取回结果，服务器先缓存再发送，重复查询不重复推理或写入历史；终结结果保留 10 分钟、最多 1024 条。服务器重启会更换 `server_instance_id`，不自动重发不确定是否已处理的语句。

旧调试客户端继续兼容 v1：先发含共享口令的 `hello`，再发 `{"type":"request","id":"r1","text":"今天很开心"}`。调试客户端：

```bash
python3 -m model_information_qwen.pi_client \
  --host 10.205.84.185 --port 9000 --token-file zerotier_chat/.env
```

服务保留会话隔离、请求 FIFO、全局模型串行调用、完整成功轮次提交和上下文预算裁剪。v1 与本地文字聊天采用用户消息到达计时；生产播放器的 v2 使用上述录音/播放活动计时。Qwen 保留自身训练时的原生聊天模板、关闭 thinking 和已记录生成参数；每轮最终回复仍只有一次生成，没有候选池或重排序。

Qwen 服务器、直接调用的 `DialogueManager` 和命令行交互聊天默认滚动保留 **最近 4 个已完成的用户/助手问答对**。第 5 轮推理读取前 4 对历史和当前用户消息，回复成功后才删除最旧一对；此后每个成功轮次继续这样滚动，`conversation_id` 保持不变。每次推理最多读取 9 条消息，提交后的历史最多 8 条；上下文超预算时仍可进一步移除最旧的完整问答对。排队请求保持 FIFO，不同设备分别维护历史；失败、被拒绝、仍在排队的请求和缓存查询不会推动窗口。

每累计删除 5 对历史，`background_revision` 增加 1。因此连续成功回复的第 1—8 轮为 `0`，第 9—13 轮为 `1`，第 14—18 轮为 `2`，第 19 轮为 `3`；树莓派依据 `(conversation_id, background_revision)` 变化切换背景，历史保持连续。v2 就绪握手返回 `max_turns=4`、`context_window_turns=4`、`background_roll_interval=5`，并声明 `sliding_context` 和 `background_revision` 能力。每个 v2 成功结果携带提交时的非负整数 `background_revision`，缓存重查保留该结果原来的编号和背景版本。v1 保持原握手及结果字段。一键启动和低层服务共用默认窗口；调用方仍可显式传入其他正整数 `max_turns` 或 `None`，背景更新间隔固定为每滑动 5 对。

同一段对话中，若连续两个成功回复的 `reply` 文字完全一致（不比较 `meme_id`），第二条回复仍正常返回；**下一次新提问先清空历史并开启新对话**，已经排队的后续请求也采用新对话。空闲 10 秒和手动重置同样开启新对话。这些实际重置会更换 `conversation_id`，清零滑动计数和背景版本；此前缓存的回复保留原编号和版本，缓存重查不会触发重置或重复推理。最近两个成功回复独立保存，因此 `max_turns=1` 仍能检测重复。仅比较相邻成功轮次，不做去空格、标点或语义相似判断；失败和被拒绝的请求不计入比较，不同设备互不影响。显式设置 `max_turns=None` 保留无限历史，重复回复保护、空闲及手动重置继续生效。

本地点歌由树莓派上的 [`meme_sing/`](../meme_sing/README.md) 处理，与 Qwen 模型参数无关。说“想听音乐”会按最近成功聊天文字和音乐文件名情绪标签选曲；播放中说“停止音乐”结束歌曲并恢复聊天。点歌、停歌及歌词不提交模型，不写入模型聊天历史；没有播放音乐时“停止”继续作为普通聊天输入。音乐播放期间保留本地停止指令识别，普通表情及其声音仍遵守原来的半双工规则。

## 语音待机与唤醒

共享树莓派播放器同时支持“小猫”和“小me”称呼。待机指令包括“小猫，睡觉吧”“小猫，睡觉”“小猫，睡觉了”“小me，睡觉吧”；唤醒指令包括“小猫，起床吧”“小猫，起床”“小猫，起床了”“小me，起床吧”。原短句、“别睡了”及有限同音别名“小咪”“小米”“小美”等继续兼容，标点、空格和大小写按规则规范化。清醒时的睡觉指令进入待机；待机时的起床指令才触发唤醒，睡觉指令不会唤醒，清醒时的起床语句仍作为普通聊天。待机中的控制在树莓派本地处理，不提交模型、不滑动历史窗口；引用、否定、条件句和问句不触发。语音转写仍使用已有离线 ASR，解析测试不替代现场麦克风验收。详见 [`meme_huanxing/README.md`](../meme_huanxing/README.md)。

## Python 与验证

```python
from model_information_qwen.model import load_chat_engine

engine = load_chat_engine()
result = engine.reply([{"role": "user", "content": "今天终于做完啦"}])
print(result["reply"], result["meme_id"])
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/aida102/emoji/.venv/bin/python \
  -m unittest discover -s model_information_qwen/tests -v
```

测试用微型模型验证真实 Qwen 生成实现与训练模板一致、关闭 thinking、单次 generate、情绪前缀和历史预算；另检查未完成训练与哈希不匹配拒绝加载、显式选择仍保留真实质量状态、默认配置回退不绕过发布检查、会话计时及本机 TCP 协议。微型模型测试不代表真实回复语义质量。真实训练产物联调由 `model_qwen/tools/verify_runtime.py` 执行，未完成训练或缺少有效发布依据时该检查会明确失败。

服务端同步后 321 项测试通过；以下工具会加载实际 Qwen，使用本机随机端口和临时口令，验证 v1/v2 各三轮 `[1,3,1]` 历史、单次生成、表情最近邻、播放后的空闲重置、缓存及重连恢复：

```bash
PYTHONUTF8=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information_qwen.tools.verify_integration

# 只读检查实际树莓派就绪、收音及播放状态。
PYTHONUTF8=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information_qwen.tools.verify_pi_handshake --check
```

实际联调记录保存在 `model_information_qwen/runs/integration_chat_smoke.json`。实际树莓派状态检查不会播放测试素材，也不提交模型请求。


## 公开源码部署前提

实际 `model_qwen/inference_config.json`、完整模型运行目录和本机发布记录不随核心源码上传。字段示例见 [inference_config.example.json](../model_qwen/inference_config.example.json)；示例不能替代完整匹配产物或有效发布记录，克隆源码不代表可以直接推理。安装依赖与跨版本启动见 [项目 README](../README.md)。
