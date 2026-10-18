# Qwen3-8B V3.0.0 最终推理

本目录 `model_information_qwen_3.0` 使用纯 loss 训练的 V3.0.0 最佳检查点，其他服务、协议、树莓派和会话行为与当前 `model_information_qwen` 一致。默认读取本目录的 `inference_config.json`，选择 `model_qwen/runs/qwen3_8b_v3_loss_ce`；原推理目录、配置和运行记录独立保留。

V3 使用 2.0.0 数据，prefix 完成 3 轮、157137 次更新。按固定 dev512 的回复及 EOS token 加权交叉熵选择 `generator/best.pt`：第 3 轮第 150000 步，选择时 dev CE 为 `3.918008093610924`。`last.pt` 是第 157137 步的可恢复训练状态，推理读取 best。仅复用已训练 planner、情绪空间和图，prefix 从新初始化训练；没有 teacher KL、回复过滤、语言保留 anchors 或回复合理性参与选优。

用户明确要求在本目录使用这版最佳 loss 模型。独立 `inference_release.json` 绑定授权、版本、配置和训练产物哈希，加载时验证该记录与已完成的 planner、generator、固定 dev 选择、dev/test loss 报告及训练回复索引。原训练主清单仍为 `ready_for_review`，`quality_status` 和 `chat_quality_status` 如实保留 `not_evaluated`；本目录发布方式为 `user_selected_loss_only`，不把 loss 结果写成回复语义验收通过。没有创建额外系统提示策略。

推理流程：完整真实历史 → EmoStance 目标情绪规划 → 训练前缀引导 Qwen **生成一次回复** → 编码实际回复情绪 → 检索训练回复库对应的 `meme_id`。生成调用复用训练包的 Qwen 原生模板，固定 `num_beams=1`、`num_return_sequences=1`、`enable_thinking=False`，没有候选池或重排序。冻结基座继续共用 `model_qwen/backbones/Qwen3-8B`，不会复制、下载或临时训练模型。

目录名包含点，Python 使用 `bootstrap.py` 注册合法别名 `model_information_qwen_v3`，包源码和相对导入始终指向本目录。`run.sh` 和 `start_server.sh` 已自动完成引导，无需重命名目录或安装第二份包。

## 本地推理

```bash
# 单条输入，输出 reply 和 meme_id。
bash model_information_qwen_3.0/run.sh chat --text 'meme，我今天很开心'

# 终端连续对话；/quit 退出。
bash model_information_qwen_3.0/run.sh chat

# 显式指定本目录已授权的 V3 配置与训练运行。
bash model_information_qwen_3.0/run.sh chat \
  --config model_information_qwen_3.0/inference_config.json \
  --model-dir model_qwen/runs/qwen3_8b_v3_loss_ce
```

脚本默认使用 `/home/aida102/emoji/.venv/bin/python`，也可设置 `QWEN_PYTHON=/环境/bin/python`；`MEME_PYTHON` 作为兼容后备。环境需要 `model_qwen` 的依赖及本目录 `requirements.txt`。

`chat_config.json` 默认关闭独立 ASR 大模型校对，保留轻量角色名字规范化。默认角色提示以训练配置的 `system_prompt_file` 解析结果为准，使训练和推理一致。`--chat-config` 指定的生成参数和角色提示必须与已验证的报告一致。历史超预算时仅移除最旧的完整对话对，当前用户输入过长会报错。

## TCP 与树莓派

服务器端以后使用同一条命令启动、复用服务和查看双方对话：

```bash
bash /home/aida102/hks/model_information_qwen_3.0/start_server.sh
```

脚本与当前 `model_information/start_server.sh` 使用相同的启动流程：读取原 `zerotier_chat/network.env` 和共享口令，监听 `10.205.84.185:9000`，连接树莓派 `10.205.84.95`，检查已有 WSL 路由与 SSH 配置。树莓派播放器已运行时保留，未运行时启动已安装的软件及离线识别模块；不需要到树莓派终端另开客户端。默认加载 `qwen3_8b_v3_loss_ce` 的第 150000 步最佳 loss prefix、配套 planner 和表情索引。

两种服务器启动脚本启动树莓派播放器时都显式传入 `--end-silence 0.7`，说完停顿约0.7秒后提交整句。已运行的播放器保留原启动参数，阈值更新需要重启树莓派播放器后生效。

重复执行同一命令会认证检查并复用已加载的 Qwen，不重复加载权重；播放器关闭后可以用同一命令补启动。复用时当前终端显示最近 20 条双向对话，并继续实时查看。首次启动的终端按 Ctrl+C 停止服务；复用服务的查看终端按 Ctrl+C 只退出查看。私有对话日志保存于 `model_information_qwen_3.0/runs/conversation.log`，认证口令和心跳不写入对话日志。

```bash
# 仅检查配置、端口及树莓派状态。
bash model_information_qwen_3.0/start_server.sh --check

# 跳过树莓派 SSH 检查与自动启动。
bash model_information_qwen_3.0/start_server.sh --server-only

# 复用已有服务并检查/启动树莓派后返回，不持续显示对话。
bash model_information_qwen_3.0/start_server.sh --no-follow

# 低层服务入口，直接运行 TCP 服务。
bash model_information_qwen_3.0/run.sh serve \
  --host 10.205.84.185 --port 9000 \
  --token-file zerotier_chat/.env
```

一键启动读取既有 `zerotier_chat/network.env`、口令和 SSH 配置。`--config`、`--model-dir` 可选择训练配置与运行目录。9000 端口被其他服务占用时会退出；先在原服务终端停止旧服务，再启动 Qwen。每个服务只加载一份权重。

TCP 协议与最新 `model_information` 一致：UTF-8、每行一个 JSON，结果含 `reply`、`reply_text`、`meme_id` 和兼容旧客户端的 JSON `text`。树莓派现有播放器使用协议 v2，就绪握手明确返回 `model_ready=true`、所选 Qwen 版本、`server_instance_id` 和默认 `max_turns=4` 后才开麦。录音开始发送 `speech_start`，整句识别结束发送含 `end:true` 的 `request`；服务器先返回 `accepted`，再返回实际推理 `result`。播放器播完表情和声音后发送 `playback_done`，失败或取消使用 `playback_failed`。现有树莓派软件无需因换模型修改协议或口令。

录音、排队推理和播放期间暂停 10 秒用户空闲计时，播放完成后重新开始计时；录音和播放等待分别有 30 秒、120 秒租约。断线后以同一个 `session_id`、`request.id` 发送 `request_status` 取回结果，服务器先缓存再发送，重复查询不重复推理或写入历史；终结结果保留 10 分钟、最多 1024 条。服务器重启会更换 `server_instance_id`，不自动重发不确定是否已处理的语句。

旧调试客户端继续兼容 v1：先发含共享口令的 `hello`，再发 `{"type":"request","id":"r1","text":"今天很开心"}`。调试客户端：

```bash
python3 model_information_qwen_3.0/bootstrap.py pi_client \
  --host 10.205.84.185 --port 9000 --token-file zerotier_chat/.env
```

服务保留会话隔离、请求 FIFO、全局模型串行调用、完整成功轮次提交和上下文预算裁剪。v1 与本地文字聊天采用用户消息到达计时；生产播放器的 v2 使用上述录音/播放活动计时。Qwen 保留自身训练时的原生聊天模板、关闭 thinking 和已记录生成参数；每轮最终回复仍只有一次生成，没有候选池或重排序。

Qwen 服务器、直接调用的 `DialogueManager` 和命令行交互聊天默认保留最近 **4 个已完成的成功问答对**（`max_turns=4`）。每次推理携带最近最多 4 轮完整历史及当前用户问题；成功提交后删除窗口外最旧的完整问答对。第 5 轮完成时首次滑动，滚动过程保持同一个 `conversation_id` 和连续上下文。窗口每滑动 5 次增加一次 `background_revision`，即没有其他重置时第 9、14、19 轮回复切换背景；背景变动不清空历史。v2 握手返回 `context_window_turns=4`、`background_roll_interval=5` 和兼容字段 `max_turns=4`，结果中的 `background_revision` 由树莓派用于选择新的背景，已有缓存保留原编号、背景版本及播放快照。排队请求保持 FIFO，不同设备分别计数；失败、被拒绝或缓存重查不滑动窗口。空闲 10 秒、连续重复回复保护和手动重置仍可开启真正的新对话，并将滑动计数归零。每次模型输入仍受原 token 预算约束。一键启动和低层服务共用该默认设置；调用方仍可显式传入 `max_turns=None` 取消历史窗口大小限制。

同一段对话中，若连续两个成功回复的 `reply` 文字完全一致（不比较 `meme_id`），第二条回复仍正常返回；**下一次新提问先清空历史并开启新对话**，已经排队的后续请求也采用新对话。v2 返回新的 `conversation_id`，此前缓存的回复保留原编号，缓存重查不会触发重置或重复推理。仅比较相邻成功轮次，不做去空格、标点或语义相似判断；失败和被拒绝的请求不计入比较，不同设备互不影响。该保护与四轮滚动窗口、空闲及手动重置并存，显式 `max_turns=None` 时仍保留重复回复保护。

本地点歌由树莓派上的 [`meme_sing/`](../meme_sing/README.md) 处理，与 Qwen 模型参数无关。说“想听音乐”会按最近成功聊天文字和音乐文件名情绪标签选曲；播放中说“停止音乐”结束歌曲并恢复聊天。点歌、停歌及歌词不提交模型，也不写入聊天历史；没有播放音乐时“停止”继续作为普通聊天输入。音乐播放期间保留本地停止指令识别，普通表情及其声音仍遵守原来的半双工规则。

## 语音待机与唤醒

共享树莓派播放器同时支持“小猫”和“小me”称呼。待机指令包括“小猫，睡觉吧”“小猫，睡觉”“小猫，睡觉了”“小me，睡觉吧”；唤醒指令包括“小猫，起床吧”“小猫，起床”“小猫，起床了”“小me，起床吧”。原短句、“别睡了”及有限同音别名“小咪”“小米”“小美”等继续兼容，标点、空格和大小写按规则规范化。清醒时的睡觉指令进入待机；待机时的起床指令才触发唤醒，睡觉指令不会唤醒，清醒时的起床语句仍作为普通聊天。待机中的控制在树莓派本地处理，不提交模型、不滑动历史窗口；引用、否定、条件句和问句不触发。语音转写仍使用已有离线 ASR，解析测试不替代现场麦克风验收。详见 [`meme_huanxing/README.md`](../meme_huanxing/README.md)。

## Python 与验证

```python
import runpy

runpy.run_path("model_information_qwen_3.0/bootstrap.py")["register_package"]()
from model_information_qwen_v3.model import load_chat_engine

engine = load_chat_engine()
result = engine.reply([{"role": "user", "content": "今天终于做完啦"}])
print(result["reply"], result["meme_id"])
```

```bash
PYTHONDONTWRITEBYTECODE=1 /home/aida102/emoji/.venv/bin/python \
  model_information_qwen_3.0/bootstrap.py unittest discover -s model_information_qwen_3.0/tests -v
```

测试用微型模型验证 Qwen 生成实现与训练模板一致、关闭 thinking、单次 generate、情绪前缀和历史预算；另检查 V3 发布记录与哈希门禁、独立默认配置、启动复用、会话计时及本机 TCP 协议。微型模型测试不代表真实回复语义质量。

以下工具按需加载实际 Qwen，使用本机随机端口和临时口令进行工程联调；不会启动生产 9000 端口。`verify_model_swap` 验证 V3 最佳检查点、v1/v2 单次生成、缓存、默认四轮滚动窗口、背景版本及重复回复保护：

```bash
PYTHONUTF8=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  model_information_qwen_3.0/bootstrap.py tools.verify_model_swap

# 只读检查实际树莓派就绪、收音及播放状态。
PYTHONUTF8=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  model_information_qwen_3.0/bootstrap.py tools.verify_pi_handshake --check
```

树莓派状态检查仅复用既有 SSH 连接读取状态，不会播放测试素材或提交模型请求。启动脚本只检查或启动既有 Pi 软件，不调用 `deploy_pi.py`，不复制树莓派源码、素材、模型、口令或配置。


## 公开源码部署前提

本目录的实际 `inference_config.json` 和 `inference_release.json` 仅留本机，公开 [inference_config.example.json](inference_config.example.json) 只说明字段与路径。运行需要完整且匹配的 V3 产物和有效发布记录；改示例路径不能使既有哈希绑定或授权自动生效，搬迁或重新训练后须重新核验。安装依赖与跨版本启动见 [项目 README](../README.md)。
