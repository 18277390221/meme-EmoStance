# 树莓派文字与模型推理通信

树莓派发送一次文字，服务器生成一次回复，并返回 `{"reply":"回复文字","meme_id":"meme-0008"}`。本目录复用 `communication/common.py` 的 TCP JSON 协议、认证和心跳；新增模型加载、对话管理和业务客户端。

## 服务器一键启动

当前设备已经配置好 ZeroTier 和共享口令。只需在服务器执行以下命令，之后每次启动都用同一条命令：

```bash
bash /home/aida102/hks/model_information/start_server.sh
```

脚本自动读取既有 `zerotier_chat/network.env`，使用服务器 `10.205.84.185:9000` 和树莓派 `10.205.84.95`，复用服务器 `zerotier_chat/.env` 的原口令。无需再运行 `init_config.py`，无需重新填写地址，也无需到树莓派终端启动通信客户端。

启动时检查端口与本地模型环境，准备既有 WSL 镜像网络的树莓派专用路由，再通过已经配置的 SSH 密钥检查树莓派软件：运行中就保持原样；未运行则在已登录的树莓派桌面上启动原软件及其离线识别模块。缺少桌面会话时会明确提示。不会替换树莓派软件、素材、模型或口令文件。

本目录与 `model_information_qwen/start_server.sh` 启动树莓派播放器时都显式传入 `--end-silence 0.7`，说完停顿约0.7秒后提交整句。已运行的播放器保留原启动参数，阈值更新需要重启树莓派播放器后生效。

树莓派实际安装的软件已经集成 STT、ZeroTier 连接和播放接口。服务器启动完成后，它会自动连接；对麦克风说话即可走“语音识别 → 模型回复 → 表情/配套声音/字幕”。请保持树莓派开机、桌面已登录，两端 ZeroTier 在线。

启动后保留服务器终端，直接查看双方对话。**两端以后都只用上面这一条服务器命令**：模型服务没有运行时正常加载；模型已经运行或正在加载时，继续通过 SSH 检查树莓派，播放器关闭或重启后未启动就补启动，播放器已运行则保留。复用服务后，当前终端显示最近最多 20 条双向对话，并继续实时打印新消息。重复执行会复用原模型，不重复加载权重，也不重启正在运行的播放器。并发启动命令会串行检查树莓派，避免同时创建两个播放器。

首次在本终端启动模型时，**Ctrl+C** 停止该模型服务；复用已有服务查看对话时，**Ctrl+C** 只退出当前查看，后台模型和树莓派继续运行。自动化脚本若只需要启动、无需持续显示，在命令后增加 `--no-follow`；该选项只让复用已有模型的命令直接返回，首次模型启动仍在前台运行。

已有模型服务使用的模型版本或地址与本次参数不同，会明确报错，要求先停止原服务再切换；不会把复用已有服务说成模型切换成功。`--check` 在模型已运行时也可执行，只读检查握手和树莓派状态。`--server-only` 继续跳过树莓派操作。其他旧服务占用 9000 时，脚本会提示在原终端退出后重试。

服务器终端实时打印接收到的树莓派文字，以及发送回树莓派的回复和表情编号。每行带时间、设备名称和请求编号，方便对应同一次请求。例如：

```text
[2026-10-17T15:00:00+08:00] [树莓派 → 服务器] {"device": "meme-player", "request_id": "r1", "text": "今天很开心"}
[2026-10-17T15:00:02+08:00] [服务器 → 树莓派] {"device": "meme-player", "request_id": "r1", "reply": "真替你开心", "meme_id": "meme-0008"}
```

此处为格式示例，实际回复由模型生成。`text` 是树莓派本地识别后发送的文字，`reply` 是服务器返回的回复，`meme_id` 是配套表情编号。回复日志在网络发送完成后打印，重连取回缓存回复也会显示；模型处理失败会打印业务错误。认证口令和心跳不进入对话日志。

新启动的服务同时将双向业务日志写入 `model_information/runs/conversation.log`（权限 `0600`），复用命令从这里实时读取。当前旧后台服务仍可直接读取其 stdout 日志，因此显示对话无需为本次改动重新加载模型。旧前台服务若没有共享日志，需要在其原终端查看，或正常停止后重新执行同一启动命令。

## 角色与语音输入

当前关闭角色系统提示词：`chat_config.json` 中 `role_prompt_enabled=false`。带情绪 prefix 的回复直接复用所选归档训练代码的 `generation_prompt`，将 user/assistant 映射为 A/B，以 `Dialogue:\nA: 用户原文\nB:` 编码，使用相同 tokenizer 添加一次 BOS，不再套用 `[INST]` 聊天模板。默认使用 V2.0.0 的配套情绪规划器和训练好的 prefix。

角色提示文件 `prompts/companion_zh.txt` 保留备用。当前严格对齐训练格式的 prefix 回复不接受角色系统提示词；`role_prompt_enabled` 应保持 `false`，开启时会在加载权重前明确报错。文件中的 meme 猫咪人设、中文口语及通常 10～25 字、最多 40 字等要求不会传给模型；代码不按 40 字硬截断。输入格式对齐不代表已验证对话理解能力改善。

默认 `asr_enabled=false`，每轮只有一次大模型生成。没有额外的大模型判断、独立纠错、候选池、排序或重生成步骤。

语音中明确用于称呼助手的近音名（如“咪咪猫，你在吗”“米米猫，陪我聊聊”）会由轻量规则先规范为“meme”。`MEME`、`Meme` 及旧称 `meme猫` 等拼写形式也可归一。引用、代码、真实宠物名字和明确命名意图会保留，不把任意“某某猫”改成角色名。`asr_enabled` 只控制独立的大模型校对，关闭时仍执行名字规则。

默认处理流程：

1. 登记用户消息到达时刻，将明确的助手称呼规范为“meme”，保留其他输入文字。
2. 根据有效用户文本和真实 A/B 历史规划目标情绪向量，通过训练好的 prefix 引导 Mistral 单次生成回复。
3. 根据最终实际回复计算情绪向量，检索最近训练回复对应的表情，返回 `reply` + `meme_id`。
4. 成功时把有效用户文本与最终回复作为完整 A/B 对保存；普通错字不会被另行改写后放入历史。

例如，默认输入“咪咪猫，我今天不开心，鸭力很大。”会以“meme，我今天不开心，鸭力很大。”进入推理和历史。回复模型结合上下文理解明显错字，但不会再生成一份“压力很大”的校对文本。服务器打印原文、采用文本和状态，方便核对名字规则的实际行为。

角色指令不会作为 A/B 历史。只有最终采用的用户文本与最终回复组成历史。树莓派使用下述 v2 握手，录音、推理和播放期间暂停空闲计时，播放完成后才开始等待用户的 10 秒倒计时；旧调试客户端继续使用原来的消息到达计时规则。

独立校对保留为可选功能：将 `chat_config.json` 的 `asr_enabled` 改为 `true` 并重启后，会先用同一份 Mistral 做一次确定性校对，再按校对结果处理。该步骤不带情绪 prefix，继续使用官方聊天模板和 `prompts/asr_repair_zh.txt` 的校对系统提示。校对结果仍需通过数字、否定、人物、情绪强度及拼音保护检查，失败时保留名字归一后的原文。默认关闭此功能。

带 prefix 的回复沿用归档训练输入上限（当前为 384 token），并受 `max_prompt_tokens` 和模型剩余上下文预算共同限制。预算不足时先裁去最旧的完整 A/B 对，保留当前用户原文及末尾 `B:`；当前输入单独超限时明确提示分短句发送，不静默截断。历史默认滚动保留最近 4 个完整成功问答对；每次推理读取这 4 对历史加当前用户输入，最多 9 条消息。成功回复后删除超出的最旧完整问答对，历史最多 8 条消息。第 5 轮成功后首次滑动窗口，保留对话编号和剩余历史；实际模型输入仍可能因 token 预算进一步裁剪。可选独立校对使用自己的官方聊天模板路径。

运行配置在 `chat_config.json`，可调整角色与校对开关、提示词路径、生成长度和采样参数。归档权重、训练代码和原 EmoStance 目录保持原样；原训练评估指标不等于本轮新增提示词的对话质量评测。

现有服务器环境已补充轻量拼音依赖。换环境时，在原模型依赖基础上安装：

```bash
uv pip install --python /home/aida102/emoji/.venv/bin/python \
  -r model_information/requirements.txt
```

仅检查配置、口令匹配和树莓派状态（不加载模型、不启动软件、不修改网络）：

```bash
bash /home/aida102/hks/model_information/start_server.sh --check
```

仅启动模型服务、跳过 SSH 检查和树莓派自动启动时，增加 `--server-only`。换模型版本可使用 `--model-dir`；换口令文件可使用 `--token-file`，但必须与树莓派一致。`ZT_CHAT_TOKEN` 环境变量仍优先于文件。启动脚本默认使用现有 `/home/aida102/emoji/.venv/bin/python`，换环境可设置 `MEME_PYTHON=/其他环境/bin/python`。

当前默认推理版本为 **V2.0.0**。启动会校验并加载 `model_canshu/V2.0.0` 的完整配套情绪规划器、情绪空间、prefix 参数和回复向量库，以及本地 `/home/aida102/emoji/Mistral-7B-Instruct-v0.3` 基座；完成后打印“模型通信服务已启动”。一键启动、低层服务入口和未指定版本的模型加载均使用此版本，模型只加载一次并复用。可通过 `--model-dir` 选择其他归档目录；低层 `run.sh` 还支持 `--device cuda`、`--dtype bfloat16` 选择设备和精度。服务器需要现有模型推理环境及足够显存；树莓派客户端只需 Python 3.10+ 标准库。

### 手动通信客户端（调试用）

树莓派已有软件不需要执行本节。若要单独调试通信，可在树莓派保存 `communication/` 和 `model_information/` 两个目录，无需复制模型权重，在项目根目录执行：

```bash
python3 -m model_information.pi_client \
  --host 10.205.84.185 --port 9000 --name raspberrypi \
  --token-file ~/zerotier_chat/.env
```

每输入一行发送一次用户话语，输出一条包含 `reply` 和 `meme_id` 的 JSON；`/quit` 退出。已有 `communication/client.py` 也可连接，返回值在其 `result.text` 中显示为 JSON。

低层服务入口 `run.sh` 仍可直接使用，但不会检查或启动树莓派软件，且需显式指定原口令文件：

```bash
bash model_information/run.sh --host 10.205.84.185 --port 9000 \
  --token-file /home/aida102/hks/zerotier_chat/.env
```

## 接入树莓派的语音识别或播放器

应用启动时创建一个客户端并持续复用。将每次收到的完整识别文字交给 `request()`，其返回值可直接交给语音合成与素材播放模块：

```python
import asyncio
from pathlib import Path
from communication.common import load_token
from model_information.pi_client import InferenceClient

async def main():
    client = InferenceClient(
        "10.205.84.185", 9000, "raspberrypi",
        token=load_token(Path.home() / "zerotier_chat/.env"),
    )
    try:
        await client.start()
        # 每次 STT 识别完成时调用一次；不必在客户端拼接历史。
        result = await client.request("今天终于把任务完成了，很开心")
        print(result["reply"], result["meme_id"])
    finally:
        await client.close()

asyncio.run(main())
```

`request()` 默认等待模型完成，支持 `timeout=秒数`。断线会自动重连，但送达状态不明确的请求会报错，不自动重发。已经被服务器接收的请求仍会完成，并在会话未过期时写入历史。后续新请求可以继续使用该实例。

## 对话历史与 10 秒规则

- **实际树莓派播放器使用协议 v2**：只有模型加载完成并明确确认 `model_ready=true` 后才开麦。用户实际开始说话时发送 `speech_start`；整句识别结束并关闭采集后发送 `request`，必须带 `end:true`。服务器先回 `accepted`，再推理并返回 `result`。`accepted` 只表示接收，不能播放，也不能开启下一轮。
- 同一树莓派只允许一个交互进行中。等待推理、准备媒体和播放期间均停麦；GIF 与配套声音都结束后发送 `playback_done`，再恢复收音。播放失败或取消使用 `playback_failed` 释放本轮。
- v2 的 10 秒表示真正等待用户说话的空闲时间。录音、排队推理、生成和播放期间不占用这 10 秒；播放完成时重新开始计时。录音活动有 30 秒租约、等待播放结束有 120 秒租约，客户端异常退出不会无限保留暂停状态。心跳不算用户活动。
- 同一进程重连保留 `session_id`。断线立即停麦，以原 `request.id` 发送 `request_status` 查询；处理中的请求等待原结果，已完成的请求返回缓存。服务器先缓存结果再发送，避免网络中断导致重复推理或重复提交历史。终结结果缓存 10 分钟、最多 1024 条，进行中的请求不被缓存淘汰。
- 每次服务器启动有新的 `server_instance_id`。若服务器重启或原请求已不可查询，树莓派提示重新说话，不自动重发不确定是否已处理的句子；本机制不承诺跨服务器重启持久保存结果。
- 一条成功接收的 `request.text` 就是一次用户发言，即 A；服务器回复为 B。文字中的换行或多句话不会拆分。
- 一次成功的用户输入与模型回复（A/B 对）计为一轮。`max_turns=4` 表示历史窗口大小：推理时读取最近最多 4 个已完成问答对和当前输入，成功后保存新问答对并删除最旧的超额完整对。第 5 次推理仍能看到第 1～4 轮，成功后历史保留第 2～5 轮；第 6 次推理据此继续滚动。滑动不会清空历史或更换 `conversation_id`。失败、无效回复、拒绝请求及缓存重取不滑动窗口，也不增加背景计数。
- 同一对话连续两轮成功回复的文字完全相同时，下一条新提问提前开启新对话：清空推理历史并生成新的 `conversation_id`。第二次重复回复仍正常返回并保留原对话编号；只比较回复文字，表情编号不同也算重复，空白和标点不同则不算。失败、无效输出及缓存重取不增加检测次数，不同设备分别检测。此机制与滚动窗口、空闲及手动重置并存；即使显式设置 `max_turns=1`，也独立保留最近两个成功回复进行重复检测。程序调用可设置正整数 `max_turns` 调整历史窗口大小，或设置 `max_turns=None` 保留无限历史，重复保护仍生效。
- 树莓派的本地点歌由 [`meme_sing/`](../meme_sing/README.md) 实现：说“想听音乐”按最近成功聊天文字匹配歌曲情绪标签；播放中说“停止音乐”恢复聊天。点歌、停歌和播放中识别的歌词不提交模型，不占问答轮次；未播放时“停止”仍按普通聊天处理。
- v2 的最终 `result` 携带 `conversation_id` 和非负整数 `background_revision`。窗口每删除一个最旧问答对，`window_slides` 增加 1，背景版本为 `window_slides // 5`；默认第 1～8 轮回复为版本 0，第 9～13 轮为版本 1，第 14～18 轮为版本 2，第 19 轮为版本 3。滑动保持 `conversation_id`；连续重复回复后的下一条输入、10 秒空闲或手动重置才生成新编号，并把滑动计数和背景版本归零。缓存回复保留生成时的编号及背景版本，不推进当前窗口或背景。树莓派按 `(conversation_id, background_revision)` 选择背景；同一对话版本变化时从完整背景组中排除当前编号后重新随机，旧回执不会倒退当前背景。目录或文件缺失时明确报错，不回退无背景 GIF。
- 表情检索和返回的 `meme_id` 不变。树莓派画面取 `sucai_meme/meme_gif_background/<原stem>-N.gif`，配套声音仍取 `sucai_meme/meme_gif/<原stem>.wav`。欢迎画面先随机一个背景，手动播放沿用当前背景；服务器新对话的首个有效回复确定背景，后续按背景版本更新。每条已接收回复的背景保持固定。

以下到达时刻规则仅适用于未协商 v2 的旧客户端和 `pi_client.py` 调试工具：

- 用户消息到达服务器时立即记时。间隔小于 10 秒，下一次推理输入为 `A、B、A`，之后为 `A、B、A、B、A`，依次累积。
- 距上次用户消息到达时间 **大于或等于 10 秒**，清空当前历史，新消息从 A 开始。后台也会清理空闲历史。心跳、模型完成时间和发送回复不延长该期限。
- 同一会话按用户消息到达顺序推理；不同设备历史独立。旧会话尚在排队或生成的回复仍返回对应请求，但不会写回已经重置的新会话。
- 新客户端自动携带随机 `session_id`，同一实例重连保留此标识，仍按最后用户到达时间执行 10 秒规则；不同实例默认独立。需要跨进程复用时可显式指定 `session_id` 或 CLI `--session-id`。旧客户端每次连接独立，同名设备不会混用历史。
- 模型失败时本轮不提交半截 A/B 历史；认证失败、格式错误和队列满的拒绝请求不算有效用户发言，也不更新空闲计时。每会话最多 8 条处理中请求，全局最多 128 条；服务器还限制每连接最多 8 条未返回请求。

旧协议计时依据用户消息到达服务器，因此旧调试客户端在生成超过 10 秒后发下一句会开启新对话；实际树莓派 v2 播放器使用上面的活动暂停规则。

## v2 握手帧与更新树莓派

认证仍复用现有口令，每行一个 JSON。省略认证字段值的流程如下：

```text
Pi → hello(protocol_version=2, session_id, token)
服务器 → hello_ok(model_ready=true, model_version, server_instance_id, max_inflight=1, max_turns=4, context_window_turns=4, background_roll_interval=5, capabilities包含conversation_id/sliding_context/background_revision)
Pi → speech_start
Pi → request(id, text, end=true)
服务器 → accepted(id, status=processing)
服务器 → result(id, reply_text, meme_id, conversation_id, background_revision)
Pi → playback_done(id)
```

控制消息返回独立 `control_ack`；重连查询使用 `request_status(id)`，返回 `accepted` 或原 `result/error`。同一个请求编号和相同正文重复到达不会重新推理；同编号不同正文返回 `request_conflict`。没有 `end:true` 的请求被拒绝，不进行片段拼接，也不上传原始音频。

两端源码应一起更新。部署工具先核对安装状态，再备份并更新白名单源码；启动命令与现有口令保持不变。它保留素材、离线模型和运行配置，并核对修改前后的 SHA。只重启指定安装目录中的播放器及其 STT，不终止其他应用。

```bash
# 只读检查；不部署、不启动软件。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.deploy_pi --check

# 部署经过联测的完整核心源码，并启动/重启既有树莓派播放器。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.deploy_pi --apply --restart \
  --report model_information/runs/pi_handshake_deployment.json
```

私有备份在树莓派 `~/.local/state/meme-model/deployments/`。结构化报告记录备份路径、实际变更和进程状态，不含口令。完整字段见 [树莓派协议说明](../meme_software/player/desktop/SERVER_PROTOCOL.md)。

## 回复与表情选择

服务通过 `EmotionalChatEngine` 复用归档 `PaperMemeEngine` 加载的本地模型、情绪规划器和回复向量库。根据当前有效用户文本和上下文规划目标情绪向量，再通过训练好的 prefix 引导 Mistral 单次生成最终回复。将这条实际回复连同上下文编码为 256 维情绪向量，使用与训练回复库相同的编码方式，按余弦距离查找最近训练回复，返回该项标注对应的 `meme_id`。`reply` 保留本轮生成的文字，不替换成检索到的训练回复。

传输格式与原通信代码一致，为 UTF-8、每行一条 JSON。v2 请求和结果示例：

```json
{"type":"request","id":"r1","text":"今天很开心","end":true}
{"type":"result","id":"r1","text":"{\"reply\":\"真替你开心\",\"meme_id\":\"meme-0008\"}","reply":"真替你开心","reply_text":"真替你开心","meme_id":"meme-0008","conversation_id":"9f1d2a83b6c544e8a7f1230984dcba65","background_revision":0}
```

上例展示格式，实际正文和表情编号由模型推理产生，`conversation_id` 和 `background_revision` 由会话管理器生成。`reply_text` 与 `reply` 相同；v2 握手的 `capabilities` 包含 `conversation_id`、`sliding_context` 和 `background_revision`。`max_turns` 兼容字段与 `context_window_turns` 都表示历史窗口大小，`background_roll_interval=5` 表示每滑动 5 个完整问答对更新一次背景。旧协议和 `pi_client.py` 调试客户端仍只返回 `reply` 和 `meme_id` 两字段。连接开始时仍需发送包含共享口令的 `hello`；业务客户端已封装握手、请求编号匹配与心跳。

## 语音待机与唤醒

共享树莓派播放器同时支持“小猫”和“小me”称呼。待机指令包括“小猫，睡觉吧”“小猫，睡觉”“小猫，睡觉了”“小me，睡觉吧”；唤醒指令包括“小猫，起床吧”“小猫，起床”“小猫，起床了”“小me，起床吧”。原短句、“别睡了”及有限同音别名“小咪”“小米”“小美”等继续兼容，标点、空格和大小写按规则规范化。清醒时的睡觉指令进入待机；待机时的起床指令才触发唤醒，睡觉指令不会唤醒，清醒时的起床语句仍作为普通聊天。待机中的控制在树莓派本地处理，不提交模型、不滑动历史窗口；引用、否定、条件句和问句不触发。语音转写仍使用已有离线 ASR，解析测试不替代现场麦克风验收。详见 [`meme_huanxing/README.md`](../meme_huanxing/README.md)。

## 文件与验证

| 文件 | 功能 |
| --- | --- |
| `start_server.sh`、`launcher.py` | 服务器一键启动，准备网络并按需启动树莓派软件 |
| `pi_startup.py` | 通过 SSH 执行的树莓派软件状态检查及桌面启动逻辑 |
| `deploy_pi.py` | 白名单源码备份、原子部署、SHA 核验和指定播放器重启 |
| `server.py`、`run.sh` | 低层模型通信服务，接收文字并返回推理结果 |
| `model.py` | 校验归档并加载本地模型；服务使用 `load_chat_engine()` |
| `chat_engine.py`、`generation.py` | 训练格式回复、可选指令校对、共享权重与情绪 prefix 生成 |
| `normalization.py`、`phonetics.py` | 输入规范化及可选独立 STT 校对的保护检查 |
| `role_names.py` | meme 称呼及近音名归一，保护宠物、引用与其他实体 |
| `prompts/`、`chat_config.json` | meme 角色、校对提示词和推理参数 |
| `sessions.py` | 管理历史、活动暂停、10 秒空闲重置、排队及设备隔离；默认滚动保留最近 4 个完整成功问答对 |
| `protocol.py` | 请求及 `reply` / `meme_id` 格式校验，兼容旧客户端 |
| `request_cache.py` | v2 请求状态、结果缓存、去重及播放完成租约 |
| `dialogue_log.py`、`console.py` | 私有双向对话日志与当前终端实时查看，支持复用后台模型 |
| `pi_client.py` | 树莓派业务客户端，返回两字段字典 |
| `tests/` | 会话、协议、真实本机 TCP 和模型加载测试 |
| `tools/verify_integration.py` | 本机 TCP 与真实归档模型的完整联调 |
| `tools/verify_pi_handshake.py` | SSH 读取实际握手/麦克风状态，或执行一次受控播放、Pi 到服务器的真实请求 |
| `tools/verify_reply_quality.py` | 真实回复的长度、规则复述、调用次数及人工语义审阅回归 |
| `tools/evaluate_asr_repair.py` | 临时启用独立校对，评估受控样本及三种回复基线 |

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m unittest discover -s model_information/tests -v

# 无需树莓派硬件或模型权重：握手、停麦与播放状态检查。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m unittest discover -s meme_software/player/desktop/tests -v

# 只读查看实际树莓派的就绪、收音、播放状态；不发模型请求或播放。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.tools.verify_pi_handshake --check

# 会真实加载本地模型；使用临时口令及本机随机端口，不占用 9000。
# 用可控单调时钟模拟消息到达间隔 4 秒和 10 秒，验证历史长度 1、3、1。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.tools.verify_integration
```

联调检查真实到达时刻对应的历史长度、有效用户文本、每轮模型生成调用次数，以及实际回复向量的最近邻结果。默认每轮校对调用为 0、情绪条件回复生成为 1。启用可选独立校对时，工具按实际模式核验调用次数。启用采样时回复正文会随运行变化。每轮记录本机端到端耗时，其中包含通信、模型处理和联调校验开销，不代表树莓派语音识别、播放或公网网络耗时。

联调结果保存在 `model_information/runs/integration_chat_smoke.json`。该目录被 Git 忽略；测试不修改训练参数或原 EmoStance 目录。旧报告保留其生成时的行为，更新代码后需重新运行联调才能取得当前单回复流程的结果。

真实回复质量回归覆盖身份、问候、情绪、模糊请求、上下文指代和短确认，默认每例使用三个随机种子。报告核对 40 字要求、明确规则复述、一次真实生成与 prefix 调用，以及原始输出和返回文字的一致性；语义是否贴切另按每例审阅标准检查。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.tools.verify_reply_quality
```

报告默认保存到 `model_information/runs/reply_quality.json`。其中 `mechanical_checks_passed` 表示上述可自动核验的条件通过，`semantic_review=pending` 提醒继续检查跑题、身份混淆和虚构操作等实际回复问题。工具只记录和判定，不截断回复、不删除提示片段，也不追加一次模型生成。

单独评估可选校对能力：

```bash
# 会加载真实模型，只在这个评估进程中临时启用独立校对。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m model_information.tools.evaluate_asr_repair --compare-replies
```

该工具记录校对开关的临时覆盖，结束后恢复进程内原值，不修改 `chat_config.json`，也不会恢复线上服务的独立校对。受控样本结果不代表真实树莓派语音识别成功率；回复对比使用缓存的校对结果，其 `reply_pipeline_seconds` 不含单独记录的校对耗时。
