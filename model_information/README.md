# 树莓派文字与模型推理通信

树莓派发送一次文字，服务器生成一次回复，并返回 `{"reply":"回复文字","meme_id":"meme-0008"}`。本目录复用 `communication/common.py` 的 TCP JSON 协议、认证和心跳；新增模型加载、对话管理和业务客户端。

## 服务器一键启动

当前设备已经配置好 ZeroTier 和共享口令。只需在服务器执行以下命令，之后每次启动都用同一条命令：

```bash
bash /home/aida102/hks/model_information/start_server.sh
```

脚本自动读取既有 `zerotier_chat/network.env`，使用服务器 `10.205.84.185:9000` 和树莓派 `10.205.84.95`，复用服务器 `zerotier_chat/.env` 的原口令。无需再运行 `init_config.py`，无需重新填写地址，也无需到树莓派终端启动通信客户端。

启动时检查端口与本地模型环境，准备既有 WSL 镜像网络的树莓派专用路由，再通过已经配置的 SSH 密钥检查树莓派软件：运行中就保持原样；未运行则在已登录的树莓派桌面上启动原软件及其离线识别模块。缺少桌面会话时会明确提示。不会替换树莓派软件、素材、模型或口令文件。

树莓派实际安装的软件已经集成 STT、ZeroTier 连接和播放接口。服务器启动完成后，它会自动连接；对麦克风说话即可走“语音识别 → 模型回复 → 表情/配套声音/字幕”。请保持树莓派开机、桌面已登录，两端 ZeroTier 在线。

启动后保留服务器终端，按 **Ctrl+C** 停止模型服务；树莓派软件保持运行，等待下次连接。重复运行脚本会提示服务已经启动或正在加载，避免重复占用显存。若其他旧服务占用 9000，脚本会提示在原终端退出后重试。

服务器终端实时打印接收到的树莓派文字，以及发送回树莓派的回复和表情编号。每行带时间、设备名称和请求编号，方便对应同一次请求。例如：

```text
[2026-10-17T15:00:00+08:00] [树莓派 → 服务器] {"device": "meme-player", "request_id": "r1", "text": "今天很开心"}
[2026-10-17T15:00:02+08:00] [服务器 → 树莓派] {"device": "meme-player", "request_id": "r1", "reply": "真替你开心", "meme_id": "meme-0008"}
```

此处为格式示例，实际回复由模型生成。回复日志在网络发送完成后打印；发送失败会显示“发送失败”，模型处理失败会打印错误状态。心跳和认证口令不打印。正在运行的旧服务需退出后重新运行启动脚本，才会使用新增日志。

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

角色指令不会作为 A/B 历史。只有最终采用的用户文本与最终回复组成历史，10 秒仍从用户原始消息到达服务器时计时。模型处理不会延长会话；旧会话的迟到结果也不会写入新会话。

独立校对保留为可选功能：将 `chat_config.json` 的 `asr_enabled` 改为 `true` 并重启后，会先用同一份 Mistral 做一次确定性校对，再按校对结果处理。该步骤不带情绪 prefix，继续使用官方聊天模板和 `prompts/asr_repair_zh.txt` 的校对系统提示。校对结果仍需通过数字、否定、人物、情绪强度及拼音保护检查，失败时保留名字归一后的原文。默认关闭此功能。

带 prefix 的回复沿用归档训练输入上限（当前为 384 token），并受 `max_prompt_tokens` 和模型剩余上下文预算共同限制。预算不足时先裁去最旧的完整 A/B 对，保留当前用户原文及末尾 `B:`；当前输入单独超限时明确提示分短句发送，不静默截断。多轮历史可能因 token 预算提前裁剪，5 轮仍是会话重置上限。可选独立校对使用自己的官方聊天模板路径。

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

## 对话历史、5 轮上限与 10 秒规则

- 一条成功接收的 `request.text` 就是一次用户发言，即 A；服务器回复为 B。文字中的换行或多句话不会拆分。
- 一次成功的用户输入与模型回复（A/B 对）计为一轮。每段最多保留 5 轮；第 6 条用户消息开启新对话，不携带前 5 轮历史，第 7 条接续新段。排队请求也遵守该边界，失败和无效回复不计入完成轮数。
- 用户消息到达服务器时立即记时。间隔小于 10 秒，下一次推理输入为 `A、B、A`，之后为 `A、B、A、B、A`，依次累积。
- 距上次用户消息到达时间 **大于或等于 10 秒**，清空当前历史，新消息从 A 开始。后台也会清理空闲历史。心跳、模型完成时间和发送回复不延长该期限。
- 同一会话按用户消息到达顺序推理；不同设备历史独立。旧会话尚在排队或生成的回复仍返回对应请求，但不会写回已经重置的新会话。
- 新客户端自动携带随机 `session_id`，同一实例重连保留此标识，仍按最后用户到达时间执行 10 秒规则；不同实例默认独立。需要跨进程复用时可显式指定 `session_id` 或 CLI `--session-id`。旧客户端每次连接独立，同名设备不会混用历史。
- 模型失败时本轮不提交半截 A/B 历史；认证失败、格式错误和队列满的拒绝请求不算有效用户发言，也不更新空闲计时。每会话最多 8 条处理中请求，全局最多 128 条；服务器还限制每连接最多 8 条未返回请求。

计时依据用户消息到达服务器，不是回复到达树莓派的时间。因此，模型本轮生成超过 10 秒且期间没有新用户消息时，下一次发送会开启新对话。

## 回复与表情选择

服务通过 `EmotionalChatEngine` 复用归档 `PaperMemeEngine` 加载的本地模型、情绪规划器和回复向量库。根据当前有效用户文本和上下文规划目标情绪向量，再通过训练好的 prefix 引导 Mistral 单次生成最终回复。将这条实际回复连同上下文编码为 256 维情绪向量，使用与训练回复库相同的编码方式，按余弦距离查找最近训练回复，返回该项标注对应的 `meme_id`。`reply` 保留本轮生成的文字，不替换成检索到的训练回复。

传输格式与原通信代码一致，为 UTF-8、每行一条 JSON。请求和结果示例：

```json
{"type":"request","id":"r1","text":"今天很开心"}
{"type":"result","id":"r1","text":"{\"reply\":\"真替你开心\",\"meme_id\":\"meme-0008\"}","reply":"真替你开心","reply_text":"真替你开心","meme_id":"meme-0008"}
```

上例展示格式，实际正文和编号由模型推理产生。`reply_text` 与 `reply` 相同，用于兼容树莓派现有播放器；业务客户端仍只返回 `reply` 和 `meme_id` 两字段。连接开始时仍需发送包含共享口令的 `hello`；业务客户端已封装握手、请求编号匹配与心跳。

## 文件与验证

| 文件 | 功能 |
| --- | --- |
| `start_server.sh`、`launcher.py` | 服务器一键启动，准备网络并按需启动树莓派软件 |
| `pi_startup.py` | 通过 SSH 执行的树莓派软件状态检查及桌面启动逻辑 |
| `server.py`、`run.sh` | 低层模型通信服务，接收文字并返回推理结果 |
| `model.py` | 校验归档并加载本地模型；服务使用 `load_chat_engine()` |
| `chat_engine.py`、`generation.py` | 训练格式回复、可选指令校对、共享权重与情绪 prefix 生成 |
| `normalization.py`、`phonetics.py` | 输入规范化及可选独立 STT 校对的保护检查 |
| `role_names.py` | meme 称呼及近音名归一，保护宠物、引用与其他实体 |
| `prompts/`、`chat_config.json` | meme 角色、校对提示词和推理参数 |
| `sessions.py` | 按到达时间管理历史、5 轮上限、10 秒重置、排队及设备隔离 |
| `protocol.py` | 请求及 `reply` / `meme_id` 格式校验，兼容旧客户端 |
| `pi_client.py` | 树莓派业务客户端，返回两字段字典 |
| `tests/` | 会话、协议、真实本机 TCP 和模型加载测试 |
| `tools/verify_integration.py` | 本机 TCP 与真实归档模型的完整联调 |
| `tools/verify_reply_quality.py` | 真实回复的长度、规则复述、调用次数及人工语义审阅回归 |
| `tools/evaluate_asr_repair.py` | 临时启用独立校对，评估受控样本及三种回复基线 |

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /home/aida102/emoji/.venv/bin/python \
  -m unittest discover -s model_information/tests -v

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
