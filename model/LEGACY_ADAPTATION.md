# 本地 EmoStance 思想适配：回复生成与 meme 选择

本目录实现接收用户单句输入、构建多轮上下文、生成新回复，并返回对应表情包编号：

```json
{"reply": "模型生成的回复", "meme_id": "meme-0062"}
```

所有新增代码、配置、数据预处理结果和训练产物均放在 `model/`。原始 EmoStance 目录只作为方法参考，不修改、不覆盖。本次只实现代码与功能测试，尚未训练模型，也没有可用的训练后权重。

## 方法与原论文的关系

参考 [EmoStance 论文](https://arxiv.org/abs/2609.02133)及[官方代码](https://github.com/18277390221/EmoStance)，保留“弱标注空间 → 回复侧取向预测 → 原型重构 → 连续 prefix 控制冻结语言模型”的思想。

本项目的流程为：

1. 先按 `dialogue_id` 划分数据。只用训练集的真实发言和 meme 标注构建情绪取向空间。
2. 用现有本地 Mistral 的冻结特征计算置信度加权 meme 质心，再结合质心相似度和共选关系构图，得到软区域分布及连续原型。meme ID 是不含语义的索引，不把标注 `reason` 当作特征或回复。
3. 学习“带角色的历史上下文 → 下一条助手回复的软取向”，结合用户到助手的取向转移先验，重构回复控制向量。
4. 冻结 Mistral 基座，只训练取向预测头和连续 prefix 投影器。语言模型损失只计算回复及 EOS，忽略上下文和 prefix 的标签。
5. 推理时由上下文预测控制向量，使用 prefix 引导 Mistral **生成新的回复**。再对生成的回复计算情绪取向向量，在训练回复库中取距离最近的一项，返回该项对应的 `meme_id`。回复文本始终使用新生成的文本。

默认距离为余弦距离 `1 - cosine`，可配置为欧氏距离。距离相同时按 ID、回复和轮次确定性排序。检索库只含训练集回复，验证集和测试集不会进入库中。

这是面向中文 meme 数据的适配，而非原论文逐项复现：复用本地 Mistral 特征，避免额外下载 DeBERTa；采用谱聚类构建软区域，并缩小 prefix 投影器隐层。原论文没有直接输出 `meme_id`，最近回复检索与 10 秒会话管理是本项目新增功能。A、B 双方所有带标注的发言都会用于训练句子取向评分器，避免仅学习用户情绪后直接用于评估助手回复。

## 目录与环境

```text
model/
├── config.json                 # 数据、基座、训练和推理配置
├── run.sh                      # 命令入口
├── pyproject.toml              # Python 依赖和包定义
├── src/emostance_meme/
│   ├── data.py                 # 文本/标注合并、完整历史、对话级划分
│   ├── space.py                # 训练集弱监督空间、软区域和原型
│   ├── backbone.py             # 本地 Mistral 加载、特征和生成
│   ├── networks.py             # 取向预测头、连续 prefix 和回复损失
│   ├── training.py             # 预处理、两阶段训练、检索库与产物保存
│   ├── retrieval.py            # 最近回复检索
│   ├── engine.py               # 模型推理，返回 reply + meme_id
│   ├── session.py              # 多轮历史、10 秒过期、并发和重置
│   ├── service.py              # Python 业务接口
│   ├── http_server.py          # HTTP 接口
│   └── cli.py                  # doctor / prepare / train / chat / serve
├── tests/                      # 无需训练真实模型的功能测试
└── runs/                       # 运行产物，已在本目录 .gitignore 中排除
```

默认直接读取现有基座：

```text
/home/aida102/emoji/Mistral-7B-Instruct-v0.3
```

模型和 tokenizer 均使用本地绝对路径与 `local_files_only=True`；权重缺失时直接报错，不联网下载。`run.sh` 默认使用已有 `/home/aida102/emoji/.venv/bin/python`，不会修改该环境或 EmoStance 原目录。需要换解释器时设置 `EMOSTANCE_PYTHON`。

在项目根目录检查环境，不加载 7B 权重、不训练：

```bash
bash model/run.sh doctor
```

如需独立安装环境，可自行在 `model/.venv` 中安装 `model/pyproject.toml` 声明的依赖，再指定该解释器。训练及 7B 推理应在有足够内存的主机上运行。

## 数据格式

推荐 JSON 数组或 JSONL，每个对象是一段完整对话：

```json
{
  "dialogue_id": "example-001",
  "source_split": "train",
  "annotator": "paper1",
  "messages": [
    {"turn_id": 0, "role": "A", "text": "今天终于完成任务了", "meme_id": "meme-0008", "confidence": 4},
    {"turn_id": 1, "role": "B", "text": "辛苦啦，给你一个么么哒", "meme_id": "meme-0062", "confidence": 4},
    {"turn_id": 2, "role": "A", "text": "接下来想好好休息", "meme_id": "meme-0002", "confidence": 3},
    {"turn_id": 3, "role": "B", "text": "那就放松一下吧", "meme_id": "meme-0008", "confidence": 4}
  ]
}
```

这只是格式示例，不是已经收集好的训练数据。支持 `user/assistant` 角色与 `content` 字段。轮次从 0 开始，要求 A/B 交替且原文完整。第二条 B 回复的输入会包含前面 A、B、A 三条消息，而不是只看最后一句。

你提供的原始 `turns` 格式可以保留模型、提示词哈希等元数据；这些元数据不参与训练。但若 `turns` 只有 `turn_id/meme_id/confidence`，还必须提供对话原文文件，通过 `dialogue_id + turn_id` 合并，例如：

```json
{
  "dialogue_id": "example-001",
  "messages": [
    {"turn_id": 0, "role": "A", "text": "今天终于完成任务了"},
    {"turn_id": 1, "role": "B", "text": "辛苦啦，给你一个么么哒"}
  ]
}
```

还支持此前逐条的 `text + context + meme_id` 格式。缺失原文会明确报错，不根据 `reason` 编造原文。中文字符间的分词空格会合并。

同一轮多个标注者的选择按置信度汇总为软分布，同一标注者的重复记录去重。单条标注对应 one-hot 标签，其置信度仍用于质心和损失权重。`max_confidence` 默认 5，应按你的实际标注标尺修改；3、4 不是概率。显式 `source_split=dev/test` 的对话不会变成训练数据；未指定划分时使用可复现的对话 ID 哈希划分。

## 数据准备好后的操作

先编辑 `model/config.json`，设置数据路径。相对路径以 `model/` 为基准，可以传入具体文件或只包含输入数据的文件夹：

```json
{
  "annotations": ["../dataset/annotations.jsonl"],
  "texts": ["../dataset/dialogues.jsonl"]
}
```

上面是需要修改的配置片段；保留原配置中的其他字段。若标注数据本身已经包含完整原文，`texts` 设为 `[]`。已有 train/dev/test 文件可一并放入 `annotations`，但其中的记录需要正确的 `source_split`，不能仅依赖文件名表达划分。

```bash
# 合并、检查和划分数据；不运行模型训练
bash model/run.sh prepare

# 数据确认完整后，再手动执行训练
bash model/run.sh train
```

预处理产物在 `model/runs/main/prepared/`。完成训练后得到：

- `space.json`：训练集情绪质心、软区域和原型。
- `adapters.pt`：取向预测头、转移先验和 prefix 投影器权重；不复制基座权重。
- `responses.json`：仅由训练集回复构成的情绪向量检索库。
- `manifest.json`：完成状态、配置、输入文件哈希和训练日志；训练全部完成后才写入。

未完成的训练目录不能用于推理。已有完成产物的目录不会被训练命令覆盖；要重新训练，请修改 `run_dir`。没有足够训练文本或仅有 dev/test 记录时会报错。没有标注的助手轮次可保留为上下文，但不会伪造表情监督用于当前训练流程。验证集未见过的 meme 会计入 `dev_unseen_meme_count`，不参与取向头的验证损失，避免伪造训练区域。

## 推理与 10 秒上下文

训练完成后启动终端对话：

```bash
bash model/run.sh chat
```

支持 `/reset` 手动重置、`/quit` 退出。每个会话累计：

```text
用户问题1 → 助手回复1 → 用户问题2 → 助手回复2 → 当前用户输入
```

最后一次用户输入接收后，若连续 **满 10 秒**没有新输入，则清空该会话。时间使用单调时钟，不按模型回复结束时间重新计时。已经接受的请求按顺序完成；超时或手动重置后，旧请求结果不会把旧历史写回新会话。失败请求不会留下没有回复的半轮记录。

内存中保留该会话完整历史；发送给模型的 token 预算默认 1024，超长时先删除最旧的完整用户/助手对，必要时仅裁剪当前用户内容并保留完整聊天模板。可通过 `max_context_tokens` 调整预算。进程退出后内存会话清空。

Python 接口（在 `PYTHONPATH=model/src` 的环境中）：

```python
from emostance_meme.engine import MemeEngine
from emostance_meme.service import ChatService

engine = MemeEngine("runs/main")
with ChatService(engine, idle_timeout=10.0, cleanup_interval=0.25) as service:
    result = service.reply("今天有点累", session_id="raspberrypi-1")
    print(result)  # {"reply": "...", "meme_id": "meme-...."}
```

HTTP 接口（训练完成后）：

```bash
bash model/run.sh serve --host 127.0.0.1 --port 8790

curl http://127.0.0.1:8790/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"raspberrypi-1","text":"今天有点累"}'
```

`POST /chat` 返回 `reply` 和 `meme_id`；`POST /reset` 接收 `{"session_id":"raspberrypi-1"}`；`GET /health` 检查服务。不同用户/设备使用不同的 `session_id`。使用多个服务进程时，会话不跨进程共享，当前实现默认单进程。HTTP 模块不自动调用或修改播放器、STT、ZeroTier 通信代码。

## 验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=model/src \
  /home/aida102/emoji/.venv/bin/python -m unittest discover -s model/tests -v
```

测试覆盖数据文本合并与划分、置信度聚合、弱标注空间、情绪距离、连续 prefix 的梯度与损失掩码、微型随机 Mistral 的生成接口、生成回复与检索回复的区分、10 秒边界、并发和真实本机 HTTP 请求。测试不训练本地 7B 模型，不下载权重；通过功能测试不代表回复质量或表情选择准确率已经验收。
