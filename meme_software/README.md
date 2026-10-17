# Meme猫软件工作区

树莓派Python/Tk原生桌面播放器：麦克风 → 本地中文STT → 服务器收文字 → 返回reply_text/meme_id → 播放本地表情与配套WAV，底部显示回复字幕。主画面不显示用户转写、素材名或时长。

**当前语音模型为第一版（V1）：原中文轻量lite＋原ASR静音断句，默认停顿0.5秒。** 详见 [STT_MODEL_VERSION.md](STT_MODEL_VERSION.md)。本项目V1是交付标记，不是新训练模型。

## 当前交互

播放表情和声音时继续收音，用户说完后才发最终句；新服务器回复先到也等旧GIF及配套声音都完整结束再播。旧播放完成不重置正在说的半句。一次只允许一个待确认请求；等待服务器或回复待播时暂停收音，开始新播放时恢复。动态图标显示“边播边听”。不使用Silero/Smart Turn，不生成语音。

## 关键代码

| 路径 | 用途 |
| --- | --- |
| player/desktop/app.py | 原生界面、收音状态、播放与待播协调 |
| player/desktop/conversation_turn.py | 对话/媒体状态、整句提交与旧媒体完成屏障 |
| player/desktop/stt_worker.py、capture_turn.py、stt_bridge.py | 常驻中文lite模型、采集、静音端点及IPC |
| player/desktop/asr_runtime/ | 离线识别与连续PCM采集核心 |
| player/desktop/server_link.py、SERVER_PROTOCOL.md | TCP文字及回复编号协议、重连、重复/过期校验 |
| player/desktop/directory_library.py | 精确按meme编号读取本地GIF及同名WAV |
| player/desktop/render_pipeline.py、native_core.py | 后台解码、有限缓存、实际配套声音播放 |
| player/desktop/start.sh | Pi图形桌面启动，端口8784、静音阈值0.5秒 |
| player/launch.py | 共用存储与本机HTTP接口 |

## 在已有树莓派环境运行

目标已核对为Pi4B / Debian12 ARM64 / CPython3.11，使用已安装的Tk/Pillow/ImageTk/ALSA。复用已离线安装的Sherpa1.13.8、NumPy1.26.4及中文lite模型；本源码不会在线安装。源码不含模型、媒体、数据库或venv，不能当成全新设备的完整离线安装包。

1. 保留原player/data及数据库，修改前备份；退出旧播放器和独立ASR，只运行一个收音实例。
2. 素材默认只读读取仓库 `sucai_meme/meme_gif` 下 `meme-四位编号-名称.gif` 及同名WAV；按完整编号映射，不排序重编号，不回退旧SQLite同编号。未知或已剔除编号不播放，不修改队友素材。
3. 从Pi自身图形桌面终端，在本meme_software目录运行：

```sh
sh player/desktop/start.sh --stt-root /path/to/existing/offline-asr
```

将路径换为已有ASR根（含.venv/bin/python和lite模型）。F11全屏、Esc退全屏，底部“退出”。依赖检查可用 `python3 player/desktop/app.py --check`。

4. 在“语音”填写真实服务器地址和现有口令文件路径；源码127.0.0.1:9000是占位，原Pi运行配置保持不变。配置保存在被忽略的player/data/speech-config.json，格式见speech-config.example.json。口令只从ZT_CHAT_TOKEN或已有.env读取，不执行.env、不上传凭据。

服务器按SERVER_PROTOCOL.md返回同request.id的一次最终result，带reply_text及meme_id（也兼容result.text内JSON字符串）。本地WAV是表情原配声音，不保证读出服务器文字。普通回显不播放，断线句不排队、未确认请求不自动重发。

## 验证与限制

15项并发/原采集行为检查、真实Windows Tk＋本机TCP两轮检查通过；麦克风和声音时长为模拟。已观察Pi更新器成功，真实Pi新界面、同时收音/播音、服务器回复及回灌仍需现场验证。未实现声学回声消除，不能保证麦克风区分用户与扬声器。

帧解码/缩放在后台，视觉帧落后可跳过，不丢弃用户收音PCM；主机测试不是Pi/TFT帧率保证。历史检查范围见player/desktop/VALIDATION.json。只维护本meme_software目录，队友文件保持原样。

在本目录执行 `sha256sum -c SHA256SUMS` 校验本次提交文件。已有素材与模型的许可不因代码提交而改变。
