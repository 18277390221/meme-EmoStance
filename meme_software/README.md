# Meme猫软件工作区

Python + Tkinter 原生树莓派桌面程序：本地中文STT最终句 → TCP服务器 → 回复文字和表情编号 → 本地表情、配套WAV和底部字幕。主画面左上角为真实收音/发送/异常状态，不显示用户转写、素材名、时长或计时。

## 关键文件

| 路径 | 作用 |
| --- | --- |
| player/desktop/app.py | 原生窗口、素材管理、语音设置与播放协调 |
| player/desktop/stt_bridge.py、stt_worker.py | 本地识别子进程与最终句IPC |
| player/desktop/asr_runtime/ | 嵌入的Sherpa识别和连续PCM采集核心 |
| player/desktop/server_link.py | JSONL TCP认证、文字收发、重连及回复校验 |
| player/desktop/SERVER_PROTOCOL.md | 服务器reply_text/meme_id接口约定 |
| player/desktop/render_pipeline.py | 后台GIF解码/缩放、有界缓存和声音时钟调度 |
| player/desktop/native_core.py | 配套WAV播放及媒体校验 |
| player/desktop/diagnose_display.py | 只读屏幕驱动/桌面模式诊断 |
| player/desktop/start.sh | Pi图形桌面启动入口，端口8784 |
| player/launch.py | 共用SQLite服务和本机HTTP接口 |

## 复用现有安装

这是源码，不含SQLite媒体、模型、wheel、venv或通信口令。目标为已检测的Pi4B / Debian12 ARM64 / Python3.11，已装Tk/Pillow/ImageTk/ALSA。STT复用已离线安装的Sherpa1.13.8、NumPy1.26.4及本地lite模型；源码不会在线下载依赖。不要把Windows虚拟环境拷到Pi。

1. 停止旧播放器后，备份并保留原data，将既有assets.sqlite3副本放到player/data；不要覆盖正在使用的数据库或重新导入已删除素材。
2. 从已有离线安装选择ASR根目录（含.venv/bin/python和models下的lite模型）。先退出单独运行的ASR，只保留本软件一个收音实例。
3. 从Pi自身图形桌面终端、在meme_software目录运行：

```sh
python3 player/desktop/app.py --check
sh player/desktop/start.sh --stt-root /path/to/existing/offline-asr
```

将示例路径换为实际目录。普通SSH会话通常没有DISPLAY，不能据此重装Tk。F11全屏/Esc退全屏，底部按钮退出。

4. 在“语音”设置填写真实服务器地址、9000端口和已有口令文件路径。**源码的127.0.0.1是占位地址**，实际部署配置保存在被Git忽略的player/data/speech-config.json。speech-config.example.json仅作格式参考。口令从ZT_CHAT_TOKEN环境变量或已有.env读取，不执行.env，不上传口令。

服务器需启动监听并由算法回调按SERVER_PROTOCOL.md返回一次最终result。返回meme_id必须在实际本机素材库中，reply_text为1～2000字。播放器选本地表情及原配声音，不生成语音；声音内容不保证等于回复文字。普通确认、重复/无关/超时ID不驱动播放。断线时新句不排队，未确认句不自动重发，重连后重说。

## 验证与边界

- Windows主机：15项TCP/IPC/回复检查、6项渲染调度/缓存检查、合成服务→原生GUI→底部字幕/配套WAV选择通过，声音检查为静音。现有211套GIF/WAV解码和ID检查通过，媒体不随源码上传。
- Pi：用户本人执行STT、回复和渲染升级，助手观察到安装器成功及备份；用户报告表情和配套声音工作，但曾普遍卡顿。最新渲染优化尚待Pi重启对照，不能把主机缓存速度当Pi/TFT实测帧率。
- 图片解码/缩放在后台，画布复用，Pillow/Tk缓存分别最多32MiB；迟到时只跳过旧视觉帧，不丢弃用户收音PCM。许多GIF源约10fps，屏幕实际刷新率仍需核对。
- 真实服务器算法/情绪匹配、唤醒与声学回声消除尚未完成或验收；场景按钮只是人工演示。扬声器可能回灌麦克风，需现场检查。
- 只维护本软件工作区，队友communication和其他目录保持原样。修改素材前备份data。

在本目录执行sha256sum -c SHA256SUMS校验本次源码；具体历史检查范围见player/desktop/VALIDATION.json。软件/依赖/模型各许可与来源沿用各项目，媒体与模型未公开上传。
