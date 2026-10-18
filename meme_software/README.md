# Meme猫软件关键源码

2026-10-18从树莓派当前运行目录只读下载的源码快照，包含播放器、本地STT、协议2客户端、音乐控制、语音待机/唤醒和USB音频恢复。发布副本将默认服务器改为localhost占位并清理机器绝对路径，不修改树莓派实际设置。

## 关键模块

| 路径 | 职责 |
| --- | --- |
| player/desktop/app.py | 原生Tk界面、全屏触摸退出、白字透灰字幕、模式协调 |
| player/desktop/server_link.py、SERVER_PROTOCOL.md | 协议2就绪握手、单请求、断线状态查询与播放回执 |
| player/desktop/conversation_turn.py | 当前Pi的严格半双工轮次，播放结束后恢复普通聊天收音 |
| player/desktop/stt_worker.py、stt_bridge.py、capture_turn.py | 常驻离线ASR、完整句与进程/麦克风生命周期 |
| player/desktop/asr_runtime/、asr_selection.py | sherpa-onnx解码、连续PCM、显式本地模型选择 |
| player/desktop/meme_sing/ | 本地音乐请求、情绪标签选曲、播放及停止控制 |
| player/desktop/meme_huanxing/ | 有称呼的睡眠/唤醒规则、自动待机与完整起床动画 |
| player/desktop/voice_recovery.py、audio_output.py | USB输入掉线恢复及稳定USB输出路由 |
| player/desktop/render_pipeline.py、native_core.py | 后台GIF解码、有界缓存、配套WAV播放 |
| player/desktop/directory_library.py | 按meme编号只读匹配GIF与同名WAV |
| player/desktop/cursor_autohide.py | 鼠标3秒隐藏、点击恢复 |
| player/launch.py | 既有素材存储及本机HTTP播放接口 |

## 当前运行方案

- STT：sherpa-onnx 1.13.8，`sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30` Transducer，CPU 2线程。不是4月Small CTC，也不是Paraformer。
- 当前Pi启动脚本结束静音为0.7秒，可通过参数调整；它是声学静音门槛，不是语义理解或总延时保证。
- 协议2收到服务器就绪才开麦；普通聊天在推理、准备和表情/实际声音播放期间暂停，完成后恢复。此处如实保留当前Pi实现，不套用旧版边播边听说明。
- 点歌、停止和语音待机/唤醒由本地规则处理，不生成TTS。音乐/待机控制窗口不把歌词或普通待机话语发送服务器。
- 自动待机仍为20秒，点击或当前有称呼的唤醒语句播放完整醒来片段。
- 只发文字；不默认保存/上传音频。没有新增声学回声消除或新训练模型。

## 在既有离线环境运行

本提交是源码，不包含模型、音乐/表情素材、数据库、venv、wheel、凭据或个人运行配置，不能当成全新设备离线安装包。保留现有`player/data`，使用已安装的Tk/Pillow/ALSA、NumPy及sherpa-onnx。音乐功能还需要已有`ffplay`；USB输出恢复使用现有用户音频服务。

在Pi图形桌面终端、meme_software目录运行：

```sh
sh player/desktop/start.sh --stt-root /path/to/existing/offline-asr --meme-dir /path/to/sucai_meme/meme_gif --music-dir /path/to/sucai_meme/meme_music
```

模型路径由`player/desktop/asr-selection.json`指定，本地应已存在对应四个文件。服务器localhost只是源码占位，请沿用本机配置和ZT_CHAT_TOKEN文件，不能把令牌写进代码或提交Git。协议说明见[SERVER_PROTOCOL.md](player/desktop/SERVER_PROTOCOL.md)。

原仓库已有的sleep_screen.py、sleep_demo.py和sleep.gif/sleep.wav保留；当前主程序使用meme_huanxing/screen.py，不新增或替换原媒体。

发布与验证边界见[CURRENT_SOURCE.md](CURRENT_SOURCE.md)，模型说明见[STT_MODEL_VERSION.md](STT_MODEL_VERSION.md)。源码校验：`sha256sum -c SHA256SUMS`。


## 2026-10-18 核心源码更新

服务器统一使用最近四个成功问答对的滚动上下文，窗口每滚动五次推进背景版本；无其他重置时，第9、14、19次成功回复切换背景。播放器按 `(conversation_id, background_revision)` 固定每次播放的背景，不因旧缓存回执倒退；详见 [SERVER_PROTOCOL.md](player/desktop/SERVER_PROTOCOL.md)。

待机新增“小猫，睡觉吧”“小猫，睡觉。”“小猫，睡觉了。”“小me，睡觉吧。”；待机中的唤醒支持对应“小猫，起床吧”“小猫，起床。”“小猫，起床了。”“小me，起床吧。”。旧“小me，睡觉/起床”继续可用。待机中的睡觉指令继续待机，清醒时的起床语句交给普通聊天。根目录与播放器内嵌的 `meme_huanxing`、`meme_sing` 源码同步发布。

本次仅更新源码、说明与回归测试，未启动或替换树莓派运行程序。`VALIDATION.json` 中原有设备下载和10项逻辑测试属于此前快照；不代表本次重新测试麦克风、扬声器或真实模型回复。ASR选择文件是本机配置，另行保留，不随核心源码提交。总项目安装、模型产物前提和三种服务器启动方式见 [项目README](../README.md)。
