# 服务器与树莓派握手协议（版本 2）

连接地址来自现有 `player/data/speech-config.json` 的 host/port。口令沿用两端已有 `ZT_CHAT_TOKEN`，不写入源码或日志。传输为 TCP UTF-8 JSONL，每帧以换行结束，最大 1 MiB。Pi 本地完成语音识别，只发送最终文字。

## 初始化与就绪

Pi 每次程序启动生成一个 session_id，同一进程断线重连继续使用它：

```json
{"type":"hello","name":"meme-player","token":"<已有口令>","protocol_version":2,"session_id":"会话UUID"}
```

服务器完成模型初始化并通过认证后回复：

```json
{"type":"hello_ok","protocol_version":2,"session_id":"连接会话UUID","model_ready":true,"model_version":"V2.0.0","server_instance_id":"本次服务器启动UUID","max_inflight":1,"idle_timeout":10,"max_turns":8,"capabilities":["end","accepted","request_status","speech_start","speech_cancel","playback_done","playback_failed","conversation_id"]}
```

Pi 必须验证 protocol_version、session_id、model_ready、server_instance_id 和 max_inflight；就绪后才允许开麦。握手不通过或断线时立即暂停收音。ping/pong 每 15 秒维护连接，心跳不会算作用户活动。

## 一轮完整对话

1. 开麦后首次识别到非空临时文字（Paraformer也支持持续有效声音检测），Pi 发送 `{"type":"speech_start"}`。这表示用户开始讲话，不传临时文字，也不在每次开麦时发送。
2. 用户停顿达到本地端点（start.sh 默认 0.7 秒；后台播放器的旧参数须重启后更新），worker 先关闭 ALSA，再提交整句：

```json
{"type":"request","id":"本句话UUID","text":"用户识别文字","end":true}
```

3. 服务器只有验证 `end:true` 的完整请求才开始推理，先发送：

```json
{"type":"accepted","id":"本句话UUID","status":"processing"}
```

accepted 只是接收确认，Pi 保留 pending，不播放、不重新开麦。服务端一次仅允许一条未完成轮次，收到完整请求时原子登记 request_id，避免重连重复推理。

4. 推理结束返回同一个 id 的最终结果：

```json
{"type":"result","id":"本句话UUID","meme_id":"meme-0157","reply_text":"听起来你今天很开心呀","conversation_id":"9f1d2a83b6c544e8a7f1230984dcba65"}
```

也支持 result.text 为包含 meme_id/reply_text 的 JSON 字符串；conversation_id可位于result顶层。conversation_id由服务器实际使用的对话上下文生成，同段固定，完成8个成功问答后的第9条用户输入、空闲重置或显式重置后改变；它不同于可跨段重用的session_id。缓存与断线重取返回原conversation_id，迟到旧请求也不能冒用新段编号。普通回显不能替代 accepted 或 result。失败使用 `{"type":"error","id":"本句话UUID","code":"错误代码","message":"原因"}`。

5. Pi 检查本地素材编号，准备画面与配套声音；确认麦克风已关闭再播放。等待推理、准备及播放期间都暂停收音。GIF 一轮和实际声音均结束后，发送：

```json
{"type":"playback_done","id":"本句话UUID"}
```

播放准备失败、未知素材或主动取消则发送 playback_failed。两者均释放该轮播放状态；成功控制回复 `{"type":"control_ack","control":"playback_done","id":"本句话UUID"}`，重复完成消息幂等。Pi 保留完成回执直到 control_ack，断线后可以补发回执。正常播放结束后等待 250 毫秒再开麦，仍须服务器处于就绪状态。

未形成完整句子时发送 `{"type":"speech_cancel"}`；成功回复 control_ack/control:speech_cancel。连续讲话超过 25 秒仍无端点时本地取消并提示重说，不强制把半句话当最终句。服务器收音租期 30 秒，播放租期 120 秒，防止丢失取消/完成消息后永久占用。

## 断线恢复与重复结果

- Pi 暂停麦克风，保留已经提交的原请求 id；不自动重发请求正文。
- 重连握手的 server_instance_id 相同，Pi 查询 `{"type":"request_status","id":"原请求UUID"}`。仍在处理时服务器发送 accepted，并在完成后发送 result；已经完成时返回缓存中的同一 result/error。
- 同一 id 的成功结果在 Pi 只触发一次播放。收到 accepted/control_ack 不清除正在等待的请求。
- 服务端完成记录保留最多 600 秒、1024 条；待处理请求不作为完成缓存淘汰。查询返回 unknown_request 时 Pi 解除等待，明确提示用户重说，避免盲目重发造成重复回答。
- server_instance_id 改变表示服务器重启。Pi 清除旧启动实例的待处理状态，提示重说，不自动重发旧话。缓存去重保证的范围是当前服务器进程和记录保留期，不跨重启或无限期保证。

USB 录音设备失联属于 Pi 本地恢复，不增加服务器消息类型：立即停歌并关闭收音，取消未提交的收音租约；已提交请求仍按原 id 和播放回执规则处理，不自动重发正文。设备缺失时持续等待，重新出现后只重启本地识别，不重载服务器模型，也不自动续播歌曲。设备存在但恢复启动失败最多 5 次，间隔 1、2、4、8、8 秒；非设备故障不盲目重试。手动暂停、关闭语音或退出会取消恢复。重新开麦仍须服务器就绪和本轮允许收音，原 0.7 秒断句、八轮对话及 conversation_id 规则保持。

USB 输出恢复也在 Pi 本地处理：按稳定 USB sink 名称重新设置默认音箱，解除意外静音并保留已有非零音量；音乐和表情配套声音使用同一个 USB 输出。USB 未连接时等待，不把板载输出当作恢复成功。硬件已连接但系统音频尚未枚举该输出时，每次恢复最多尝试重启一次当前用户的 WirePlumber，不反复重载音频服务，不重载服务器模型。应分别验证音乐和表情声音的实际 sink 路由；ASR 已启动或播放进程已创建不能单独证明声音已恢复。

## 会话与素材

服务端每段默认最多8个成功问答，握手中的 `max_turns:8` 表示这一上限；第9条用户输入开启新对话并使用新的conversation_id，Pi据此重新随机背景。失败和无效回复不计入成功轮数。每次推理仍受既有384 token输入预算限制，预算不足时裁去最旧的完整A/B对。真正空闲满10秒也开启新对话；这10秒从真正等待用户时计算，收音、推理及播放等待期间暂停该计时，播放完成或取消后重新开始。心跳和状态查询不延长会话。

本地点歌由Pi的 `meme_sing` 处理，不增加服务器消息类型。点歌时取消已发出的speech_start租约，不发送request，不占问答轮次，也不伪造playback_done。音乐播放期间仅本地识别停止指令，歌词不发送模型；停止或自然结束后恢复聊天。未播放音乐时“停止”仍作为普通request。音乐期间服务器原10秒空闲规则保持，下一次聊天的真实conversation_id可能变化。

语音待机由Pi的 `meme_huanxing` 本地处理，不增加服务器消息类型。带有效称呼的“小me，睡觉去吧”取消未提交的收音租约，并在音乐停止、当前回复播放结束和麦克风关闭确认后进入待机。待机仅本地识别“小me，别睡了”“小me，别睡了，起床了”等唤醒指令；其他话不发送request，待机和唤醒不占八轮问答计数，也不伪造playback_done。每个本地待机窗口使用独立标识，迟到事件不影响新窗口。已清醒时同样的唤醒语句作为普通request。

待机循环安装目录 `sleep.gif` 的第0～14帧，声音读取原 `sleep.wav` 并使用已绑定USB音箱。唤醒先确认关麦，停止待机声音，再播放第15帧至结尾；最后一帧自身时长结束后恢复普通收音。保留20秒真正空闲自动待机，音乐、回复播放和推理期间不自动待机。USB失联、断线或关闭语音取消待机；恢复完成后按原就绪和输出路由规则回到普通状态，不续播中断片段。待机期间服务器原10秒空闲规则继续生效，因此唤醒后的下一条普通对话可能得到新的conversation_id，Pi仍以真实服务器编号选择背景。

meme_id 必须对应本机现有完整素材编号，reply_text 为 1～2000 字非空文字。表情检索和服务器meme_id不变；原编号及WAV来自 `sucai_meme/meme_gif`，实际GIF来自同级 `meme_gif_background/<原stem>-N.gif`。例如meme-0004在背景3时读取 `meme-0004-大笑秃头猫-3.gif`，声音仍为原目录 `meme-0004-大笑秃头猫.wav`。

收到新的conversation_id时，Pi从编号1～10内覆盖全部启用表情的完整背景组中随机选择一个，该段所有表情保持一致；下一段允许再次选到同一背景。不完整背景组不选，所选文件缺失时明确失败，不换背景、不回退无背景GIF。欢迎画面先随机背景，手动meme播放沿用当前背景；新段首次有效服务器结果重新确定背景。旧v2服务器若未提供conversation_id，兼容模式只能在其启动期间保持固定背景；要按真实新对话切换须更新服务器。未知、重复、已剔除编号不猜测替换。服务器不传音频、播放器不生成语音，本地配套声音内容不保证与回复文字相同。

## 验证

运行 `python3 -m unittest discover -s player/desktop/tests -v` 可检查握手字段、先停麦再发送、accepted 不结束等待、播放暂停收音、同 id 恢复、重复结果、重启及播放回执。协议 TCP 联调不等于真实麦克风/扬声器测试；现场仍须说一句完整话并核对服务端 id、Pi 字幕/表情以及声音结束后恢复收音。
