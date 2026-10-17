# Meme猫播放器：软件工作区

`meme_software/` 是本项目的软件代码贡献工作区。本次提供现有播放器的关键源码；其他团队目录保持原样。

## 程序入口

使用 **Python + Tkinter 原生桌面窗口** 显示表情、播放配套 WAV，并将回复字幕放在表情画面底部。默认全屏，画面内隐藏素材名称、时长与计时。无需浏览器或 WebView。

| 路径 | 用途 |
| --- | --- |
| `player/desktop/app.py` | 原生窗口、播放控制、素材管理与本机接口 |
| `player/desktop/native_core.py` | GIF/图片逐帧处理、缩放、ALSA WAV 播放 |
| `player/desktop/start.sh` | Linux 启动入口，明确使用 **8784** |
| `player/desktop/install_shortcut.py` | 可选的个人桌面/应用菜单快捷方式 |
| `player/launch.py` | 复用的 SQLite 素材服务、回复校验及 HTTP Handler |
| `player/website/library.json` | 原有场景配置与素材来源元数据，非媒体文件 |
| `player/data/README.md` | 现有素材数据库的放置说明 |
| `API.md` | 算法/服务器联调接口与当前边界 |

## 在树莓派使用

这是**源码提交，不是含素材和依赖的完整离线安装包**。需要已经安装 Python 3、Tkinter、Pillow（含 ImageTk）及 ALSA 的 `aplay`。已获取的目标机报告为树莓派 4B / Debian 12 ARM64 / Python 3.11，报告中这些依赖均已存在；不能由此推断其他电脑也已安装。

1. 在现有播放器停止后，备份并保留它的 `data/assets.sqlite3`，将其副本放到本目录 `player/data/assets.sqlite3`。数据库包含现有配对媒体；本次不上传数据库、音频、视频或模型。不要用空库覆盖原库，也不要重新导入已剔除的素材。
2. 在树莓派 Linux 桌面的终端进入 `meme_software`，执行：

```sh
python3 player/desktop/app.py --check
sh player/desktop/start.sh
```

相对路径以脚本位置定位。`--check` 不创建窗口；完整启动需要实际桌面会话，普通 SSH 会话不等于已进入屏幕桌面。

F11 切换全屏，Esc 退出全屏；底部按钮可选择表情、重播、管理或退出。界面里的场景按钮是人工演示，不代表已接入真实情绪识别或论文模型。

## 当前边界与验证

- 本次未接入收音识别、远程服务器或情绪算法；预留的本机回复接口见 `API.md`。离线 STT 模块独立维护，本次未上传。
- 原生管理支持 GIF/PNG/JPEG/WebP 配未压缩 WAV；视频或其他音频格式需要先转换。旧 `launch.py` 为共用后端源码，保留了历史网页入口，但本次未提交网页界面/媒体；请用 `desktop/start.sh` 启动原生软件。
- 现有 Windows 原生窗口、接口驱动画面及底部字幕已检查；211 套 GIF 共 10,117 帧及 WAV 格式已在主机验证。历史验证记录见 `player/desktop/VALIDATION.json`，相应媒体未随本次代码提交。
- 树莓派原生窗口、扬声器、触摸、实际帧率及 8784 运行仍需现场验收，不能将主机结果视为树莓派实测。
- 修改/导入素材前备份 `data`；管理窗口导入会自动备份数据库并生成新编号。不要操作其他队友的目录。

校验本次源码：在本目录执行 `sha256sum -c SHA256SUMS`。Python、Tk、Pillow、ALSA 的许可由各自项目维护；本次不新增第三方依赖文件，也不对媒体授权作承诺。
