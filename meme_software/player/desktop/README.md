# 原生桌面播放器

本目录必须与上一级的 `launch.py`、`website/library.json` 和 `data/assets.sqlite3` 配套。依赖与素材准备见 ../../README.md。

从 `meme_software` 目录执行：

```sh
python3 player/desktop/app.py --check
sh player/desktop/start.sh
```

默认全屏，使用本机接口端口 **8784**；`start.sh` 已明确写入该端口。端口被占用时明确提示，不终止其他进程或自动换端口。

```sh
sh player/desktop/start.sh --windowed
sh player/desktop/start.sh --port 8785
sh player/desktop/start.sh --no-api --mute
python3 player/desktop/install_shortcut.py
```

上述依次为普通窗口、显式覆盖端口、仅画面测试和可选个人桌面/应用菜单快捷方式。快捷方式不需要 sudo，不加入开机自启；移动目录后应重新创建，遇到不同内容的同名入口会拒绝覆盖。

表情/声音来自本地素材库，字幕位于表情底部；管理信息在单独的管理窗口。F11 切换全屏、Esc 退出全屏。管理支持新增 GIF/PNG/JPEG/WebP 与未压缩 WAV，导入前备份数据库，生成新编号。

默认演示场景不是模型推理。收音、语音识别、远程服务器和情绪模型尚未与本模块连通。程序通过共用后端的本机 HTTP 接口接收已有 `meme_id` 和 `reply_text`，详见 ../../API.md；窗口本身完全原生，不打开浏览器。

历史验证见 VALIDATION.json：Windows 独立窗口与素材解码已检查；树莓派实际显示、声音、触摸与帧率尚未验收。SHA256SUMS 校验本目录文件。
