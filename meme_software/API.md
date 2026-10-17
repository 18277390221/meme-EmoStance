# 本机播放接口

启动 `player/desktop/start.sh` 后，原生程序在 `127.0.0.1:8784` 提供接口。只接受本机请求；不是公网服务，不应直接改成外网监听。本机接口不负责收音、语音识别、情绪分类或请求远程服务器。

## 流程与边界

未来由独立联调模块获得服务器的回复，映射到本地已有素材 ID，再将 ID 和回复文字交给本接口。服务器通信协议、认证、重连和情绪空间匹配尚未实现；本次不伪造这些结果。`GET /api/assets` 是实际导入的素材清单，算法方应以此核对 ID。`GET /api/library` 还包含历史基础元数据，不能据此认定每项都有本地媒体。

## 接口

| 方法/路径 | 用途 |
| --- | --- |
| `GET /api/health` | 本地服务健康状态，mode 仍为 local-demo |
| `GET /api/assets` | 数据库内实际素材 ID 和元数据 |
| `GET /api/state` | 最新事件、revision 和回复 |
| `POST /api/response` | 播放选定素材、配套声音并显示底部字幕 |
| `POST /api/stop` | 停止当前播放，请求体为 `{}` |

请求内容类型须为 `application/json`。回复请求示例（需本地库中实际存在 meme-0157）：

```json
{"meme_id":"meme-0157","reply_text":"你好呀，我在这里。","request_id":"demo-001"}
```

`reply_text` 为 1～2000 字的非空文字。`request_id` 为 1～80 位字母/数字/下划线/短横线；省略时自动生成。同一进程内，最近 200 个 request_id 会去重：同一 ID 和相同内容不会重复播放，同一 ID 改变内容返回 400。新的回复使用新 request_id。未知编号或字段不合法返回 400，HTTP 200 表示事件已接收，不等于扬声器/画面已完成播放。

可在树莓派的另一个终端用 Python 标准库发送演示请求；先确认播放器已启动：

```python
import json
from urllib.request import Request, urlopen

body = {"meme_id": "meme-0157", "reply_text": "你好呀，我在这里。",
        "request_id": "demo-001"}
request = Request("http://127.0.0.1:8784/api/response",
                  data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=5) as response:
    print(response.read().decode("utf-8"))
```

HTTP 事件是最新状态，不是排队播放列表；新回复会替换当前播放。`--no-api` 模式不提供这些接口。传给原生窗口的素材需要为支持的图片/动画及 WAV；历史后端允许的视频等类型不能因此视为原生版支持。
