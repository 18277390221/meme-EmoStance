"""Qwen console dialogue and authenticated TCP serving entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def chat(argv):
    parser = argparse.ArgumentParser(description="Qwen3-8B EmoStance 单次生成对话")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--chat-config", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default=None)
    parser.add_argument("--text", default=None, help="单次推理；省略则从标准输入逐行读取")
    args = parser.parse_args(argv)
    from .model import load_chat_engine
    from .server import DEFAULT_MAX_TURNS
    try:
        engine = load_chat_engine(args.model_dir, config_path=args.config,
            settings_path=args.chat_config, device=args.device, dtype=args.dtype)
        history, last_user_at = [], None
        timeout = float(engine.config.get("idle_timeout", 10))
        def reply(text):
            nonlocal history, last_user_at
            now = time.monotonic()
            if last_user_at is not None and now - last_user_at >= timeout:
                history = []
            last_user_at = now
            turn = engine.reply_turn([*history, {"role": "user", "content": text}])
            history.extend(({"role": "user", "content": turn.user_text},
                            {"role": "assistant", "content": turn.reply}))
            if DEFAULT_MAX_TURNS is not None:
                del history[:-2 * DEFAULT_MAX_TURNS]
            print(json.dumps(turn.as_reply(), ensure_ascii=False), flush=True)
        if args.text is not None:
            reply(args.text)
        else:
            if sys.stdin.isatty():
                print("输入文字开始对话，/quit 退出。", file=sys.stderr, flush=True)
            for line in sys.stdin:
                text = line.rstrip("\r\n")
                if text == "/quit":
                    break
                if text.strip():
                    reply(text)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"推理失败：{exc}\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mode = argv.pop(0) if argv and argv[0] in ("chat", "serve") else "serve"
    if mode == "chat":
        chat(argv)
    else:
        from .server import main as serve
        serve(argv)
