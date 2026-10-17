from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from .config import Config, local_backbone


def doctor(config):
    required = ["numpy", "torch", "transformers", "accelerate"]
    packages = {name: importlib.util.find_spec(name) is not None for name in required}
    result = {"packages": packages, "base_model": str(local_backbone(config.base_model)),
              "model_downloads": False, "annotations_configured": bool(config.annotations)}
    if packages["torch"]:
        import torch
        result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all(packages.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="本地 Mistral 回复生成、情绪向量检索和 10 秒会话管理")
    parser.add_argument("--config", help="配置 JSON 路径；默认 model/config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="检查本地权重与依赖，不加载权重、不运行训练")
    subparsers.add_parser("prepare", help="连接文本/标注并按对话划分，仅生成数据文件")
    subparsers.add_parser("train", help="缓存 Mistral 特征并训练情绪预测器和 prefix")
    chat = subparsers.add_parser("chat", help="使用已训练的产物进行终端对话")
    chat.add_argument("--run-dir")
    server = subparsers.add_parser("serve", help="提供 POST /chat 和 POST /reset 接口")
    server.add_argument("--run-dir")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=8790)
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config)
        if args.command == "doctor":
            return doctor(config)
        if args.command == "prepare":
            from .training import prepare
            dataset = prepare(config)
            print(json.dumps(dataset["report"], ensure_ascii=False, indent=2))
        elif args.command == "train":
            from .training import train
            manifest = train(config)
            print(json.dumps({"status": manifest["status"], "run_dir": config.run_dir}, ensure_ascii=False))
        else:
            from .engine import MemeEngine
            engine = MemeEngine(args.run_dir or config.run_dir, device=config.device, dtype=config.dtype)
            if args.command == "serve":
                from .http_server import serve
                serve(engine, host=args.host, port=args.port, idle_timeout=config.idle_timeout)
            else:
                from .service import ChatService
                with ChatService(engine, idle_timeout=config.idle_timeout, cleanup_interval=0.25) as service:
                    print("输入一句话；/reset 清空上下文，/quit 退出。连续 10 秒无用户输入将自动重置。")
                    while True:
                        try:
                            text = input("> ")
                        except (EOFError, KeyboardInterrupt):
                            break
                        if text.strip() == "/quit":
                            break
                        if text.strip() == "/reset":
                            service.sessions.reset("default")
                            print("上下文已重置")
                            continue
                        if text.strip():
                            print(json.dumps(service.reply(text), ensure_ascii=False))
        return 0
    except (ValueError, FileNotFoundError, ImportError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
