"""Create a private shared-token file once; never print the token."""

import argparse
import os
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser(description="生成通信口令文件，不覆盖已有文件")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(".env"))
    args = parser.parse_args()
    try:
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(f"已有配置，保留：{args.output}")
        return
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(f"ZT_CHAT_TOKEN={secrets.token_urlsafe(32)}\n")
    print(f"已生成：{args.output}（权限 600；请安全复制到树莓派同目录）")


if __name__ == "__main__":
    main()
