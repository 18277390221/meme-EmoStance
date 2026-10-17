"""One-command launcher for the existing server and Raspberry Pi installation.

Reuse configured credentials and the existing mirrored-WSL route helper. SSH
only probes/starts the installed Pi app; it does not replace its files or data.
The final model server stays in the foreground and owns its lifetime lock.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

from communication.common import load_token
from model_information.model import DEFAULT_MODEL_DIR, PROJECT_ROOT, inspect_model_version
from zerotier_chat.prepare_wsl_network import load_config, prepare


class AlreadyRunning(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchSettings:
    network: dict
    model_dir: Path
    token_file: Path
    ssh_key: Path
    known_hosts: Path
    software_dir: str = "/home/aida/meme/meme-software"

    @property
    def host(self):
        return self.network["WSL_ZT_IP"]

    @property
    def port(self):
        return int(self.network["CHAT_PORT"])


def load_settings(args):
    network = load_config(args.network_config)
    user = network.get("PI_SSH_USER", "aida")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,31}", user):
        raise ValueError("network.env 的 PI_SSH_USER 不是有效用户名")
    network["PI_SSH_USER"] = user
    if args.token_file is not None:
        token_file = args.token_file.expanduser().resolve()
    else:
        original = PROJECT_ROOT / "zerotier_chat/.env"
        token_file = original if original.is_file() else PROJECT_ROOT / "communication/.env"
    return LaunchSettings(network, args.model_dir.expanduser().resolve(), token_file,
        Path.home() / ".ssh/hks_zerotier_pi_ed25519",
        PROJECT_ROOT / "zerotier_chat/.ssh_known_hosts")


def check_local(settings):
    token = load_token(settings.token_file)
    info = inspect_model_version(settings.model_dir)
    if not info["base_model_exists"]:
        raise ValueError(f"本地 Mistral 基座不存在：{info['base_model_path']}")
    for name in ("torch", "transformers", "numpy"):
        if importlib.util.find_spec(name) is None:
            raise ValueError(f"当前 Python 缺少 {name}；请用 MEME_PYTHON 指向现有模型环境")
    return token, info


def check_port(host, port):
    """Fail before loading GPU weights; never stop or replace another process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            if exc.errno == 98:
                raise RuntimeError(
                    f"{host}:{port} 已被其他服务占用。若为旧通信服务，"
                    "请在其终端输入 /quit 或按 Ctrl+C 后重试"
                ) from exc
            raise RuntimeError(f"无法监听 {host}:{port}，请检查 Windows ZeroTier 和 WSL 镜像网卡") from exc


@contextmanager
def startup_lock(path):
    """Prevent concurrent launches even while the model has not bound its port."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning("模型服务已在运行或启动中，请查看原启动终端，无需重复启动") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, (str(os.getpid()) + "\n").encode("ascii"))
        yield
    finally:
        # Do not unlink: an existing waiter must never lock a different inode.
        os.close(descriptor)


def pi_action(settings, token, *, start=False):
    """Send configuration through SSH stdin, never put secrets in argv/logs."""
    if not settings.ssh_key.is_file():
        raise ValueError(f"缺少已配置的树莓派 SSH 密钥：{settings.ssh_key}")
    if not settings.known_hosts.is_file():
        raise ValueError(f"缺少树莓派已核验主机记录：{settings.known_hosts}")
    source = (PROJECT_ROOT / "model_information/pi_startup.py").read_text(encoding="utf-8")
    options = {
        "action": "start" if start else "check", "host": settings.host,
        "port": settings.port, "software_dir": settings.software_dir,
        "expected_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
        "-o", f"UserKnownHostsFile={settings.known_hosts}", "-i", str(settings.ssh_key),
        "--", f"{settings.network['PI_SSH_USER']}@{settings.network['PI_ZT_IP']}",
        "python3 -c " + shlex.quote(source),
    ]
    try:
        result = subprocess.run(command, input=json.dumps(options), text=True,
            capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("连接或启动树莓派软件超时，请确认树莓派已开机并连接 ZeroTier") from exc
    if result.returncode:
        # Remote helper errors never include file contents, hashes or tokens.
        detail = result.stderr.strip() or "SSH 未成功完成"
        try:
            remote = json.loads(result.stdout)
            if isinstance(remote, dict) and remote.get("state") == "error" and isinstance(remote.get("message"), str):
                detail = remote["message"]
        except (ValueError, TypeError):
            pass
        raise RuntimeError(f"树莓派准备失败：{detail}")
    try:
        response = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("树莓派未返回有效的软件状态") from exc
    if not isinstance(response, dict) or response.get("state") not in ("running", "started", "startable"):
        raise RuntimeError("树莓派返回了未知的软件状态")
    if start and response["state"] == "startable":
        raise RuntimeError("树莓派软件尚未启动，未进入模型服务")
    return response


def main():
    parser = argparse.ArgumentParser(description="服务器一键启动模型，并连接树莓派现有软件")
    parser.add_argument("--check", action="store_true", help="仅检查配置和树莓派状态，不启动软件、模型或修改网络")
    parser.add_argument("--server-only", action="store_true", help="仅启动服务器，不通过 SSH 检查或启动树莓派软件")
    parser.add_argument("--network-config", type=Path, default=PROJECT_ROOT / "zerotier_chat/network.env")
    parser.add_argument("--token-file", type=Path, help="默认复用 zerotier_chat/.env 的原口令")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    try:
        settings = load_settings(args)
        if args.check:
            token, info = check_local(settings)
            check_port(settings.host, settings.port)
            status = None if args.server_only else pi_action(settings, token)
            print(f"检查通过：服务器 {settings.host}:{settings.port}，归档模型 {info['version']}", flush=True)
            print(f"复用已有口令配置：{settings.token_file}（不重新生成）", flush=True)
            if status:
                print(f"树莓派 {settings.network['PI_ZT_IP']} 软件状态：{status['state']}", flush=True)
            return 0
        with startup_lock(PROJECT_ROOT / "model_information/runs/server.lock"):
            token, info = check_local(settings)
            check_port(settings.host, settings.port)
            prepare(settings.network)
            if not args.server_only:
                status = pi_action(settings, token, start=True)
                label = "已运行，保留当前软件" if status["state"] == "running" else "已启动"
                print(f"树莓派 {settings.network['PI_ZT_IP']}：{label}", flush=True)
            print(f"正在加载归档模型 {info['version']}；服务器 {settings.host}:{settings.port}……", flush=True)
            from model_information.model import load_chat_engine
            from model_information.server import InferenceServer
            engine = load_chat_engine(settings.model_dir)
            asyncio.run(InferenceServer(engine, token).run(settings.host, settings.port))
        return 0
    except AlreadyRunning as exc:
        print(str(exc), flush=True)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
