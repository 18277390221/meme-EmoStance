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
import time
import uuid

from communication.common import FRAME_LIMIT, load_token
from .console import follow_file, resolve_log_source
from .dialogue_log import DEFAULT_LOG_PATH
from .model import DEFAULT_CONFIG_PATH, DEFAULT_MODEL_DIR, PROJECT_ROOT, inspect_model_version
from zerotier_chat.prepare_wsl_network import load_config, prepare


class AlreadyRunning(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchSettings:
    network: dict
    model_dir: Path | None
    token_file: Path
    ssh_key: Path
    known_hosts: Path
    software_dir: str = "/home/aida/meme/meme-software"
    config_path: Path = DEFAULT_CONFIG_PATH

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
    configuration = getattr(args, "config", None) or DEFAULT_CONFIG_PATH
    return LaunchSettings(network, args.model_dir.expanduser().resolve() if args.model_dir else None, token_file,
        Path.home() / ".ssh/hks_zerotier_pi_ed25519",
        PROJECT_ROOT / "zerotier_chat/.ssh_known_hosts", config_path=Path(configuration).expanduser().resolve())


def check_local(settings):
    token = load_token(settings.token_file)
    info = inspect_model_version(settings.model_dir, config_path=settings.config_path)
    if not info["base_model_exists"]:
        raise ValueError(f"本地 Qwen3-8B 基座不存在：{info['base_model_path']}")
    for name in ("torch", "transformers", "numpy"):
        if importlib.util.find_spec(name) is None:
            raise ValueError(f"当前 Python 缺少 {name}；请用 QWEN_PYTHON 指向现有模型环境")
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
        yield descriptor
    finally:
        # Do not unlink: an existing waiter must never lock a different inode.
        os.close(descriptor)


def model_lock_held(path):
    """Inspect an existing lifetime lock without creating or modifying it."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(descriptor)


def selected_model_directory(settings, info):
    """Use the configuration's resolved run when no --model-dir is supplied."""
    value = info.get("model_dir") or info.get("run_dir") or settings.model_dir or DEFAULT_MODEL_DIR
    return str(Path(value).expanduser().resolve())


def record_runtime(descriptor, settings, info):
    """Store public runtime identity while the model lifetime lock is held."""
    metadata = {"pid": os.getpid(), "host": settings.host, "port": settings.port,
                "model_dir": selected_model_directory(settings, info), "version": info["version"],
                "dialogue_log": str(DEFAULT_LOG_PATH)}
    data = (str(os.getpid()) + "\n" + json.dumps(metadata, ensure_ascii=False) + "\n").encode("utf-8")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, data)
    os.ftruncate(descriptor, len(data))


def active_server_status(settings, token, info):
    """Confirm a reused runtime without submitting a turn or loading weights.

    New launchers record their selected endpoint and archive in server.lock.
    Older PID-only launchers are checked by their authenticated v2 greeting.
    A held lock without a listener is still allowed to be loading the model.
    """
    path = PROJECT_ROOT / "model_information_qwen/runs/server.lock"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    if len(lines) > 1:
        try:
            metadata = json.loads(lines[1])
        except (ValueError, TypeError) as exc:
            raise RuntimeError("已有模型服务的运行记录无效，请检查原启动终端") from exc
        if not isinstance(metadata, dict) or str(metadata.get("pid")) != lines[0]:
            raise RuntimeError("已有模型服务的运行记录无效，请检查原启动终端")
        if (metadata.get("host") != settings.host or metadata.get("port") != settings.port
                or metadata.get("model_dir") != selected_model_directory(settings, info)
                or metadata.get("version") != info["version"]):
            raise RuntimeError("已有模型服务使用其他模型或地址；请先停止原服务，再按新配置启动")
    try:
        connection = socket.create_connection((settings.host, settings.port), timeout=3)
    except OSError:
        return "starting"
    with connection:
        connection.settimeout(3)
        hello = {"type": "hello", "name": "meme-launcher", "protocol_version": 2,
                 "session_id": uuid.uuid4().hex, "token": token}
        try:
            connection.sendall((json.dumps(hello) + "\n").encode("utf-8"))
            with connection.makefile("rb") as stream:
                data = stream.readline(FRAME_LIMIT + 1)
            if not data.endswith(b"\n") or len(data) > FRAME_LIMIT:
                raise ValueError("invalid frame")
            reply = json.loads(data)
        except (OSError, ValueError, RecursionError) as exc:
            raise RuntimeError("已有端口未完成模型就绪握手，请检查原启动终端") from exc
    if not isinstance(reply, dict):
        raise RuntimeError("已有端口未返回有效的模型握手")
    if reply.get("type") == "error" and reply.get("code") == "auth_failed":
        raise RuntimeError("当前口令与已有模型服务不一致，请复用原服务的口令配置")
    if (reply.get("type") != "hello_ok" or reply.get("protocol_version") != 2
            or reply.get("session_id") != hello["session_id"] or reply.get("model_ready") is not True):
        raise RuntimeError("已有服务未提供版本 2 的模型就绪握手")
    if reply.get("model_version") != info["version"]:
        raise RuntimeError("已有模型服务的版本与本次选择不一致；请先停止原服务再切换版本")
    return "ready"


@contextmanager
def pi_operation_lock(path, timeout=60):
    """Serialize Pi starts from concurrent launchers, independently of the GPU."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("另一个命令仍在启动树莓派，请稍后再执行同一启动命令")
                time.sleep(.1)
        yield
    finally:
        os.close(descriptor)


def pi_action(settings, token, *, start=False):
    if start:
        with pi_operation_lock(PROJECT_ROOT / "model_information_qwen/runs/pi-start.lock"):
            return _pi_action(settings, token, start=True)
    return _pi_action(settings, token)


def _pi_action(settings, token, *, start=False):
    """Send configuration through SSH stdin, never put secrets in argv/logs."""
    if not settings.ssh_key.is_file():
        raise ValueError(f"缺少已配置的树莓派 SSH 密钥：{settings.ssh_key}")
    if not settings.known_hosts.is_file():
        raise ValueError(f"缺少树莓派已核验主机记录：{settings.known_hosts}")
    source = (PROJECT_ROOT / "model_information_qwen/pi_startup.py").read_text(encoding="utf-8")
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


def print_pi_status(settings, status):
    label = "已运行，保留当前软件" if status["state"] == "running" else "已启动"
    print(f"树莓派 {settings.network['PI_ZT_IP']}：{label}", flush=True)
    if status.get("stt_running") is False:
        print("树莓派播放器已运行，但语音识别进程尚未运行；尚不能确认可以语音对话", flush=True)


def follow_dialogue(settings, *, tail=20):
    lock_path = PROJECT_ROOT / "model_information_qwen/runs/server.lock"
    path, pid = resolve_log_source(lock_path, DEFAULT_LOG_PATH)
    print("正在显示双方对话，按 Ctrl+C 退出查看（已有模型服务继续运行）：", flush=True)

    def running():
        try:
            owner = lock_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            return False
        return owner == str(pid) and model_lock_held(lock_path)

    follow_file(path, running, tail=tail)
    print("原模型服务已结束，对话查看已停止", flush=True)


def reuse_server(settings, args):
    if args.server_only:
        print("模型服务已在运行或启动中，保留已有服务", flush=True)
        if not args.no_follow:
            follow_dialogue(settings)
        return 0
    token, info = check_local(settings)
    state = active_server_status(settings, token, info)
    prepare(settings.network)
    status = pi_action(settings, token, start=True)
    if state == "ready":
        print(f"复用已有 {info['version']} 模型服务：{settings.host}:{settings.port}", flush=True)
    else:
        print("已有模型服务正在运行或加载中；树莓派将等待模型就绪信号", flush=True)
    print_pi_status(settings, status)
    if not args.no_follow:
        follow_dialogue(settings)
    return 0


def main():
    parser = argparse.ArgumentParser(description="服务器一键启动模型，并连接树莓派现有软件")
    parser.add_argument("--check", action="store_true", help="仅检查配置和树莓派状态，不启动软件、模型或修改网络")
    parser.add_argument("--server-only", action="store_true", help="仅启动服务器，不通过 SSH 检查或启动树莓派软件")
    parser.add_argument("--no-follow", action="store_true", help="复用已有模型服务时启动树莓派后直接返回，不显示实时对话")
    parser.add_argument("--network-config", type=Path, default=PROJECT_ROOT / "zerotier_chat/network.env")
    parser.add_argument("--token-file", type=Path, help="默认复用 zerotier_chat/.env 的原口令")
    parser.add_argument("--model-dir", type=Path, default=None, help="覆盖推理配置中的运行目录")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="默认 model_qwen/inference_config.json")
    args = parser.parse_args()
    try:
        settings = load_settings(args)
        lock_path = PROJECT_ROOT / "model_information_qwen/runs/server.lock"
        if args.check:
            token, info = check_local(settings)
            state = None
            if model_lock_held(lock_path):
                state = active_server_status(settings, token, info)
            else:
                check_port(settings.host, settings.port)
            status = None if args.server_only else pi_action(settings, token)
            print(f"检查通过：服务器 {settings.host}:{settings.port}，Qwen 模型 {info['version']}", flush=True)
            if state == "starting":
                print("模型服务仍在启动，尚未收到模型就绪握手", flush=True)
            print(f"复用已有口令配置：{settings.token_file}（不重新生成）", flush=True)
            if status:
                print(f"树莓派 {settings.network['PI_ZT_IP']} 软件状态：{status['state']}", flush=True)
            return 0
        try:
            with startup_lock(lock_path) as descriptor:
                token, info = check_local(settings)
                record_runtime(descriptor, settings, info)
                check_port(settings.host, settings.port)
                prepare(settings.network)
                if not args.server_only:
                    print_pi_status(settings, pi_action(settings, token, start=True))
                print(f"正在加载 Qwen 模型 {info['version']}；服务器 {settings.host}:{settings.port}……", flush=True)
                from .model import load_chat_engine
                from .server import InferenceServer
                engine = load_chat_engine(settings.model_dir, config_path=settings.config_path)
                asyncio.run(InferenceServer(engine, token, model_version=info["version"]).run(settings.host, settings.port))
        except AlreadyRunning:
            return reuse_server(settings, args)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"启动失败：{exc}\n")
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
