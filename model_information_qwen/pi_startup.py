"""Check or launch the existing Pi desktop software without installing files.

The launcher sends this module's source to ``ssh ... python3 -c SOURCE`` and
passes settings as JSON on stdin. Only a new player process and its private
state-directory log may be created; existing software and credentials remain
unchanged. No token or token digest is returned or printed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


DEFAULT_HOST = "10.205.84.185"
PLAYER_PORT = 8784
PLAYER_URL = f"http://127.0.0.1:{PLAYER_PORT}"
END_SILENCE_SECONDS = .7
LITE_MODEL = "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01"
DISPLAY_KEYS = {
    "DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS",
    "DESKTOP_SESSION", "XDG_SESSION_TYPE",
}


class StartupError(RuntimeError):
    """A prerequisite failed without changing the running Pi software."""


def _read_shared_token_hash(path: Path) -> str:
    """Parse exactly the literal format used by the existing ServerLink."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise StartupError("树莓派缺少可读取的既有通信口令文件，请检查 ~/zerotier_chat/.env") from exc
    token = next((line.split("=", 1)[1].strip() for line in lines if line.startswith("ZT_CHAT_TOKEN=")), "")
    if len(token) < 16:
        raise StartupError("树莓派既有通信口令文件格式无效")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _processes() -> list[dict]:
    """Inspect only this SSH user's processes and selected desktop variables."""
    result = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = [part.decode("utf-8", errors="replace") for part in (process / "cmdline").read_bytes().split(b"\0") if part]
            comm = (process / "comm").read_text().strip()
            if not argv:
                continue
            desktop = {}
            if comm in {"lxsession", "openbox", "lxpanel", "Xorg", "labwc", "wayfire"} or any(Path(arg).name == "app.py" for arg in argv[1:]):
                for record in (process / "environ").read_bytes().split(b"\0"):
                    key, separator, value = record.partition(b"=")
                    name = key.decode("utf-8", errors="replace")
                    if separator and name in DISPLAY_KEYS:
                        desktop[name] = value.decode("utf-8", errors="replace")
            result.append({"pid": int(process.name), "argv": argv, "comm": comm,
                           "cwd": (process / "cwd").resolve(), "desktop": desktop})
        except (OSError, ValueError):
            continue
    return result


def _matches_script(process: dict, script: Path) -> bool:
    if "python" not in process.get("comm", ""):
        return False
    for argument in process.get("argv", [])[1:]:
        if Path(argument).name != script.name:
            continue
        candidate = Path(argument)
        if not candidate.is_absolute():
            candidate = process["cwd"] / candidate
        if candidate.resolve() == script.resolve():
            return True
    return False


def _argument_value(argv: list[str], option: str, default: str) -> str:
    value = default
    for index, argument in enumerate(argv):
        if argument.startswith(option + "="):
            value = argument.split("=", 1)[1]
        elif argument == option and index + 1 < len(argv):
            value = argv[index + 1]
    return value


def _player_health() -> bool:
    try:
        with urlopen(PLAYER_URL + "/api/health", timeout=2) as response:
            payload = json.loads(response.read(4096))
            return response.status == 200 and isinstance(payload, dict) and payload.get("status") == "ready"
    except (OSError, URLError, ValueError, TimeoutError):
        return False


def _software_settings(player_root: Path) -> dict:
    config = player_root / "data/speech-config.json"
    settings = {"host": DEFAULT_HOST, "port": 9000, "asr_root": "",
                "token_file": str(Path.home() / "zerotier_chat/.env")}
    if config.is_file():
        try:
            saved = json.loads(config.read_text(encoding="utf-8"))
            if not isinstance(saved, dict):
                raise ValueError("not an object")
            settings.update({key: saved[key] for key in settings if key in saved})
        except (OSError, ValueError, UnicodeError) as exc:
            raise StartupError("播放器现有 speech-config.json 无效，请在语音设置检查配置") from exc
    return settings


def _valid_asr_root(root: Path) -> bool:
    model = root / "models" / LITE_MODEL
    python = root / ".venv/bin/python"
    return python.is_file() and os.access(python, os.X_OK) and (model / "model.int8.onnx").is_file() and (model / "tokens.txt").is_file()


def _select_asr_root(software: Path, configured: str = "") -> Path:
    if configured:
        root = Path(configured).expanduser().resolve()
        if not _valid_asr_root(root):
            raise StartupError("播放器已配置的 STT 目录缺少可用虚拟环境或 lite 模型")
        return root
    bundled = software / "stt"
    if _valid_asr_root(bundled):
        return bundled.resolve()
    candidates = []
    for parent in (software, software.parent):
        for candidate in sorted(parent.glob("offline-asr-upgrade*/offline-asr-upgrade")):
            candidate = candidate.resolve()
            if candidate not in candidates and _valid_asr_root(candidate):
                candidates.append(candidate)
    if not candidates:
        raise StartupError("未找到完整离线 STT 安装：需要 .venv/bin/python、lite 模型和 tokens.txt")
    if len(candidates) != 1:
        raise StartupError("找到多份完整 STT 安装，请先在播放器语音设置明确选择一份")
    return candidates[0]


def _desktop_environment(processes: list[dict]) -> dict[str, str]:
    choices = [dict(process.get("desktop", {})) for process in processes if process.get("desktop", {}).get("DISPLAY")]
    choices.append({"DISPLAY": ":0", "XAUTHORITY": str(Path.home() / ".Xauthority"),
                    "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
                    "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{os.getuid()}/bus"})
    for choice in choices:
        display = choice.get("DISPLAY", "")
        match = re.fullmatch(r":([0-9]+)(?:\.[0-9]+)?", display)
        if match is None or not Path(f"/tmp/.X11-unix/X{match.group(1)}").exists():
            continue
        authority = Path(choice.get("XAUTHORITY", str(Path.home() / ".Xauthority"))).expanduser()
        if not authority.is_file() or not os.access(authority, os.R_OK):
            continue
        choice["XAUTHORITY"] = str(authority)
        environment = os.environ.copy()
        # The existing app reads its token from the verified local file. A
        # server's SSH environment must not silently override that credential.
        environment.pop("ZT_CHAT_TOKEN", None)
        environment.update(choice)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        probe = shutil.which("xdpyinfo")
        if probe is not None:
            try:
                check = subprocess.run([probe, "-display", display], env=environment,
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if check.returncode != 0:
                continue
        return environment
    raise StartupError("树莓派没有当前用户可访问的已登录 X 桌面；请先在树莓派屏幕登录桌面")


def _check_resources(software: Path) -> None:
    required = (
        "player/desktop/app.py", "player/desktop/start.sh", "player/desktop/server_link.py",
        "player/desktop/stt_bridge.py", "player/desktop/stt_worker.py",
        "player/desktop/asr_runtime/speech.py", "player/desktop/asr_runtime/audio.py",
        "player/launch.py", "player/website/library.json", "player/data/assets.sqlite3",
    )
    missing = [name for name in required if not (software / name).is_file()]
    if missing:
        raise StartupError("树莓派软件或素材缺少文件：" + ", ".join(missing))
    if shutil.which("arecord") is None or shutil.which("aplay") is None:
        raise StartupError("树莓派缺少 ALSA arecord/aplay，无法启动既有收音与播放功能")


def run(settings: dict) -> dict:
    """Return running/startable/started, or raise a credential-safe error."""
    if not isinstance(settings, dict) or settings.get("action") not in {"check", "start"}:
        raise StartupError("action 必须为 check 或 start")
    host, port = settings.get("host"), settings.get("port", 9000)
    if not isinstance(host, str) or not host.strip() or isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise StartupError("服务器地址或端口无效")
    expected = settings.get("expected_token_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
        raise StartupError("缺少服务器通信口令核验信息")
    token_file = Path.home() / "zerotier_chat/.env"
    if not hmac.compare_digest(_read_shared_token_hash(token_file), expected.lower()):
        raise StartupError("树莓派与服务器的通信口令不一致；保留现有配置，请核对同一份口令")
    path = settings.get("software_dir", "/home/aida/meme/meme-software")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise StartupError("software_dir 必须是树莓派软件绝对路径")
    software = Path(path).resolve()
    _check_resources(software)
    configured = _software_settings(software / "player")
    try:
        configured_port = int(configured["port"])
    except (ValueError, TypeError) as exc:
        raise StartupError("播放器已配置的服务器端口无效") from exc
    if configured["host"] != host or configured_port != port:
        raise StartupError("播放器现有服务器地址/端口与启动目标不一致；请在语音设置调整，脚本不会改写配置")
    if not isinstance(configured["token_file"], str) or Path(configured["token_file"]).expanduser().resolve() != token_file.resolve():
        raise StartupError("播放器使用了其他口令文件；请在语音设置核对 ~/zerotier_chat/.env")
    if not isinstance(configured["asr_root"], str):
        raise StartupError("播放器现有 STT 目录配置无效，请在语音设置检查")
    processes = _processes()
    app = software / "player/desktop/app.py"
    running = [process for process in processes if _matches_script(process, app)]
    healthy = _player_health()
    base = {"software_dir": str(software), "player_url": PLAYER_URL}
    if running:
        provided_api = [process for process in running if "--no-api" not in process["argv"]
                        and _argument_value(process["argv"], "--port", str(PLAYER_PORT)) == str(PLAYER_PORT)]
        if not provided_api:
            raise StartupError("已有播放器未提供 8784 本机接口；已保留进程，不自动重启或修改端口")
        if not healthy:
            raise StartupError("已有播放器进程，但 8784 本机接口未就绪；已保留进程，请在树莓派检查")
        if any("--no-stt" in process["argv"] for process in running):
            raise StartupError("已有播放器以 --no-stt 启动，未自动重启；请在树莓派正常退出后重试")
        return {**base, "state": "running", "pid": running[0]["pid"],
                "stt_running": any(_matches_script(process, software / "player/desktop/stt_worker.py") for process in processes)}
    if healthy:
        raise StartupError("8784 接口已有服务，但不属于指定软件；不会启动重复实例或终止其他程序")
    asr = _select_asr_root(software, configured["asr_root"])
    environment = _desktop_environment(processes)
    # Pass the policy after the wrapper's own defaults so even an older
    # start.sh with --end-silence 0.5 honors the requested 0.7-second wait.
    command = ["/bin/sh", str(software / "player/desktop/start.sh"), "--stt-root", str(asr),
               "--end-silence", str(END_SILENCE_SECONDS)]
    python = Path("/usr/bin/python3")
    try:
        check = subprocess.run(
            [str(python), str(app), "--check"], env=environment,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=20, text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StartupError("播放器依赖与素材检查未能完成，尚未启动软件") from exc
    if check.returncode != 0:
        raise StartupError("播放器依赖或素材检查失败，尚未启动；请在树莓派运行 app.py --check 检查")
    base.update({"stt_root": str(asr), "display": environment["DISPLAY"]})
    if settings["action"] == "check":
        return {**base, "state": "startable"}
    directory = Path.home() / ".local/state/meme-model"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_path = directory / "pi-player.log"
    try:
        descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, "ab") as log_file:
            child = subprocess.Popen(
                command, cwd=software / "player/desktop", env=environment,
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
    except OSError as exc:
        raise StartupError("无法启动树莓派播放器；请检查本机状态目录和软件执行权限") from exc
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise StartupError("播放器启动后退出；请在树莓派检查 " + str(log_path))
        if _player_health():
            return {**base, "state": "started", "pid": child.pid, "log_file": str(log_path)}
        time.sleep(.25)
    raise StartupError("已发起播放器启动，但 8784 接口未在 20 秒内就绪；未再次启动，请检查 " + str(log_path))


def main() -> int:
    try:
        settings = json.load(sys.stdin)
        result = run(settings)
    except (StartupError, OSError, ValueError, TypeError) as exc:
        print("树莓派启动检查失败：" + str(exc), file=sys.stderr, flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
