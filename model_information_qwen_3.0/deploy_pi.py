"""Deploy only the reviewed player source files to the existing Raspberry Pi.

Credentials, configuration, media, and STT/model weights are never copied.
The remote transaction checks the inspected SHA values before replacing any
file, retains private backups, and rolls source back if verification fails.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

if __package__ in (None, ""):
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parent / "bootstrap.py"))["register_package"]()

from communication.common import load_token
from model_information_qwen_v3.launcher import load_settings
from model_information_qwen_v3.model import DEFAULT_CONFIG_PATH, PROJECT_ROOT


ALLOWED_FILES = frozenset({
    "player/desktop/server_link.py", "player/desktop/conversation_turn.py",
    "player/desktop/app.py", "player/desktop/capture_turn.py",
    "player/desktop/stt_bridge.py", "player/desktop/stt_worker.py",
    "player/desktop/voice_recovery.py", "player/desktop/audio_output.py",
    "player/desktop/asr_runtime/audio.py",
    "player/launch.py", "player/desktop/start.sh",
    "player/desktop/meme_sing/__init__.py", "player/desktop/meme_sing/core.py",
    "player/desktop/meme_sing/catalog.py", "player/desktop/meme_sing/playback.py",
    "player/desktop/meme_sing/controller.py",
    "player/desktop/meme_huanxing/__init__.py", "player/desktop/meme_huanxing/commands.py",
    "player/desktop/meme_huanxing/controller.py", "player/desktop/meme_huanxing/screen.py",
    "player/desktop/meme_huanxing/README.md",
    "player/desktop/directory_library.py", "player/desktop/native_core.py",
    "player/desktop/render_pipeline.py",
    "player/desktop/asr_selection.py",
    "player/desktop/asr_runtime/speech.py", "player/desktop/SERVER_PROTOCOL.md",
    "player/desktop/README.md", "README.md", "SHA256SUMS", "player/desktop/SHA256SUMS",
    "player/desktop/meme_sing/README.md", "player/desktop/sleep_idle.py",
})
DEFAULT_FILES = (
    "player/desktop/server_link.py", "player/desktop/conversation_turn.py",
    "player/desktop/app.py", "player/desktop/capture_turn.py",
    "player/desktop/stt_bridge.py", "player/desktop/stt_worker.py",
    "player/desktop/voice_recovery.py", "player/desktop/audio_output.py",
    "player/desktop/asr_runtime/audio.py",
    "player/launch.py", "player/desktop/start.sh",
    "player/desktop/meme_sing/__init__.py", "player/desktop/meme_sing/core.py",
    "player/desktop/meme_sing/catalog.py", "player/desktop/meme_sing/playback.py",
    "player/desktop/meme_sing/controller.py",
    "player/desktop/meme_huanxing/__init__.py", "player/desktop/meme_huanxing/commands.py",
    "player/desktop/meme_huanxing/controller.py", "player/desktop/meme_huanxing/screen.py",
    "player/desktop/meme_huanxing/README.md",
    "player/desktop/directory_library.py", "player/desktop/native_core.py",
    "player/desktop/render_pipeline.py",
    "player/desktop/asr_selection.py",
    "player/desktop/asr_runtime/speech.py", "player/desktop/SERVER_PROTOCOL.md",
)
MAX_FILE_BYTES = 2 * 1024 * 1024


class DeploymentError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


# Sent as source, with pi_startup in a separate namespace. No remote package
# installation or import from the files being replaced is necessary.
REMOTE_HELPER = r'''
import base64
import hashlib
import json
import os
import math
from pathlib import Path
import signal
import tempfile
import time
import uuid


class DeployFailure(RuntimeError):
    pass


REMOTE_ALLOWED_FILES = frozenset({
    "player/desktop/server_link.py", "player/desktop/conversation_turn.py",
    "player/desktop/app.py", "player/desktop/capture_turn.py",
    "player/desktop/stt_bridge.py", "player/desktop/stt_worker.py",
    "player/desktop/voice_recovery.py", "player/desktop/audio_output.py",
    "player/desktop/asr_runtime/audio.py",
    "player/launch.py", "player/desktop/start.sh",
    "player/desktop/meme_sing/__init__.py", "player/desktop/meme_sing/core.py",
    "player/desktop/meme_sing/catalog.py", "player/desktop/meme_sing/playback.py",
    "player/desktop/meme_sing/controller.py",
    "player/desktop/meme_huanxing/__init__.py", "player/desktop/meme_huanxing/commands.py",
    "player/desktop/meme_huanxing/controller.py", "player/desktop/meme_huanxing/screen.py",
    "player/desktop/meme_huanxing/README.md",
    "player/desktop/directory_library.py", "player/desktop/native_core.py",
    "player/desktop/render_pipeline.py",
    "player/desktop/asr_selection.py",
    "player/desktop/asr_runtime/speech.py", "player/desktop/SERVER_PROTOCOL.md",
    "player/desktop/README.md", "README.md", "SHA256SUMS", "player/desktop/SHA256SUMS",
    "player/desktop/meme_sing/README.md", "player/desktop/sleep_idle.py",
})


def _target(root, name, allowed):
    if not isinstance(name, str) or name not in allowed:
        raise DeployFailure("部署文件不在允许的源码白名单")
    target = root / name
    current = root
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            raise DeployFailure("部署目标或其父目录是符号链接：" + name)
    if not target.parent.is_dir():
        packages = {root / "player/desktop/meme_sing", root / "player/desktop/meme_huanxing"}
        if (target.parent not in packages or target.parent.exists()
                or not target.parent.parent.is_dir()):
            raise DeployFailure("部署源码目录不存在：" + name)
    return target


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _protected_hashes(root):
    # Inspect bytes only for their digest; saved user options never enter the
    # deployment payload, source backup, or report.
    names = ("player/data/speech-config.json", "player/desktop/asr-selection.json")
    return {name: _sha((root / name).read_bytes()) if (root / name).is_file() else None
            for name in names}


def _asr_report(root, status):
    config = root / "player/desktop/asr-selection.json"
    if not config.is_file():
        base = status.get("stt_root")
        model = (Path(base) / "models" /
                 "sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01") if base else None
        return {"profile": "lite", "threads": 2, "silence_rms": .008,
                "model_exists": bool(model and (model / "model.int8.onnx").is_file())}
    options = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(options, dict):
        raise DeployFailure("现有 ASR 选择配置不是 JSON 对象")
    profile = options.get("profile")
    if profile not in {"lite", "standard", "paraformer"}:
        raise DeployFailure("现有 ASR 选择配置的模型类型无效")
    model = options.get("model")
    if isinstance(model, str) and model.strip():
        path = Path(model).expanduser()
        model_path = path if path.is_absolute() else config.parent / path
        required = {"standard": ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx", "tokens.txt"),
                    "paraformer": ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt"),
                    "lite": ("model.int8.onnx", "tokens.txt")}[profile]
        exists = all((model_path / name).is_file() for name in required)
    else:
        exists = False
    threads = options.get("threads", 3)
    silence = options.get("silence_rms", .008)
    return {"profile": profile, "threads": threads if type(threads) is int and 1 <= threads <= 4 else None,
            "silence_rms": silence if type(silence) in (int, float) and math.isfinite(silence) else None,
            "model_exists": exists}


def _atomic(path, data, mode):
    descriptor, temporary = tempfile.mkstemp(prefix=".meme-deploy-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _owned_processes(startup, root):
    scripts = (root / "player/desktop/app.py", root / "player/desktop/stt_worker.py")
    return [process for process in startup["_processes"]()
            if any(startup["_matches_script"](process, script) for script in scripts)]


def _process_report(startup, root):
    return [{"pid": process["pid"], "kind": "app" if startup["_matches_script"](
             process, root / "player/desktop/app.py") else "stt"}
            for process in _owned_processes(startup, root)]


def _stop_owned(startup, root):
    # Resolve every PID again immediately before signaling. No process group,
    # command-name glob, or global pkill can affect another application.
    requested = _owned_processes(startup, root)
    stopped = []
    for process in requested:
        if any(current["pid"] == process["pid"] for current in _owned_processes(startup, root)):
            try:
                os.kill(process["pid"], signal.SIGTERM)
                stopped.append(process["pid"])
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        alive = [process for process in _owned_processes(startup, root)
                 if process["pid"] in stopped]
        if not alive:
            return stopped
        time.sleep(.1)
    failure = DeployFailure("指定播放器或 STT 未在 10 秒内正常退出；未强杀进程")
    remaining = {process["pid"] for process in _owned_processes(startup, root)}
    failure.stop_requested_pids = stopped
    failure.stopped_pids = [pid for pid in stopped if pid not in remaining]
    raise failure


def remote_run(payload, startup, state_root=None):
    if not isinstance(payload, dict) or payload.get("action") not in {"check", "apply"}:
        raise DeployFailure("action 必须是 check 或 apply")
    configured = dict(payload.get("settings", {}))
    configured["action"] = "check"
    status = startup["run"](configured)
    root = Path(configured["software_dir"]).resolve()
    names = payload.get("files")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise DeployFailure("部署文件清单无效或包含重复文件")
    targets = {name: _target(root, name, REMOTE_ALLOWED_FILES) for name in names}
    previous = {name: path.read_bytes() if path.is_file() else None for name, path in targets.items()}
    report = {"state": "checked", "software": status, "files": {
        name: {"sha256": _sha(data) if data is not None else None}
        for name, data in previous.items()}, "processes": _process_report(startup, root),
        "protected_files": _protected_hashes(root), "asr": _asr_report(root, status)}
    if payload["action"] == "check":
        return report
    entries = payload.get("entries")
    if not isinstance(entries, dict) or set(entries) != set(names):
        raise DeployFailure("部署内容与白名单文件清单不一致")
    if payload.get("expected_protected_files") != report["protected_files"]:
        raise DeployFailure("现有语音或 ASR 配置自检查后发生变化，已中止部署")
    prepared = {}
    modes = {}
    # Complete every validation and compile before any installed file changes.
    with tempfile.TemporaryDirectory(prefix="meme-source-check-") as directory:
        for name, target in targets.items():
            entry = entries[name]
            actual = report["files"][name]["sha256"]
            if entry.get("expected_sha256") != actual:
                raise DeployFailure("远端源码自检查后发生变化，已中止：" + name)
            try:
                data = base64.b64decode(entry["content"], validate=True)
            except (KeyError, ValueError, TypeError):
                raise DeployFailure("部署源码编码无效：" + name)
            if len(data) > 2 * 1024 * 1024 or _sha(data) != entry.get("sha256"):
                raise DeployFailure("部署源码大小或 SHA 校验失败：" + name)
            candidate = Path(directory) / Path(name).name
            candidate.write_bytes(data)
            if name.endswith(".py"):
                compile(candidate.read_bytes(), name, "exec")
            prepared[name] = data
            modes[name] = target.stat().st_mode & 0o777 if previous[name] is not None else 0o644
    directory = state_root or Path.home() / ".local/state/meme-model/deployments"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = directory / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex)
    backup.mkdir(mode=0o700)
    manifest = {name: {"before_sha256": report["files"][name]["sha256"],
                        "after_sha256": _sha(prepared[name]), "mode": modes[name]}
                for name in names}
    for name, data in previous.items():
        if data is None:
            continue
        saved = backup / name
        saved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        saved.write_bytes(data)
        saved.chmod(0o600)
    manifest_path = backup / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.chmod(0o600)
    changed = []
    was_running = any(process["kind"] == "app" for process in report["processes"])
    report.update({"backup_dir": str(backup), "restart_requested": bool(payload.get("restart")),
                   "stopped_pids": [], "stop_requested_pids": [], "rollback": False})
    restart_attempted = False
    created_directories = []
    try:
        # Create only fixed, reviewed feature packages after all validation.
        for package_name in ("meme_sing", "meme_huanxing"):
            package = root / "player/desktop" / package_name
            if any(path.parent == package for path in targets.values()) and not package.exists():
                package.mkdir(mode=0o755)
                created_directories.append(package)
        # A second hash check closes the backup-to-replacement interval too.
        for name, target in targets.items():
            _target(root, name, REMOTE_ALLOWED_FILES)
            current = target.read_bytes() if target.is_file() else None
            if current != previous[name]:
                raise DeployFailure("远端源码在部署期间发生变化：" + name)
            if prepared[name] != current:
                _atomic(target, prepared[name], modes[name])
                changed.append(name)
        for name, target in targets.items():
            if _sha(target.read_bytes()) != _sha(prepared[name]):
                raise DeployFailure("原子替换后的远端 SHA 校验失败：" + name)
        if _protected_hashes(root) != report["protected_files"]:
            raise DeployFailure("现有语音或 ASR 配置在部署期间发生变化，已回滚源码")
        report["changed_files"] = changed.copy()
        report["files"] = {name: {"sha256": _sha(prepared[name])} for name in names}
        if payload.get("restart"):
            restart_attempted = True
            report["stopped_pids"] = _stop_owned(startup, root)
            report["stop_requested_pids"] = report["stopped_pids"].copy()
            configured["action"] = "start"
            report["software"] = startup["run"](configured)
        report["state"] = "applied"
        report["requires_restart"] = was_running and not payload.get("restart") and bool(changed)
    except Exception as exc:
        report["state"] = "error"
        report["message"] = str(exc)
        rollback_errors = []
        if hasattr(exc, "stopped_pids"):
            report["stopped_pids"] = exc.stopped_pids
            report["stop_requested_pids"] = getattr(exc, "stop_requested_pids", exc.stopped_pids)
        # Newly started processes must not keep running rolled-back code.
        if restart_attempted:
            try:
                report["cleanup_stopped_pids"] = _stop_owned(startup, root)
            except Exception as stopped_exc:
                report["process_cleanup_error"] = str(stopped_exc)
        for name in reversed(changed):
            try:
                current = targets[name].read_bytes() if targets[name].is_file() else None
                if current != prepared[name]:
                    rollback_errors.append("现场源码在替换后发生并发修改，已保留当前文件：" + name)
                    continue
                if previous[name] is None:
                    targets[name].unlink(missing_ok=True)
                else:
                    _atomic(targets[name], previous[name], modes[name])
            except OSError:
                rollback_errors.append("无法恢复原文件：" + name)
        for created in reversed(created_directories):
            try:
                created.rmdir()  # Never remove files added by another writer.
            except OSError:
                pass
        report["rollback"] = not rollback_errors
        report["rollback_errors"] = rollback_errors
        if was_running and restart_attempted and not rollback_errors:
            try:
                configured["action"] = "start"
                report["software"] = startup["run"](configured)
            except Exception as recovery_exc:
                report["recovery_error"] = str(recovery_exc)
    report["processes"] = _process_report(startup, root)
    return report
'''


def _remote_source():
    startup = (PROJECT_ROOT / "model_information_qwen_3.0/pi_startup.py").read_text(encoding="utf-8")
    return (
        "import json, sys\nstartup = {'__name__': 'pi_startup_remote'}\n"
        + "exec(compile(" + repr(startup) + ", 'pi_startup.py', 'exec'), startup)\n"
        + REMOTE_HELPER
        + "\ntry:\n    report = remote_run(json.load(sys.stdin), startup)\n"
        + "except Exception as exc:\n    report = {'state': 'error', 'message': str(exc)}\n"
        + "print(json.dumps(report, ensure_ascii=False), flush=True)\n"
        + "raise SystemExit(1 if report['state'] == 'error' else 0)\n"
    )


def _remote_call(settings, payload):
    if not settings.ssh_key.is_file() or not settings.known_hosts.is_file():
        raise DeploymentError("缺少既有 SSH 私钥或已核验主机记录")
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
        "-o", "StrictHostKeyChecking=yes", "-o", "IdentitiesOnly=yes",
        "-o", f"UserKnownHostsFile={settings.known_hosts}", "-i", str(settings.ssh_key),
        "--", f"{settings.network['PI_SSH_USER']}@{settings.network['PI_ZT_IP']}",
        "python3 -c " + shlex.quote(_remote_source()),
    ]
    try:
        result = subprocess.run(command, input=json.dumps(payload), text=True,
                                capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise DeploymentError("树莓派部署 SSH 超时；请使用 --check 核对进程和文件状态") from exc
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise DeploymentError("树莓派未返回有效部署报告；请使用 --check 核对状态") from exc
    if not isinstance(report, dict) or report.get("state") not in {"checked", "applied", "error"}:
        raise DeploymentError("树莓派返回未知部署状态")
    if result.returncode or report["state"] == "error":
        raise DeploymentError(report.get("message", "树莓派部署失败"), report)
    return report


def deploy(settings, *, files=DEFAULT_FILES, apply=False, restart=False, source_root=None,
           expected_sha256=None, expected_protected_files=None):
    """Inspect, then optionally replace exactly the inspected source revision."""
    files = tuple(files)
    if not files or len(set(files)) != len(files) or any(name not in ALLOWED_FILES for name in files):
        raise DeploymentError("部署文件必须是无重复的显式源码白名单")
    if restart and not apply:
        raise DeploymentError("--restart 必须同时指定 --apply")
    token = load_token(settings.token_file)
    payload = {"action": "check", "files": list(files),
        "settings": {"software_dir": settings.software_dir, "host": settings.host,
                     "port": settings.port,
                     "expected_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest()}}
    report = _remote_call(settings, payload)
    if not apply:
        return report
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, dict) or any(name not in expected_sha256 for name in files):
            raise DeploymentError("已审阅报告没有包含本次全部部署文件，不能覆盖未审阅源码")
        changed = [name for name in files if report["files"][name]["sha256"] != expected_sha256[name]]
        if changed:
            raise DeploymentError("远端源码在审阅后发生变化，须先重新比较并合并：" + ", ".join(changed))
    if expected_protected_files is not None and report.get("protected_files") != expected_protected_files:
        raise DeploymentError("现有语音或 ASR 配置在审阅后发生变化，须先重新核对")
    source_root = Path(source_root or PROJECT_ROOT / "meme_software")
    entries = {}
    for name in files:
        path = source_root / name
        if (source_root.resolve() == (PROJECT_ROOT / "meme_software").resolve()
                and not path.exists()):
            for package_name in ("meme_sing", "meme_huanxing"):
                prefix = "player/desktop/" + package_name + "/"
                if name.startswith(prefix):
                    path = PROJECT_ROOT / package_name / name[len(prefix):]
                    break
        if path.is_symlink() or not path.is_file():
            raise DeploymentError("本地白名单源码不存在或为符号链接：" + name)
        data = path.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise DeploymentError("本地源码超出允许大小：" + name)
        if name.endswith(".py"):
            compile(data, name, "exec")
        entries[name] = {"content": base64.b64encode(data).decode("ascii"),
                         "sha256": hashlib.sha256(data).hexdigest(),
                         "expected_sha256": report["files"][name]["sha256"]}
    payload.update({"action": "apply", "entries": entries, "restart": bool(restart),
                    "expected_protected_files": report.get("protected_files", {})})
    return _remote_call(settings, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="白名单部署树莓派功能源码，保留口令、素材和配置")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="只读核对文件 SHA、软件与进程状态（默认）")
    action.add_argument("--apply", action="store_true", help="先备份，再原子替换通过核对的源码")
    parser.add_argument("--restart", action="store_true", help="部署成功后仅重启指定播放器及其 STT")
    parser.add_argument("--file", action="append", dest="files", choices=sorted(ALLOWED_FILES))
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "meme_software")
    parser.add_argument("--expected-report", type=Path,
                        help="apply 必须与这份先前已审阅 check 报告的源码及配置 SHA 一致")
    parser.add_argument("--network-config", type=Path, default=PROJECT_ROOT / "zerotier_chat/network.env")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH,
                        help="选定 Qwen 模型配置；只读取启动设置，不加载或训练模型")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, help="保存不含口令的结构化部署报告")
    args = parser.parse_args(argv)
    try:
        expected = None
        expected_protected = None
        if args.expected_report:
            reviewed = json.loads(args.expected_report.read_text(encoding="utf-8"))
            if not isinstance(reviewed, dict) or not isinstance(reviewed.get("files"), dict):
                raise DeploymentError("已审阅部署报告格式无效")
            expected = {name: entry["sha256"] for name, entry in reviewed["files"].items()}
            expected_protected = reviewed.get("protected_files")
            if not isinstance(expected_protected, dict):
                raise DeploymentError("已审阅部署报告缺少受保护配置的 SHA")
        result = deploy(load_settings(args), files=args.files or DEFAULT_FILES,
                        apply=args.apply, restart=args.restart, source_root=args.source_root,
                        expected_sha256=expected, expected_protected_files=expected_protected)
        code = 0
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, DeploymentError) as exc:
        result = getattr(exc, "report", None) or {"state": "error", "message": str(exc)}
        code = 1
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
