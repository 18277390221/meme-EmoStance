"""Launcher checks using fake credentials/SSH/model and ephemeral loopback ports."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from model_information_qwen import launcher


TOKEN = "fake-local-launcher-token-123"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        (self.root / "zerotier_chat").mkdir(parents=True)
        (self.root / "communication").mkdir()
        (self.root / "model_information_qwen").mkdir()
        self.home = Path(self.temporary.name) / "home"
        (self.home / ".ssh").mkdir(parents=True)
        self.network_path = self.root / "zerotier_chat/network.env"
        self.network_path.write_text("\n".join([
            "PI_ZT_IP=192.0.2.95", "WINDOWS_ZT_IP=192.0.2.185", "WSL_ZT_IP=192.0.2.185",
            "CHAT_PORT=9000", "PI_SSH_USER=aida", "WSL_DISTRO_NAME=Ubuntu",
        ]) + "\n", encoding="utf-8")
        self.args = argparse.Namespace(network_config=self.network_path, model_dir=self.root / "archive/v1.0.0", token_file=None)
        self.project_patch = patch("model_information_qwen.launcher.PROJECT_ROOT", self.root)
        self.home_patch = patch.object(Path, "home", return_value=self.home)
        self.project_patch.start()
        self.home_patch.start()
        self.addCleanup(self.project_patch.stop)
        self.addCleanup(self.home_patch.stop)

    def settings(self):
        return launcher.load_settings(self.args)

    def pi_settings(self):
        settings = self.settings()
        settings.ssh_key.write_text("fake-key-fixture", encoding="utf-8")
        settings.known_hosts.write_text("fake-host-fixture", encoding="utf-8")
        (self.root / "model_information_qwen/pi_startup.py").write_text("import sys\nprint('fake helper')\n", encoding="utf-8")
        return settings

    def main_mocks(self, arguments=()):
        """Return patches; every potentially external effect is explicitly fake."""
        settings = self.settings()
        engine = object()
        server = MagicMock()
        server.run = AsyncMock()
        return settings, engine, server, [
            patch.object(sys, "argv", ["launcher", *arguments]),
            patch("model_information_qwen.launcher.load_settings", return_value=settings),
            patch("model_information_qwen.launcher.check_local", return_value=(TOKEN, {"version": "v1.0.0"})),
            patch("model_information_qwen.launcher.check_port"),
            patch("model_information_qwen.launcher.prepare"),
            patch("model_information_qwen.launcher.pi_action", return_value={"state": "started"}),
            patch("model_information_qwen.model.load_chat_engine", return_value=engine),
            patch("model_information_qwen.server.InferenceServer", return_value=server),
            patch("model_information_qwen.launcher.active_server_status", return_value="ready", create=True),
            patch("model_information_qwen.launcher.follow_dialogue", create=True),
        ]

    def test_settings_prefer_original_zerotier_token_without_reading_or_rewriting_it(self):
        original = self.root / "zerotier_chat/.env"
        fallback = self.root / "communication/.env"
        original.write_text("ZT_CHAT_TOKEN=fake-original\n", encoding="utf-8")
        fallback.write_text("ZT_CHAT_TOKEN=fake-fallback\n", encoding="utf-8")
        before = original.read_bytes(), fallback.read_bytes()
        settings = self.settings()
        self.assertEqual(settings.token_file, original)
        self.assertEqual(settings.host, "192.0.2.185")
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.ssh_key, self.home / ".ssh/hks_zerotier_pi_ed25519")
        self.assertEqual(before, (original.read_bytes(), fallback.read_bytes()))

    def test_settings_use_communication_fallback_when_original_file_is_absent(self):
        fallback = self.root / "communication/.env"
        fallback.write_text("ZT_CHAT_TOKEN=fake-fallback\n", encoding="utf-8")
        self.assertEqual(self.settings().token_file, fallback)

    def test_default_settings_leave_model_selection_to_inference_config(self):
        self.args.model_dir = None
        settings = self.settings()
        self.assertIsNone(settings.model_dir)
        self.assertEqual(settings.config_path, launcher.DEFAULT_CONFIG_PATH.resolve())

    def test_explicit_qwen_configuration_path_is_resolved(self):
        self.args.config = self.root / "nested/../qwen-config.json"
        self.assertEqual(self.settings().config_path, self.root / "qwen-config.json")

    def test_local_inspection_uses_selected_configuration_without_loading_model(self):
        self.args.model_dir = None
        settings = self.settings()
        info = {"version": "qwen-selected", "model_dir": str(launcher.DEFAULT_MODEL_DIR),
                "base_model_exists": True, "base_model_path": "/local/Qwen3-8B"}
        with patch.object(launcher, "load_token", return_value=TOKEN), \
             patch.object(launcher, "inspect_model_version", return_value=info) as inspect, \
             patch.object(launcher.importlib.util, "find_spec", return_value=object()), \
             patch("model_information_qwen.model.load_chat_engine") as load:
            self.assertEqual(launcher.check_local(settings), (TOKEN, info))
        inspect.assert_called_once_with(None, config_path=settings.config_path)
        load.assert_not_called()

    def test_explicit_token_model_and_network_paths_override_defaults(self):
        (self.root / "zerotier_chat/.env").write_text("unused fake original", encoding="utf-8")
        alternate = self.root / "alternate.env"
        alternate.write_text(self.network_path.read_text().replace("CHAT_PORT=9000", "CHAT_PORT=9123"), encoding="utf-8")
        self.args.network_config = alternate
        self.args.token_file = self.root / "nested/../explicit-fake.env"
        self.args.model_dir = self.root / "nested/../other-model"
        settings = self.settings()
        self.assertEqual(settings.token_file, self.root / "explicit-fake.env")
        self.assertEqual(settings.model_dir, self.root / "other-model")
        self.assertEqual(settings.port, 9123)

    def test_invalid_ssh_username_is_rejected_before_any_process(self):
        self.network_path.write_text(self.network_path.read_text().replace("PI_SSH_USER=aida", "PI_SSH_USER=aida;bad"), encoding="utf-8")
        with patch("model_information_qwen.launcher.subprocess.run") as run, self.assertRaisesRegex(ValueError, "用户名"):
            self.settings()
        run.assert_not_called()

    def test_occupied_ephemeral_port_is_clear_and_never_kills_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertNotEqual(port, 9000)
            with patch("model_information_qwen.launcher.os.kill", create=True) as kill, self.assertRaisesRegex(RuntimeError, "已被其他服务占用") as error:
                launcher.check_port("127.0.0.1", port)
            kill.assert_not_called()
            self.assertIn(str(port), str(error.exception))
            self.assertIn("/quit", str(error.exception))
            self.assertEqual(listener.getsockname()[1], port)
        launcher.check_port("127.0.0.1", port)

    def test_unavailable_bind_address_explains_mirrored_network(self):
        socket_instance = MagicMock()
        socket_instance.__enter__.return_value = socket_instance
        socket_instance.bind.side_effect = OSError(errno.EADDRNOTAVAIL, "fake missing local address")
        with patch("model_information_qwen.launcher.socket.socket", return_value=socket_instance), self.assertRaisesRegex(RuntimeError, "WSL 镜像网卡"):
            launcher.check_port("192.0.2.185", 9123)

    def test_startup_lock_blocks_second_instance_and_releases_same_inode(self):
        path = self.root / "runtime/server.lock"
        with launcher.startup_lock(path):
            self.assertEqual(path.read_text().strip(), str(os.getpid()))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o077, 0)
            inode = path.stat().st_ino
            with self.assertRaises(launcher.AlreadyRunning):
                with launcher.startup_lock(path):
                    self.fail("A second launcher acquired the active lifetime lock")
        with launcher.startup_lock(path):
            self.assertEqual(path.stat().st_ino, inode)
        self.assertTrue(path.is_file())

    def test_startup_lock_is_cross_process_and_released_after_exception(self):
        path = self.root / "runtime/cross-process.lock"
        source = """import sys
from pathlib import Path
from model_information_qwen.launcher import startup_lock, AlreadyRunning
try:
    with startup_lock(Path(sys.argv[1])):
        print('acquired')
except AlreadyRunning:
    print('busy')
    raise SystemExit(3)
"""
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        project = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(RuntimeError, "fake failure"):
            with launcher.startup_lock(path):
                result = subprocess.run([sys.executable, "-c", source, str(path)], cwd=project,
                                        env=environment, capture_output=True, text=True, timeout=2)
                self.assertEqual((result.returncode, result.stdout.strip()), (3, "busy"), result.stderr)
                raise RuntimeError("fake failure")
        with launcher.startup_lock(path):
            pass

    def test_model_lock_probe_never_creates_or_changes_runtime_records(self):
        path = self.root / "model_information_qwen/runs/server.lock"
        self.assertFalse(launcher.model_lock_held(path))
        self.assertFalse(path.parent.exists())
        with launcher.startup_lock(path):
            before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)
            self.assertTrue(launcher.model_lock_held(path))
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino))
        self.assertFalse(launcher.model_lock_held(path))
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino))

    def test_runtime_record_identifies_model_endpoint_without_storing_token(self):
        settings = self.settings()
        path = self.root / "model_information_qwen/runs/server.lock"
        with launcher.startup_lock(path) as descriptor:
            launcher.record_runtime(descriptor, settings, {"version": "v1.0.0"})
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], str(os.getpid()))
            self.assertEqual(json.loads(lines[1]), {"pid": os.getpid(), "host": settings.host,
                "port": settings.port, "model_dir": str(settings.model_dir), "version": "v1.0.0",
                "dialogue_log": str(launcher.DEFAULT_LOG_PATH)})
            self.assertNotIn(TOKEN, "\n".join(lines))

    def test_runtime_configuration_mismatch_is_rejected_before_connecting(self):
        settings = self.settings()
        path = self.root / "model_information_qwen/runs/server.lock"
        for key, value in (("host", "192.0.2.200"), ("port", 9123),
                           ("model_dir", "/different/model"), ("version", "V2.0.0")):
            with self.subTest(key=key), launcher.startup_lock(path) as descriptor:
                launcher.record_runtime(descriptor, settings, {"version": "v1.0.0"})
                lines = path.read_text(encoding="utf-8").splitlines()
                metadata = json.loads(lines[1])
                metadata[key] = value
                path.write_text(lines[0] + "\n" + json.dumps(metadata) + "\n", encoding="utf-8")
                with patch("model_information_qwen.launcher.socket.create_connection") as connect:
                    with self.assertRaisesRegex(RuntimeError, "其他模型或地址"):
                        launcher.active_server_status(settings, TOKEN, {"version": "v1.0.0"})
                connect.assert_not_called()

    def test_default_runtime_records_and_reuses_resolved_selected_model(self):
        self.args.model_dir = None
        settings = self.settings()
        selected = self.root / "model_qwen/runs/custom-selected"
        info = {"version": "v1.0.0", "model_dir": str(selected)}
        path = self.root / "model_information_qwen/runs/server.lock"
        with launcher.startup_lock(path) as descriptor:
            launcher.record_runtime(descriptor, settings, info)
            metadata = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
            self.assertEqual(metadata["model_dir"], str(selected))
            self.assertNotEqual(metadata["model_dir"], "None")
            with patch.object(launcher.socket, "create_connection", return_value=self.fake_model_connection()):
                self.assertEqual(launcher.active_server_status(settings, TOKEN, info), "ready")

    def fake_model_connection(self, **overrides):
        connection = MagicMock()
        connection.__enter__.return_value = connection
        def response_file(mode):
            hello = json.loads(connection.sendall.call_args.args[0])
            reply = {"type": "hello_ok", "protocol_version": 2, "model_ready": True,
                     "session_id": hello["session_id"], "model_version": "v1.0.0", **overrides}
            return io.BytesIO((json.dumps(reply) + "\n").encode("utf-8"))
        connection.makefile.side_effect = response_file
        return connection

    def test_legacy_running_model_is_verified_by_read_only_authenticated_v2_greeting(self):
        settings = self.settings()
        connection = self.fake_model_connection()
        with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), patch(
                "model_information_qwen.launcher.socket.create_connection", return_value=connection) as connect:
            self.assertEqual(launcher.active_server_status(settings, TOKEN, {"version": "v1.0.0"}), "ready")
        connect.assert_called_once_with((settings.host, settings.port), timeout=3)
        connection.sendall.assert_called_once()
        hello = json.loads(connection.sendall.call_args.args[0])
        self.assertEqual((hello["type"], hello["protocol_version"], hello["token"]), ("hello", 2, TOKEN))
        self.assertTrue(hello["session_id"])
        self.assertNotIn("text", hello)

    def test_running_model_greeting_rejects_wrong_version_and_auth_without_echoing_secret(self):
        settings = self.settings()
        for overrides, message in (({"model_version": "V2.0.0"}, "版本与本次选择不一致"),
                ({"type": "error", "code": "auth_failed", "message": TOKEN}, "口令与已有模型服务不一致"),
                ({"model_ready": False}, "未提供版本 2"),
                ({"session_id": "wrong-session"}, "未提供版本 2")):
            with self.subTest(overrides=overrides), patch("model_information_qwen.launcher.socket.create_connection",
                    return_value=self.fake_model_connection(**overrides)):
                with self.assertRaisesRegex(RuntimeError, message) as error:
                    launcher.active_server_status(settings, TOKEN, {"version": "v1.0.0"})
                self.assertNotIn(TOKEN, str(error.exception))

    def test_running_model_connection_refused_reports_starting_without_loading_weights(self):
        settings = self.settings()
        with patch("model_information_qwen.launcher.socket.create_connection", side_effect=ConnectionRefusedError):
            self.assertEqual(launcher.active_server_status(settings, TOKEN, {"version": "v1.0.0"}), "starting")

    def test_pi_operation_lock_waits_for_previous_starter_to_finish(self):
        path = self.root / "model_information_qwen/runs/pi-start.lock"
        waiting, acquired = threading.Event(), threading.Event()
        errors = []
        def contender():
            waiting.set()
            try:
                with launcher.pi_operation_lock(path, timeout=1):
                    acquired.set()
            except Exception as error:
                errors.append(error)
        with launcher.pi_operation_lock(path):
            thread = threading.Thread(target=contender, daemon=True)
            thread.start()
            self.assertTrue(waiting.wait(1))
            self.assertFalse(acquired.wait(.05))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(acquired.is_set())

    def test_busy_pi_operation_times_out_before_a_second_ssh_is_launched(self):
        settings = self.pi_settings()
        path = self.root / "model_information_qwen/runs/pi-start.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        pi_lock = launcher.pi_operation_lock
        with path.open("w") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch("model_information_qwen.launcher.pi_operation_lock", side_effect=lambda lock: pi_lock(lock, timeout=.01)):
                with patch("model_information_qwen.launcher.subprocess.run") as run:
                    with self.assertRaisesRegex(RuntimeError, "另一个命令仍在启动树莓派"):
                        launcher.pi_action(settings, TOKEN, start=True)
                run.assert_not_called()
            self.assertTrue(launcher.model_lock_held(path))

    def test_pi_ssh_uses_verified_identity_and_hash_stdin_without_raw_secret(self):
        settings = self.pi_settings()
        response = subprocess.CompletedProcess([], 0, json.dumps({"state": "running"}), "")
        with patch("model_information_qwen.launcher.subprocess.run", return_value=response) as run:
            result = launcher.pi_action(settings, TOKEN)
        self.assertEqual(result, {"state": "running"})
        argv = run.call_args.args[0]
        self.assertEqual(argv[0:2], ["ssh", "-T"])
        for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "IdentitiesOnly=yes", "ConnectTimeout=5"):
            self.assertIn(option, argv)
        self.assertIn("UserKnownHostsFile=" + str(settings.known_hosts), argv)
        self.assertIn(str(settings.ssh_key), argv)
        self.assertIn("aida@192.0.2.95", argv)
        expected_hash = hashlib.sha256(TOKEN.encode()).hexdigest()
        self.assertNotIn(TOKEN, " ".join(argv))
        self.assertNotIn(expected_hash, " ".join(argv))
        payload = json.loads(run.call_args.kwargs["input"])
        self.assertEqual(payload["expected_token_sha256"], expected_hash)
        self.assertNotIn(TOKEN, run.call_args.kwargs["input"])
        self.assertEqual(payload["action"], "check")
        self.assertEqual((payload["host"], payload["port"]), (settings.host, settings.port))
        self.assertGreater(run.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 60)

    def test_pi_start_action_accepts_started_and_existing_running_states(self):
        settings = self.pi_settings()
        for state in ("started", "running"):
            with self.subTest(state=state), patch("model_information_qwen.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps({"state": state}), "")) as run:
                self.assertEqual(launcher.pi_action(settings, TOKEN, start=True)["state"], state)
                self.assertEqual(json.loads(run.call_args.kwargs["input"])["action"], "start")

    def test_pi_check_reports_startable_without_starting_software(self):
        settings = self.pi_settings()
        with patch("model_information_qwen.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"state":"startable"}', "")) as run:
            self.assertEqual(launcher.pi_action(settings, TOKEN)["state"], "startable")
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["action"], "check")
        self.assertFalse((self.root / "model_information_qwen/runs/pi-start.lock").exists())

    def test_pi_helper_failure_timeout_and_invalid_status_fail_clearly(self):
        settings = self.pi_settings()
        with patch("model_information_qwen.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "fake remote helper failure")), self.assertRaisesRegex(RuntimeError, "树莓派准备失败"):
            launcher.pi_action(settings, TOKEN)
        with patch("model_information_qwen.launcher.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 25)), self.assertRaisesRegex(RuntimeError, "超时"):
            launcher.pi_action(settings, TOKEN)
        for output in ("not JSON", "[]", '{"state":"unknown"}'):
            with self.subTest(output=output), patch("model_information_qwen.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")), self.assertRaises(RuntimeError):
                launcher.pi_action(settings, TOKEN)

    def test_missing_ssh_files_fail_before_subprocess(self):
        settings = self.pi_settings()
        settings.ssh_key.unlink()
        with patch("model_information_qwen.launcher.subprocess.run") as run, self.assertRaisesRegex(ValueError, "SSH 密钥"):
            launcher.pi_action(settings, TOKEN)
        run.assert_not_called()

    def test_check_mode_is_read_only_and_never_prepares_network_loads_gpu_or_starts_pi(self):
        settings, engine, server, patches = self.main_mocks(("--check",))
        mocks = [item.start() for item in patches]
        try:
            mocks[5].return_value = {"state": "startable"}
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(launcher.main(), 0)
            mocks[3].assert_called_once_with(settings.host, settings.port)
            mocks[4].assert_not_called()
            mocks[5].assert_called_once_with(settings, TOKEN)
            mocks[6].assert_not_called()
            mocks[7].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
            self.assertNotIn(TOKEN, output.getvalue())
            self.assertFalse((self.root / "model_information_qwen/runs/server.lock").exists())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_default_mode_prepares_then_starts_pi_before_loading_model_and_serving(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            events = []
            mocks[2].side_effect = lambda settings: (events.append("check_local") or (TOKEN, {"version": "v1.0.0"}))
            mocks[3].side_effect = lambda host, port: events.append("check_port")
            mocks[4].side_effect = lambda network: events.append("prepare")
            mocks[5].side_effect = lambda *args, **kwargs: (events.append("pi_start") or {"state": "started"})
            mocks[6].side_effect = lambda model_dir, *, config_path: (events.append("load_engine") or engine)
            async def run(host, port):
                events.append("serve")
            server.run.side_effect = run
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            self.assertEqual(events, ["check_local", "check_port", "prepare", "pi_start", "load_engine", "serve"])
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[8].assert_not_called()
            mocks[6].assert_called_once_with(settings.model_dir, config_path=settings.config_path)
            mocks[7].assert_called_once_with(engine, TOKEN, model_version="v1.0.0")
            server.run.assert_awaited_once_with(settings.host, settings.port)
            mocks[9].assert_not_called()
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"):
                pass  # Launcher released the lock after server exit.
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_starts_missing_pi_without_reloading_or_rebinding_model(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            events = []
            mocks[2].side_effect = lambda settings: (events.append("check_local") or (TOKEN, {"version": "v1.0.0"}))
            mocks[4].side_effect = lambda network: events.append("prepare")
            mocks[5].side_effect = lambda *args, **kwargs: (events.append("pi_start") or {"state": "started"})
            output = io.StringIO()
            path = self.root / "model_information_qwen/runs/server.lock"
            with launcher.startup_lock(path), redirect_stdout(output):
                self.assertEqual(launcher.main(), 0)
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Reusing the server must preserve its lifetime lock")
            self.assertEqual(events, ["check_local", "prepare", "pi_start"])
            mocks[2].assert_called_once_with(settings)
            mocks[8].assert_called_once_with(settings, TOKEN, {"version": "v1.0.0"})
            mocks[3].assert_not_called()
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            for index in (6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_called_once_with(settings)
            self.assertNotIn(TOKEN, output.getvalue())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_keeps_already_running_pi_and_does_not_load_model_again(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[5].return_value = {"state": "running"}
            output = io.StringIO()
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(output):
                self.assertEqual(launcher.main(), 0)
            mocks[2].assert_called_once_with(settings)
            mocks[3].assert_not_called()
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[6].assert_not_called()
            mocks[7].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_called_once_with(settings)
            self.assertIn("保留当前软件", output.getvalue())
            self.assertNotIn(TOKEN, output.getvalue())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_pi_ssh_failure_is_reported_and_keeps_existing_server_lock(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[5].side_effect = RuntimeError("fake Pi SSH connection failed")
            path = self.root / "model_information_qwen/runs/server.lock"
            output, errors = io.StringIO(), io.StringIO()
            with launcher.startup_lock(path):
                inode = path.stat().st_ino
                with redirect_stdout(output), redirect_stderr(errors), self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 1)
                self.assertEqual(path.stat().st_ino, inode)
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Failed Pi startup must not release the existing server lock")
            self.assertIn("fake Pi SSH connection failed", errors.getvalue())
            self.assertNotIn("已启动", output.getvalue())
            self.assertNotIn("已运行，保留当前软件", output.getvalue())
            mocks[3].assert_not_called()
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[6].assert_not_called()
            mocks[7].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_server_only_with_running_server_skips_pi_network_and_model_work(self):
        settings, engine, server, patches = self.main_mocks(("--server-only",))
        mocks = [item.start() for item in patches]
        try:
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            for index in (2, 3, 4, 5, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_called_once_with(settings)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_no_follow_reuses_server_and_starts_pi_without_waiting_in_console(self):
        settings, engine, server, patches = self.main_mocks(("--no-follow",))
        mocks = [item.start() for item in patches]
        try:
            path = self.root / "model_information_qwen/runs/server.lock"
            with launcher.startup_lock(path), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Ensure-only startup must preserve the existing server lock")
            mocks[8].assert_called_once_with(settings, TOKEN, {"version": "v1.0.0"})
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[9].assert_not_called()
            for index in (3, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_no_follow_server_only_with_running_server_returns_without_remote_actions(self):
        settings, engine, server, patches = self.main_mocks(("--server-only", "--no-follow"))
        mocks = [item.start() for item in patches]
        try:
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            for index in (2, 3, 4, 5, 6, 7, 8, 9):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_interrupting_reused_console_preserves_server_lock_and_does_not_reload_model(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[9].side_effect = KeyboardInterrupt
            path = self.root / "model_information_qwen/runs/server.lock"
            with launcher.startup_lock(path), redirect_stdout(io.StringIO()):
                before = (path.read_bytes(), path.stat().st_ino)
                self.assertEqual(launcher.main(), 0)
                self.assertEqual(before, (path.read_bytes(), path.stat().st_ino))
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Stopping the console must not stop the existing model server")
            mocks[9].assert_called_once_with(settings)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            for index in (3, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_check_with_running_server_is_read_only_and_does_not_bind_occupied_port(self):
        settings, engine, server, patches = self.main_mocks(("--check",))
        mocks = [item.start() for item in patches]
        try:
            mocks[5].return_value = {"state": "startable"}
            path = self.root / "model_information_qwen/runs/server.lock"
            output = io.StringIO()
            with launcher.startup_lock(path), redirect_stdout(output):
                inode = path.stat().st_ino
                self.assertEqual(launcher.main(), 0)
                self.assertEqual(path.stat().st_ino, inode)
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Read-only checks must preserve the active server lock")
            mocks[2].assert_called_once_with(settings)
            mocks[3].assert_not_called()
            mocks[4].assert_not_called()
            mocks[5].assert_called_once_with(settings, TOKEN)
            mocks[6].assert_not_called()
            mocks[7].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
            self.assertIn("startable", output.getvalue())
            self.assertNotIn(TOKEN, output.getvalue())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_check_server_only_with_running_server_checks_local_without_remote_actions(self):
        settings, engine, server, patches = self.main_mocks(("--check", "--server-only"))
        mocks = [item.start() for item in patches]
        try:
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            mocks[2].assert_called_once_with(settings)
            for index in (3, 4, 5, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_network_prepare_failure_does_not_start_pi_or_reload_model(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[4].side_effect = RuntimeError("fake network preparation failed")
            path = self.root / "model_information_qwen/runs/server.lock"
            with launcher.startup_lock(path), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 1)
                with self.assertRaises(launcher.AlreadyRunning):
                    with launcher.startup_lock(path):
                        self.fail("Failed Pi preparation must preserve the active server lock")
            mocks[2].assert_called_once_with(settings)
            mocks[3].assert_not_called()
            mocks[4].assert_called_once_with(settings.network)
            for index in (5, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_configuration_mismatch_prevents_pi_and_network_changes(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[8].side_effect = RuntimeError("fake active model version mismatch")
            output = io.StringIO()
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(output), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
            self.assertEqual(error.exception.code, 1)
            mocks[2].assert_called_once_with(settings)
            mocks[8].assert_called_once_with(settings, TOKEN, {"version": "v1.0.0"})
            for index in (3, 4, 5, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_not_called()
            self.assertNotIn("已启动", output.getvalue())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_running_server_still_loading_can_start_pi_without_claiming_model_ready(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[8].return_value = "starting"
            output = io.StringIO()
            with launcher.startup_lock(self.root / "model_information_qwen/runs/server.lock"), redirect_stdout(output):
                self.assertEqual(launcher.main(), 0)
            mocks[8].assert_called_once_with(settings, TOKEN, {"version": "v1.0.0"})
            mocks[3].assert_not_called()
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[6].assert_not_called()
            mocks[7].assert_not_called()
            server.run.assert_not_awaited()
            mocks[9].assert_called_once_with(settings)
            self.assertNotIn("已就绪", output.getvalue())
            self.assertNotIn(TOKEN, output.getvalue())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_server_only_skips_pi_but_keeps_network_and_model_startup(self):
        settings, engine, server, patches = self.main_mocks(("--server-only",))
        mocks = [item.start() for item in patches]
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_not_called()
            mocks[6].assert_called_once_with(settings.model_dir, config_path=settings.config_path)
            server.run.assert_awaited_once_with(settings.host, settings.port)
            mocks[9].assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_port_preflight_failure_prevents_pi_network_and_model_side_effects(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            mocks[3].side_effect = RuntimeError("fake occupied port")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                launcher.main()
            self.assertEqual(error.exception.code, 1)
            for index in (4, 5, 6, 7):
                mocks[index].assert_not_called()
            mocks[9].assert_not_called()
        finally:
            for item in reversed(patches):
                item.stop()

    def test_shell_launcher_keeps_qwen_python_priority_utf8_and_arguments(self):
        fake_python = self.root / "fake-python"
        fake_python.write_text(f"#!{sys.executable}\n" +
            "import json, os, sys\n" +
            "print(json.dumps({'argv':sys.argv[1:], 'utf8':os.environ.get('PYTHONUTF8'), " +
            "'encoding':os.environ.get('PYTHONIOENCODING')}))\n", encoding="utf-8")
        fake_python.chmod(0o700)
        script = Path(__file__).resolve().parents[1] / "start_server.sh"
        arguments = ["--check", "--server-only", "--no-follow", "--config", str(self.root / "qwen-config.json")]
        environment = {**os.environ, "QWEN_PYTHON": str(fake_python), "MEME_PYTHON": "/missing/python"}
        result = subprocess.run(["bash", str(script), *arguments], env=environment,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["argv"], ["-m", "model_information_qwen.launcher", *arguments])
        self.assertEqual((payload["utf8"], payload["encoding"]), ("1", "utf-8"))


if __name__ == "__main__":
    unittest.main()
