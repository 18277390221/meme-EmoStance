"""Launcher checks using fake credentials/SSH/model and ephemeral loopback ports."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import errno
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
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from model_information import launcher


TOKEN = "fake-local-launcher-token-123"


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project"
        (self.root / "zerotier_chat").mkdir(parents=True)
        (self.root / "communication").mkdir()
        (self.root / "model_information").mkdir()
        self.home = Path(self.temporary.name) / "home"
        (self.home / ".ssh").mkdir(parents=True)
        self.network_path = self.root / "zerotier_chat/network.env"
        self.network_path.write_text("\n".join([
            "PI_ZT_IP=192.0.2.95", "WINDOWS_ZT_IP=192.0.2.185", "WSL_ZT_IP=192.0.2.185",
            "CHAT_PORT=9000", "PI_SSH_USER=aida", "WSL_DISTRO_NAME=Ubuntu",
        ]) + "\n", encoding="utf-8")
        self.args = argparse.Namespace(network_config=self.network_path, model_dir=self.root / "archive/v1.0.0", token_file=None)
        self.project_patch = patch("model_information.launcher.PROJECT_ROOT", self.root)
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
        (self.root / "model_information/pi_startup.py").write_text("import sys\nprint('fake helper')\n", encoding="utf-8")
        return settings

    def main_mocks(self, arguments=()):
        """Return patches; every potentially external effect is explicitly fake."""
        settings = self.settings()
        engine = object()
        server = MagicMock()
        server.run = AsyncMock()
        return settings, engine, server, [
            patch.object(sys, "argv", ["launcher", *arguments]),
            patch("model_information.launcher.load_settings", return_value=settings),
            patch("model_information.launcher.check_local", return_value=(TOKEN, {"version": "v1.0.0"})),
            patch("model_information.launcher.check_port"),
            patch("model_information.launcher.prepare"),
            patch("model_information.launcher.pi_action", return_value={"state": "started"}),
            patch("model_information.model.load_chat_engine", return_value=engine),
            patch("model_information.server.InferenceServer", return_value=server),
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
        with patch("model_information.launcher.subprocess.run") as run, self.assertRaisesRegex(ValueError, "用户名"):
            self.settings()
        run.assert_not_called()

    def test_occupied_ephemeral_port_is_clear_and_never_kills_listener(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            self.assertNotEqual(port, 9000)
            with patch("model_information.launcher.os.kill", create=True) as kill, self.assertRaisesRegex(RuntimeError, "已被其他服务占用") as error:
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
        with patch("model_information.launcher.socket.socket", return_value=socket_instance), self.assertRaisesRegex(RuntimeError, "WSL 镜像网卡"):
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
from model_information.launcher import startup_lock, AlreadyRunning
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

    def test_pi_ssh_uses_verified_identity_and_hash_stdin_without_raw_secret(self):
        settings = self.pi_settings()
        response = subprocess.CompletedProcess([], 0, json.dumps({"state": "running"}), "")
        with patch("model_information.launcher.subprocess.run", return_value=response) as run:
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
            with self.subTest(state=state), patch("model_information.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps({"state": state}), "")) as run:
                self.assertEqual(launcher.pi_action(settings, TOKEN, start=True)["state"], state)
                self.assertEqual(json.loads(run.call_args.kwargs["input"])["action"], "start")

    def test_pi_check_reports_startable_without_starting_software(self):
        settings = self.pi_settings()
        with patch("model_information.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"state":"startable"}', "")) as run:
            self.assertEqual(launcher.pi_action(settings, TOKEN)["state"], "startable")
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["action"], "check")

    def test_pi_helper_failure_timeout_and_invalid_status_fail_clearly(self):
        settings = self.pi_settings()
        with patch("model_information.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "fake remote helper failure")), self.assertRaisesRegex(RuntimeError, "树莓派准备失败"):
            launcher.pi_action(settings, TOKEN)
        with patch("model_information.launcher.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 25)), self.assertRaisesRegex(RuntimeError, "超时"):
            launcher.pi_action(settings, TOKEN)
        for output in ("not JSON", "[]", '{"state":"unknown"}'):
            with self.subTest(output=output), patch("model_information.launcher.subprocess.run", return_value=subprocess.CompletedProcess([], 0, output, "")), self.assertRaises(RuntimeError):
                launcher.pi_action(settings, TOKEN)

    def test_missing_ssh_files_fail_before_subprocess(self):
        settings = self.pi_settings()
        settings.ssh_key.unlink()
        with patch("model_information.launcher.subprocess.run") as run, self.assertRaisesRegex(ValueError, "SSH 密钥"):
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
            self.assertNotIn(TOKEN, output.getvalue())
            self.assertFalse((self.root / "model_information/runs/server.lock").exists())
        finally:
            for item in reversed(patches):
                item.stop()

    def test_default_mode_prepares_then_starts_pi_before_loading_model_and_serving(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            events = []
            mocks[4].side_effect = lambda network: events.append("prepare")
            mocks[5].side_effect = lambda *args, **kwargs: (events.append("pi_start") or {"state": "started"})
            mocks[6].side_effect = lambda model_dir: (events.append("load_engine") or engine)
            async def run(host, port):
                events.append("serve")
            server.run.side_effect = run
            with redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            self.assertEqual(events, ["prepare", "pi_start", "load_engine", "serve"])
            mocks[4].assert_called_once_with(settings.network)
            mocks[5].assert_called_once_with(settings, TOKEN, start=True)
            mocks[6].assert_called_once_with(settings.model_dir)
            mocks[7].assert_called_once_with(engine, TOKEN)
            server.run.assert_awaited_once_with(settings.host, settings.port)
            with launcher.startup_lock(self.root / "model_information/runs/server.lock"):
                pass  # Launcher released the lock after server exit.
        finally:
            for item in reversed(patches):
                item.stop()

    def test_already_running_returns_without_model_load_or_pi_or_network_actions(self):
        settings, engine, server, patches = self.main_mocks()
        mocks = [item.start() for item in patches]
        try:
            with launcher.startup_lock(self.root / "model_information/runs/server.lock"), redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(), 0)
            for index in (2, 3, 4, 5, 6, 7):
                mocks[index].assert_not_called()
            server.run.assert_not_awaited()
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
            mocks[6].assert_called_once_with(settings.model_dir)
            server.run.assert_awaited_once_with(settings.host, settings.port)
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
        finally:
            for item in reversed(patches):
                item.stop()


if __name__ == "__main__":
    unittest.main()
