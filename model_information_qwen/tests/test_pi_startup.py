"""Mocked checks never contact or start a real Raspberry Pi."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from model_information_qwen import pi_startup as startup


TOKEN = "mock-only-pi-startup-shared-token"
DIGEST = hashlib.sha256(TOKEN.encode()).hexdigest()


class PiStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="pi-startup-test-")
        self.addCleanup(temporary.cleanup)
        self.home = Path(temporary.name)
        self.software = self.home / "meme/meme-software"
        self.software.mkdir(parents=True)
        token = self.home / "zerotier_chat/.env"
        token.parent.mkdir()
        token.write_text("ZT_CHAT_TOKEN=" + TOKEN + "\n")
        self.settings = {"action": "check", "host": startup.DEFAULT_HOST, "port": 9000,
                         "expected_token_sha256": DIGEST, "software_dir": str(self.software)}
        self.patchers = [
            patch.object(startup.Path, "home", return_value=self.home),
            patch.object(startup, "_check_resources"),
            patch.object(startup, "_processes", return_value=[]),
            patch.object(startup, "_player_health", return_value=False),
            patch.object(startup, "_select_asr_root", return_value=self.home / "installed-asr"),
            patch.object(startup, "_desktop_environment", return_value={"DISPLAY": ":0", "PYTHONDONTWRITEBYTECODE": "1"}),
            patch.object(startup.subprocess, "run", return_value=Mock(returncode=0)),
            patch.object(startup.subprocess, "Popen"),
        ]
        self.mocks = [patcher.start() for patcher in self.patchers]
        for patcher in self.patchers:
            self.addCleanup(patcher.stop)
        (self.resources, self.processes, self.health, self.asr, self.desktop,
         self.resource_check, self.popen) = self.mocks[1:]

    def app_process(self, *extra):
        return {"pid": 123, "comm": "python3", "argv": ["/usr/bin/python3", "app.py", "--port", "8784", *extra],
                "cwd": self.software / "player/desktop", "desktop": {"DISPLAY": ":0"}}

    def test_token_check_never_returns_token_or_digest(self):
        result = startup.run(self.settings)
        self.assertEqual(result["state"], "startable")
        serialized = json.dumps(result)
        self.assertNotIn(TOKEN, serialized)
        self.assertNotIn(DIGEST, serialized)
        self.popen.assert_not_called()

    def test_mismatched_token_stops_before_any_process_or_resource_action(self):
        self.settings["expected_token_sha256"] = "0" * 64
        with self.assertRaisesRegex(startup.StartupError, "口令不一致") as raised:
            startup.run(self.settings)
        self.assertNotIn(TOKEN, str(raised.exception))
        self.assertNotIn(DIGEST, str(raised.exception))
        self.resources.assert_not_called()
        self.processes.assert_not_called()
        self.popen.assert_not_called()

    def test_running_player_is_preserved_for_start_action(self):
        self.settings["action"] = "start"
        self.processes.return_value = [self.app_process()]
        self.health.return_value = True
        result = startup.run(self.settings)
        self.assertEqual((result["state"], result["pid"]), ("running", 123))
        self.popen.assert_not_called()
        self.desktop.assert_not_called()

    def test_running_player_without_ready_api_is_not_restarted(self):
        self.processes.return_value = [self.app_process()]
        with self.assertRaisesRegex(startup.StartupError, "已保留进程"):
            startup.run(self.settings)
        self.popen.assert_not_called()

    def test_other_port_player_does_not_claim_unrelated_ready_api(self):
        self.processes.return_value = [self.app_process("--port", "8785")]
        self.health.return_value = True
        with self.assertRaisesRegex(startup.StartupError, "未提供 8784"):
            startup.run(self.settings)
        self.popen.assert_not_called()

    def test_cli_error_uses_stderr_and_does_not_return_secrets(self):
        self.settings["expected_token_sha256"] = "0" * 64
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(startup.sys, "stdin", io.StringIO(json.dumps(self.settings))), \
                patch.object(startup.sys, "stdout", stdout), patch.object(startup.sys, "stderr", stderr):
            self.assertEqual(startup.main(), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("口令不一致", stderr.getvalue())
        self.assertNotIn(TOKEN, stderr.getvalue())
        self.assertNotIn(DIGEST, stderr.getvalue())

    def test_unrelated_healthy_api_or_no_stt_player_prevents_duplicate_launch(self):
        self.health.return_value = True
        with self.assertRaisesRegex(startup.StartupError, "不属于指定软件"):
            startup.run(self.settings)
        self.processes.return_value = [self.app_process("--no-stt")]
        with self.assertRaisesRegex(startup.StartupError, "--no-stt"):
            startup.run(self.settings)
        self.popen.assert_not_called()

    def test_config_mismatch_is_not_silently_rewritten(self):
        config = self.software / "player/data/speech-config.json"
        config.parent.mkdir(parents=True)
        original = json.dumps({"host": "other-server", "port": 9000})
        config.write_text(original)
        with self.assertRaisesRegex(startup.StartupError, "地址/端口"):
            startup.run(self.settings)
        self.assertEqual(config.read_text(), original)
        self.popen.assert_not_called()

    def test_missing_desktop_or_invalid_resources_fail_without_start(self):
        self.desktop.side_effect = startup.StartupError("没有已登录 X 桌面")
        with self.assertRaisesRegex(startup.StartupError, "X 桌面"):
            startup.run(self.settings)
        self.desktop.side_effect = None
        self.resource_check.return_value = Mock(returncode=1)
        with self.assertRaisesRegex(startup.StartupError, "依赖或素材检查失败"):
            startup.run(self.settings)
        self.popen.assert_not_called()

    def test_start_is_detached_and_waits_for_api(self):
        self.settings["action"] = "start"
        self.health.side_effect = [False, False, True]
        child = Mock(pid=456)
        child.poll.return_value = None
        self.popen.return_value = child
        with patch.object(startup.time, "sleep"):
            result = startup.run(self.settings)
        self.assertEqual(result["state"], "started")
        self.assertEqual(result["pid"], 456)
        args, kwargs = self.popen.call_args
        self.assertEqual(args[0], ["/bin/sh", str(self.software / "player/desktop/start.sh"),
                                    "--stt-root", str(self.home / "installed-asr"),
                                    "--end-silence", "0.7"])
        # Older wrappers prepend their 0.5-second default before "$@".
        # Argparse must see the explicit startup value last and use 0.7.
        parser = argparse.ArgumentParser()
        parser.add_argument("--end-silence", type=float, default=1.2)
        effective, _ = parser.parse_known_args(["--end-silence", "0.5", *args[0][2:]])
        self.assertEqual(effective.end_silence, .7)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(Path(result["log_file"]).stat().st_mode & 0o777, 0o600)

    def test_child_exit_is_reported_and_not_retried(self):
        self.settings["action"] = "start"
        child = Mock()
        child.poll.return_value = 1
        self.popen.return_value = child
        with self.assertRaisesRegex(startup.StartupError, "启动后退出"):
            startup.run(self.settings)
        self.assertEqual(self.popen.call_count, 1)


class ResourceDiscoveryTests(unittest.TestCase):
    def test_bundled_asr_is_preferred_and_ambiguous_fallback_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pi-asr-discovery-") as directory:
            software = Path(directory) / "meme-software"
            software.mkdir()
            with patch.object(startup, "_valid_asr_root", return_value=True):
                self.assertEqual(startup._select_asr_root(software), software / "stt")
            for name in ("offline-asr-upgrade-install-one", "offline-asr-upgrade-install-two"):
                (Path(directory) / name / "offline-asr-upgrade").mkdir(parents=True)
            with patch.object(startup, "_valid_asr_root", side_effect=lambda path: path.name == "offline-asr-upgrade"):
                with self.assertRaisesRegex(startup.StartupError, "多份"):
                    startup._select_asr_root(software)

    def test_invalid_configured_asr_does_not_guess_another_installation(self):
        with patch.object(startup, "_valid_asr_root", return_value=False):
            with self.assertRaisesRegex(startup.StartupError, "已配置的 STT"):
                startup._select_asr_root(Path("/some/software"), "/selected/asr")

    def test_missing_desktop_fails_without_spawning_or_guessing_ssh_display(self):
        with patch.object(startup.Path, "exists", return_value=False), patch.object(startup.subprocess, "run") as probe:
            with self.assertRaisesRegex(startup.StartupError, "已登录 X 桌面"):
                startup._desktop_environment([{"desktop": {"DISPLAY": "localhost:10.0"}}])
            probe.assert_not_called()

    def test_token_file_uses_literal_parsing_without_shell_execution(self):
        with tempfile.TemporaryDirectory(prefix="pi-token-literal-") as directory:
            path = Path(directory) / "config"
            path.write_text("# ignored\nZT_CHAT_TOKEN=$(echo never-execute)\n")
            with patch.object(startup.subprocess, "run") as command:
                digest = startup._read_shared_token_hash(path)
            self.assertEqual(digest, hashlib.sha256(b"$(echo never-execute)").hexdigest())
            command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
