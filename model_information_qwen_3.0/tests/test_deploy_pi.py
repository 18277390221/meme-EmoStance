import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_information_qwen_v3 import deploy_pi
from model_information_qwen_v3.launcher import LaunchSettings


class RemoteDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "software"
        self.directory = self.root / "player/desktop"
        self.directory.mkdir(parents=True)
        self.state = Path(self.temporary.name) / "private-state"
        self.names = ["player/desktop/server_link.py", "player/desktop/conversation_turn.py"]
        self.original = {self.names[0]: b"old_link = True\n", self.names[1]: b"old_turn = True\n"}
        for name, data in self.original.items():
            (self.root / name).write_bytes(data)
        # Deliberately unrelated content must never be written or backed up.
        (self.root / ".env").write_text("private credential\n")
        (self.root / "player/data").mkdir()
        (self.root / "player/data/media.bin").write_bytes(b"local-media")
        self.namespace = {}
        exec(deploy_pi.REMOTE_HELPER, self.namespace)
        self.calls = []
        self.startup = {"run": self.startup_run, "_processes": lambda: [],
                        "_matches_script": lambda process, script: False}

    def startup_run(self, settings):
        self.calls.append(settings["action"])
        return {"state": "started" if settings["action"] == "start" else "startable",
                "software_dir": str(self.root)}

    def payload(self, action="check", *, restart=False):
        payload = {"action": action, "files": self.names,
                   "settings": {"software_dir": str(self.root)}, "restart": restart}
        if action == "apply":
            payload["expected_protected_files"] = self.namespace["_protected_hashes"](self.root)
            payload["entries"] = {name: {
                "expected_sha256": hashlib.sha256(self.original[name]).hexdigest(),
                "sha256": hashlib.sha256(b"new_version = 2\n").hexdigest(),
                "content": base64.b64encode(b"new_version = 2\n").decode("ascii")}
                for name in self.names}
        return payload

    def run_remote(self, payload):
        return self.namespace["remote_run"](payload, self.startup, self.state)

    def assert_original(self):
        for name, expected in self.original.items():
            self.assertEqual((self.root / name).read_bytes(), expected)
        self.assertEqual((self.root / ".env").read_text(), "private credential\n")
        self.assertEqual((self.root / "player/data/media.bin").read_bytes(), b"local-media")

    def test_check_does_not_create_backup_or_change_files(self):
        report = self.run_remote(self.payload())
        self.assertEqual(report["state"], "checked")
        self.assertEqual(report["files"][self.names[0]]["sha256"],
                         hashlib.sha256(self.original[self.names[0]]).hexdigest())
        self.assertFalse(self.state.exists())
        self.assert_original()

    def test_apply_backs_up_private_source_and_verifies_sha(self):
        (self.root / self.names[0]).chmod(0o640)
        report = self.run_remote(self.payload("apply"))
        self.assertEqual(report["state"], "applied")
        self.assertFalse(report["requires_restart"])
        backup = Path(report["backup_dir"])
        self.assertEqual(backup.stat().st_mode & 0o777, 0o700)
        for name in self.names:
            self.assertEqual((self.root / name).read_bytes(), b"new_version = 2\n")
            self.assertEqual((backup / name).read_bytes(), self.original[name])
            self.assertEqual((backup / name).stat().st_mode & 0o777, 0o600)
        self.assertEqual((self.root / self.names[0]).stat().st_mode & 0o777, 0o640)
        self.assertFalse((backup / ".env").exists())
        self.assertFalse((backup / "player/data/media.bin").exists())
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_changed_remote_sha_aborts_before_backup_or_mutation(self):
        payload = self.payload("apply")
        payload["entries"][self.names[1]]["expected_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "自检查后发生变化"):
            self.run_remote(payload)
        self.assert_original()
        self.assertFalse(self.state.exists())

    def test_changed_asr_selection_aborts_without_overwriting_user_config(self):
        config = self.directory / "asr-selection.json"
        config.write_text('{"profile":"paraformer"}')
        payload = self.payload("apply")
        config.write_text('{"profile":"standard"}')
        with self.assertRaisesRegex(RuntimeError, "ASR 配置"):
            self.run_remote(payload)
        self.assert_original()
        self.assertEqual(config.read_text(), '{"profile":"standard"}')
        self.assertFalse(self.state.exists())

    def test_compile_all_files_before_changing_first(self):
        payload = self.payload("apply")
        invalid = b"if (:\n"
        payload["entries"][self.names[1]].update({"content": base64.b64encode(invalid).decode(),
                                                "sha256": hashlib.sha256(invalid).hexdigest()})
        with self.assertRaises(SyntaxError):
            self.run_remote(payload)
        self.assert_original()
        self.assertFalse(self.state.exists())

    def test_failure_after_first_replace_restores_original(self):
        original_atomic = self.namespace["_atomic"]
        calls = 0

        def fail_second(path, data, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected replacement failure")
            original_atomic(path, data, mode)

        self.namespace["_atomic"] = fail_second
        report = self.run_remote(self.payload("apply"))
        self.assertEqual(report["state"], "error")
        self.assertTrue(report["rollback"])
        self.assert_original()
        self.assertEqual(report["processes"], [])

    def test_rollback_preserves_concurrent_source_edit_after_our_replace(self):
        original_atomic = self.namespace["_atomic"]
        calls = 0

        def concurrent_edit(path, data, mode):
            nonlocal calls
            calls += 1
            original_atomic(path, data, mode)
            if calls == 1:
                path.write_bytes(b"concurrent_site_fix = True\n")

        self.namespace["_atomic"] = concurrent_edit
        report = self.run_remote(self.payload("apply"))
        self.assertEqual(report["state"], "error")
        self.assertFalse(report["rollback"])
        self.assertIn("并发修改", report["rollback_errors"][0])
        self.assertEqual((self.root / self.names[0]).read_bytes(), b"concurrent_site_fix = True\n")
        self.assertEqual((self.root / self.names[1]).read_bytes(), self.original[self.names[1]])

    def test_partial_stop_failure_records_signaled_pids_and_survivors(self):
        survivor = {"pid": 102, "script": self.root / "player/desktop/stt_worker.py"}
        self.startup["_processes"] = lambda: [survivor]
        self.startup["_matches_script"] = lambda process, script: process["script"] == script

        def partial_failure(startup, root):
            failure = self.namespace["DeployFailure"]("partially stopped")
            failure.stopped_pids = [101]
            failure.stop_requested_pids = [101, 102]
            raise failure

        self.namespace["_stop_owned"] = partial_failure
        report = self.run_remote(self.payload("apply", restart=True))
        self.assertEqual(report["state"], "error")
        self.assertTrue(report["rollback"])
        self.assertEqual(report["stopped_pids"], [101])
        self.assertEqual(report["stop_requested_pids"], [101, 102])
        self.assertEqual(report["process_cleanup_error"], "partially stopped")
        self.assertEqual(report["processes"], [{"pid": 102, "kind": "stt"}])
        self.assert_original()

    def test_start_failure_rolls_back_and_reports_current_processes(self):
        def fail_start(settings):
            if settings["action"] == "start":
                raise RuntimeError("desktop startup failed")
            return {"state": "startable"}

        self.startup["run"] = fail_start
        report = self.run_remote(self.payload("apply", restart=True))
        self.assertEqual(report["state"], "error")
        self.assertTrue(report["rollback"])
        self.assertEqual(report["message"], "desktop startup failed")
        self.assertEqual(report["processes"], [])
        self.assert_original()

    def test_restart_only_after_successful_source_verification(self):
        report = self.run_remote(self.payload("apply", restart=True))
        self.assertEqual(report["state"], "applied")
        self.assertEqual(self.calls, ["check", "start"])
        self.assertEqual(report["software"]["state"], "started")

    def test_remote_rejects_credentials_even_if_payload_invents_allowlist(self):
        payload = self.payload()
        payload.update({"files": [".env"], "allowed_files": [".env"]})
        with self.assertRaisesRegex(RuntimeError, "白名单"):
            self.run_remote(payload)
        self.assert_original()

    def test_remote_rejects_symlink_before_any_mutation(self):
        (self.root / self.names[0]).unlink()
        (self.root / self.names[0]).symlink_to(self.root / ".env")
        with self.assertRaisesRegex(RuntimeError, "符号链接"):
            self.run_remote(self.payload("apply"))
        self.assertEqual((self.root / ".env").read_text(), "private credential\n")

    def test_base64_and_sha_must_both_be_valid(self):
        payload = self.payload("apply")
        payload["entries"][self.names[0]]["content"] = "invalid base64@@"
        with self.assertRaisesRegex(RuntimeError, "编码无效"):
            self.run_remote(payload)
        self.assert_original()

    def test_nested_speech_dependency_can_be_deployed(self):
        name = "player/desktop/asr_runtime/speech.py"
        path = self.root / name
        path.parent.mkdir()
        path.write_bytes(b"old_speech = True\n")
        self.names.append(name)
        self.original[name] = path.read_bytes()
        report = self.run_remote(self.payload("apply"))
        self.assertEqual(report["state"], "applied")
        self.assertEqual(path.read_bytes(), b"new_version = 2\n")
        self.assertEqual((Path(report["backup_dir"]) / name).read_bytes(), self.original[name])

    def test_recovery_deployment_adds_strategy_and_updates_capture_without_config_changes(self):
        recovery = "player/desktop/voice_recovery.py"
        audio = "player/desktop/asr_runtime/audio.py"
        audio_path = self.root / audio
        audio_path.parent.mkdir()
        audio_path.write_bytes(b"old_capture = True\n")
        self.names = [recovery, audio]
        report = self.run_remote(self.payload())
        self.assertIsNone(report["files"][recovery]["sha256"])
        self.assertFalse((self.root / recovery).exists())
        originals = {recovery: None, audio: audio_path.read_bytes()}
        data = b"recovery_enabled = True\n"
        payload = self.payload()
        payload.update(action="apply", expected_protected_files=self.namespace["_protected_hashes"](self.root),
                       entries={name: {"expected_sha256": (hashlib.sha256(old).hexdigest() if old else None),
                                       "sha256": hashlib.sha256(data).hexdigest(),
                                       "content": base64.b64encode(data).decode()}
                                for name, old in originals.items()})
        self.assertEqual(self.run_remote(payload)["state"], "applied")
        for name in self.names:
            self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual((self.root / ".env").read_text(), "private credential\n")
        self.assertEqual((self.root / "player/data/media.bin").read_bytes(), b"local-media")

    def test_music_package_check_is_readonly_and_apply_creates_only_package(self):
        name = "player/desktop/meme_sing/core.py"
        package = self.directory / "meme_sing"
        self.names = [name]
        report = self.run_remote(self.payload())
        self.assertIsNone(report["files"][name]["sha256"])
        self.assertFalse(package.exists())
        payload = self.payload()
        data = b"music_enabled = True\n"
        payload.update(action="apply", expected_protected_files=self.namespace["_protected_hashes"](self.root),
                       entries={name: {"expected_sha256": None, "sha256": hashlib.sha256(data).hexdigest(),
                                       "content": base64.b64encode(data).decode()}})
        self.assertEqual(self.run_remote(payload)["state"], "applied")
        self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual((self.root / ".env").read_text(), "private credential\n")

    def test_usb_output_module_first_install_is_reviewed_and_preserves_asr_config(self):
        name = "player/desktop/audio_output.py"
        self.names = [name]
        config = self.directory / "asr-selection.json"
        config.write_text('{"profile":"standard","end_silence":0.7}')
        protected = self.namespace["_protected_hashes"](self.root)
        report = self.run_remote(self.payload())
        self.assertIsNone(report["files"][name]["sha256"])
        self.assertFalse((self.root / name).exists())
        self.assertFalse(self.state.exists())
        data = b"usb_output_enabled = True\n"
        payload = self.payload()
        payload.update(action="apply", expected_protected_files=protected,
                       entries={name: {"expected_sha256": None,
                                       "sha256": hashlib.sha256(data).hexdigest(),
                                       "content": base64.b64encode(data).decode()}})
        self.assertEqual(self.run_remote(payload)["state"], "applied")
        self.assertEqual((self.root / name).read_bytes(), data)
        self.assertEqual(self.namespace["_protected_hashes"](self.root), protected)
        self.assertEqual((self.root / ".env").read_text(), "private credential\n")

    def test_music_package_symlink_is_rejected_without_writing_destination(self):
        (self.directory / "meme_sing").symlink_to(self.root / "player/data", target_is_directory=True)
        self.names = ["player/desktop/meme_sing/core.py"]
        with self.assertRaisesRegex(RuntimeError, "符号链接"):
            self.run_remote(self.payload())
        self.assertFalse((self.root / "player/data/core.py").exists())

    def standby_payload(self):
        self.names = ["player/desktop/meme_huanxing/" + filename for filename in
                      ("__init__.py", "commands.py", "controller.py", "screen.py", "README.md")]
        data = b"standby_enabled = True\n"
        payload = self.payload()
        payload.update(action="apply", expected_protected_files=self.namespace["_protected_hashes"](self.root),
                       entries={name: {"expected_sha256": None,
                                       "sha256": hashlib.sha256(data).hexdigest(),
                                       "content": base64.b64encode(data).decode()}
                                for name in self.names})
        return payload

    def test_standby_package_check_is_readonly(self):
        payload = self.standby_payload()
        payload["action"] = "check"
        before = set(self.root.rglob("*"))
        report = self.run_remote(payload)
        self.assertEqual(report["state"], "checked")
        self.assertTrue(all(item["sha256"] is None for item in report["files"].values()))
        self.assertFalse((self.directory / "meme_huanxing").exists())
        self.assertFalse(self.state.exists())
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assert_original()

    def test_standby_package_apply_creates_only_reviewed_package_and_files(self):
        payload = self.standby_payload()
        report = self.run_remote(payload)
        self.assertEqual(report["state"], "applied")
        self.assertEqual(set(report["changed_files"]), set(self.names))
        for name in self.names:
            self.assertEqual((self.root / name).read_bytes(), b"standby_enabled = True\n")
        self.assertFalse((self.directory / "meme_sing").exists())
        self.assertEqual(self.namespace["_protected_hashes"](self.root), payload["expected_protected_files"])
        self.assert_original()

    def test_standby_package_invalid_last_module_is_validated_before_directory_creation(self):
        payload = self.standby_payload()
        invalid = b"if (:\n"
        payload["entries"]["player/desktop/meme_huanxing/screen.py"].update(
            content=base64.b64encode(invalid).decode(), sha256=hashlib.sha256(invalid).hexdigest())
        with self.assertRaises(SyntaxError):
            self.run_remote(payload)
        self.assertFalse((self.directory / "meme_huanxing").exists())
        self.assertFalse(self.state.exists())
        self.assert_original()

    def test_standby_package_rollback_removes_only_new_empty_package(self):
        payload = self.standby_payload()
        atomic = self.namespace["_atomic"]
        calls = 0

        def fail_second(path, data, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected package replacement failure")
            atomic(path, data, mode)

        self.namespace["_atomic"] = fail_second
        report = self.run_remote(payload)
        self.assertEqual(report["state"], "error")
        self.assertTrue(report["rollback"])
        self.assertFalse((self.directory / "meme_huanxing").exists())
        self.assert_original()

    def test_standby_rollback_keeps_preexisting_package_and_unrelated_files(self):
        package = self.directory / "meme_huanxing"
        package.mkdir()
        marker = package / "local_site.txt"
        marker.write_bytes(b"keep existing content")
        payload = self.standby_payload()
        atomic = self.namespace["_atomic"]
        calls = 0

        def fail_second(path, data, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected package replacement failure")
            atomic(path, data, mode)

        self.namespace["_atomic"] = fail_second
        report = self.run_remote(payload)
        self.assertTrue(report["rollback"])
        self.assertEqual(marker.read_bytes(), b"keep existing content")
        self.assertEqual(list(package.iterdir()), [marker])

    def test_standby_rollback_keeps_file_concurrently_added_to_new_package(self):
        payload = self.standby_payload()
        package = self.directory / "meme_huanxing"
        marker = package / "concurrent_site.txt"
        atomic = self.namespace["_atomic"]
        calls = 0

        def fail_second(path, data, mode):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected package replacement failure")
            atomic(path, data, mode)
            marker.write_bytes(b"keep concurrent content")

        self.namespace["_atomic"] = fail_second
        report = self.run_remote(payload)
        self.assertTrue(report["rollback"])
        self.assertEqual(marker.read_bytes(), b"keep concurrent content")
        self.assertEqual(list(package.iterdir()), [marker])

    def test_standby_package_symlink_is_rejected_for_check_and_apply(self):
        payload = self.standby_payload()
        (self.directory / "meme_huanxing").symlink_to(self.root / "player/data", target_is_directory=True)
        for action in ("check", "apply"):
            with self.subTest(action=action), self.assertRaisesRegex(RuntimeError, "符号链接"):
                self.run_remote(dict(payload, action=action))
        self.assertFalse((self.root / "player/data/commands.py").exists())
        self.assertFalse(self.state.exists())
        self.assert_original()

    def test_standby_module_symlink_is_rejected_without_writing_destination(self):
        payload = self.standby_payload()
        package = self.directory / "meme_huanxing"
        package.mkdir()
        (package / "commands.py").symlink_to(self.root / "player/data/media.bin")
        with self.assertRaisesRegex(RuntimeError, "符号链接"):
            self.run_remote(payload)
        self.assertFalse((package / "__init__.py").exists())
        self.assertFalse(self.state.exists())
        self.assert_original()

    def test_standby_parent_is_file_is_rejected_before_any_mutation(self):
        payload = self.standby_payload()
        package = self.directory / "meme_huanxing"
        package.write_bytes(b"existing regular file")
        with self.assertRaisesRegex(RuntimeError, "目录不存在"):
            self.run_remote(payload)
        self.assertEqual(package.read_bytes(), b"existing regular file")
        self.assertFalse(self.state.exists())
        self.assert_original()

    def test_stop_only_targets_exact_owned_app_and_stt(self):
        processes = [{"pid": 101, "script": self.root / "player/desktop/app.py"},
                     {"pid": 102, "script": self.root / "player/desktop/stt_worker.py"},
                     {"pid": 103, "script": self.root / "other/app.py"}]
        self.startup["_processes"] = lambda: list(processes)
        self.startup["_matches_script"] = lambda process, script: process["script"] == script
        signaled = []

        def kill(pid, sig):
            signaled.append((pid, sig))
            processes[:] = [process for process in processes if process["pid"] != pid]

        with patch.object(self.namespace["os"], "kill", kill):
            stopped = self.namespace["_stop_owned"](self.startup, self.root)
        self.assertEqual(stopped, [101, 102])
        self.assertEqual([pid for pid, _ in signaled], [101, 102])
        self.assertTrue(all(sig == self.namespace["signal"].SIGTERM for _, sig in signaled))
        self.assertEqual(processes[0]["pid"], 103)

    def test_signal_timeout_does_not_escalate_to_force_kill(self):
        self.startup["_processes"] = lambda: [{"pid": 101, "script": self.root / "player/desktop/app.py"}]
        self.startup["_matches_script"] = lambda process, script: process["script"] == script
        signals = []
        with patch.object(self.namespace["os"], "kill", lambda pid, sig: signals.append(sig)), \
             patch.object(self.namespace["time"], "monotonic", side_effect=[0, 11]):
            with self.assertRaisesRegex(RuntimeError, "未强杀进程"):
                self.namespace["_stop_owned"](self.startup, self.root)
        self.assertEqual(signals, [self.namespace["signal"].SIGTERM])

    def test_file_added_by_deployment_can_be_rolled_back(self):
        name = self.names[1]
        (self.root / name).unlink()
        payload = self.payload("apply", restart=True)
        payload["entries"][name]["expected_sha256"] = None
        original_run = self.startup["run"]

        def fail_start(settings):
            if settings["action"] == "start":
                raise RuntimeError("start failed")
            return original_run(settings)

        self.startup["run"] = fail_start
        report = self.run_remote(payload)
        self.assertTrue(report["rollback"])
        self.assertFalse((self.root / name).exists())
        self.assertEqual((self.root / self.names[0]).read_bytes(), self.original[self.names[0]])


class LocalDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.token = "unit-test-existing-token-value"
        token_file = self.root / ".env"
        token_file.write_text("ZT_CHAT_TOKEN=" + self.token + "\n")
        self.settings = LaunchSettings({"WSL_ZT_IP": "10.0.0.1", "CHAT_PORT": "9000",
            "PI_ZT_IP": "10.0.0.2", "PI_SSH_USER": "aida"}, self.root, token_file,
            self.root / "key", self.root / "known_hosts")
        self.names = list(deploy_pi.DEFAULT_FILES[:2])
        for name in self.names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("new_version = 2\n")

    def test_local_apply_sends_token_hash_only_and_expected_remote_sha(self):
        requests = []

        def fake_remote(settings, payload):
            requests.append(json.loads(json.dumps(payload)))
            if payload["action"] == "check":
                return {"state": "checked", "files": {name: {"sha256": "a" * 64} for name in self.names}}
            return {"state": "applied"}

        with patch.object(deploy_pi, "_remote_call", fake_remote):
            report = deploy_pi.deploy(self.settings, files=self.names, apply=True,
                                      restart=True, source_root=self.root)
        self.assertEqual(report["state"], "applied")
        self.assertEqual([request["action"] for request in requests], ["check", "apply"])
        self.assertNotIn(self.token, json.dumps(requests))
        self.assertEqual(requests[0]["settings"]["expected_token_sha256"],
                         hashlib.sha256(self.token.encode()).hexdigest())
        self.assertEqual(requests[1]["entries"][self.names[0]]["expected_sha256"], "a" * 64)

    def test_local_rejects_non_source_and_duplicate_names(self):
        for files in ([".env"], self.names + self.names):
            with self.subTest(files=files), self.assertRaises(deploy_pi.DeploymentError):
                deploy_pi.deploy(self.settings, files=files)

    def test_restart_requires_apply(self):
        with self.assertRaisesRegex(deploy_pi.DeploymentError, "--restart"):
            deploy_pi.deploy(self.settings, files=self.names, restart=True)

    def test_default_manifest_includes_all_voice_handshake_dependencies(self):
        self.assertTrue({"player/desktop/capture_turn.py", "player/desktop/stt_bridge.py",
                         "player/desktop/stt_worker.py", "player/desktop/asr_runtime/speech.py",
                         "player/launch.py", "player/desktop/asr_selection.py", "player/desktop/start.sh"}
                        .issubset(deploy_pi.DEFAULT_FILES))
        namespace = {}
        exec(deploy_pi.REMOTE_HELPER, namespace)
        self.assertEqual(deploy_pi.ALLOWED_FILES, namespace["REMOTE_ALLOWED_FILES"])
        self.assertTrue({"player/desktop/directory_library.py", "player/desktop/native_core.py",
                         "player/desktop/render_pipeline.py"}.issubset(deploy_pi.DEFAULT_FILES))
        self.assertTrue({"player/desktop/voice_recovery.py", "player/desktop/audio_output.py",
                         "player/desktop/asr_runtime/audio.py"}
                        .issubset(deploy_pi.DEFAULT_FILES))

    def test_default_manifest_includes_complete_standby_package(self):
        required = {"player/desktop/meme_huanxing/" + filename for filename in
                    ("__init__.py", "commands.py", "controller.py", "screen.py", "README.md")}
        self.assertTrue(required.issubset(deploy_pi.DEFAULT_FILES))
        self.assertTrue(required.issubset(deploy_pi.ALLOWED_FILES))
        namespace = {}
        exec(deploy_pi.REMOTE_HELPER, namespace)
        self.assertTrue(required.issubset(namespace["REMOTE_ALLOWED_FILES"]))

    def test_default_source_uses_shared_standby_package_without_duplicate_copy(self):
        name = "player/desktop/meme_huanxing/commands.py"
        shared = self.root / "meme_huanxing/commands.py"
        shared.parent.mkdir()
        data = b"shared_standby_commands = True\n"
        shared.write_bytes(data)
        requests = []

        def fake_remote(settings, payload):
            requests.append(payload)
            return {"state": "checked" if payload["action"] == "check" else "applied",
                    "files": {name: {"sha256": None}}}

        with patch.object(deploy_pi, "PROJECT_ROOT", self.root), \
                patch.object(deploy_pi, "_remote_call", fake_remote):
            report = deploy_pi.deploy(self.settings, files=[name], apply=True)
        self.assertEqual(report["state"], "applied")
        self.assertEqual(base64.b64decode(requests[-1]["entries"][name]["content"]), data)
        self.assertFalse((self.root / "meme_software" / name).exists())

    def test_explicit_source_root_does_not_fallback_to_shared_standby_package(self):
        name = "player/desktop/meme_huanxing/commands.py"
        shared = self.root / "meme_huanxing/commands.py"
        shared.parent.mkdir()
        shared.write_bytes(b"shared_standby_commands = True\n")
        with patch.object(deploy_pi, "PROJECT_ROOT", self.root), \
                patch.object(deploy_pi, "_remote_call", return_value={"state": "checked",
                    "files": {name: {"sha256": None}}}) as remote:
            with self.assertRaisesRegex(deploy_pi.DeploymentError, "源码不存在"):
                deploy_pi.deploy(self.settings, files=[name], apply=True,
                                 source_root=self.root / "explicit-reviewed-stage")
        self.assertEqual(remote.call_count, 1)

    def test_default_source_prefers_present_reviewed_desktop_copy(self):
        name = "player/desktop/meme_huanxing/commands.py"
        shared = self.root / "meme_huanxing/commands.py"
        shared.parent.mkdir()
        shared.write_bytes(b"shared_standby_commands = True\n")
        desktop = self.root / "meme_software" / name
        desktop.parent.mkdir(parents=True)
        data = b"reviewed_desktop_commands = True\n"
        desktop.write_bytes(data)
        requests = []

        def fake_remote(settings, payload):
            requests.append(payload)
            return {"state": "checked" if payload["action"] == "check" else "applied",
                    "files": {name: {"sha256": None}}}

        with patch.object(deploy_pi, "PROJECT_ROOT", self.root), \
                patch.object(deploy_pi, "_remote_call", fake_remote):
            deploy_pi.deploy(self.settings, files=[name], apply=True)
        self.assertEqual(base64.b64decode(requests[-1]["entries"][name]["content"]), data)

    def test_remote_command_does_not_include_token_or_payload(self):
        self.settings.ssh_key.write_text("fake local key for existence check")
        self.settings.known_hosts.write_text("fake known host")
        captured = {}

        def fake_run(command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            return type("Result", (), {"stdout": '{"state":"checked"}', "returncode": 0})()

        payload = {"action": "check", "private_marker": "stdin-payload-marker"}
        with patch.object(deploy_pi.subprocess, "run", fake_run):
            deploy_pi._remote_call(self.settings, payload)
        self.assertNotIn("stdin-payload-marker", " ".join(captured["command"]))
        self.assertIn("stdin-payload-marker", captured["kwargs"]["input"])
        self.assertIn("StrictHostKeyChecking=yes", captured["command"])
        self.assertIn("IdentitiesOnly=yes", captured["command"])

    def test_remote_source_reads_only_the_qwen_startup_helper(self):
        helper = self.root / "model_information_qwen_3.0/pi_startup.py"
        helper.parent.mkdir()
        helper.write_text("QWEN_PI_STARTUP_HELPER_MARKER = True\n")
        legacy = self.root / "model_information/pi_startup.py"
        legacy.parent.mkdir()
        legacy.write_text("LEGACY_PI_STARTUP_HELPER_MARKER = True\n")
        with patch.object(deploy_pi, "PROJECT_ROOT", self.root):
            source = deploy_pi._remote_source()
        self.assertIn("QWEN_PI_STARTUP_HELPER_MARKER", source)
        self.assertNotIn("LEGACY_PI_STARTUP_HELPER_MARKER", source)

    def test_cli_passes_selected_qwen_config_without_loading_or_remote_apply(self):
        config = self.root / "selected_qwen.json"
        with patch.object(deploy_pi, "load_settings", return_value=self.settings) as load, \
                patch.object(deploy_pi, "deploy", return_value={"state": "checked"}) as inspect, \
                patch("sys.stdout", io.StringIO()):
            code = deploy_pi.main(["--check", "--config", str(config)])
        self.assertEqual(code, 0)
        self.assertEqual(load.call_args.args[0].config, config)
        self.assertIsNone(load.call_args.args[0].model_dir)
        self.assertFalse(inspect.call_args.kwargs["apply"])

    def test_reviewed_snapshot_mismatch_never_calls_mutation(self):
        calls = []

        def fake_remote(settings, payload):
            calls.append(payload["action"])
            return {"state": "checked", "files": {name: {"sha256": "a" * 64} for name in self.names},
                    "protected_files": {"player/desktop/asr-selection.json": "c" * 64}}

        with patch.object(deploy_pi, "_remote_call", fake_remote):
            with self.assertRaisesRegex(deploy_pi.DeploymentError, "审阅后发生变化"):
                deploy_pi.deploy(self.settings, files=self.names, apply=True, source_root=self.root,
                    expected_sha256={name: "b" * 64 for name in self.names})
        self.assertEqual(calls, ["check"])

    def test_reviewed_protected_config_mismatch_never_calls_mutation(self):
        with patch.object(deploy_pi, "_remote_call", return_value={"state": "checked",
            "files": {name: {"sha256": "a" * 64} for name in self.names},
            "protected_files": {"player/desktop/asr-selection.json": "c" * 64}}) as remote:
            with self.assertRaisesRegex(deploy_pi.DeploymentError, "配置在审阅后发生变化"):
                deploy_pi.deploy(self.settings, files=self.names, apply=True, source_root=self.root,
                    expected_sha256={name: "a" * 64 for name in self.names},
                    expected_protected_files={"player/desktop/asr-selection.json": "d" * 64})
        self.assertEqual(remote.call_count, 1)

    def test_reviewed_snapshot_must_cover_all_deployed_files(self):
        with patch.object(deploy_pi, "_remote_call", return_value={"state": "checked",
                "files": {name: {"sha256": "a" * 64} for name in self.names}}) as remote:
            with self.assertRaisesRegex(deploy_pi.DeploymentError, "没有包含本次全部"):
                deploy_pi.deploy(self.settings, files=self.names, apply=True, source_root=self.root,
                                 expected_sha256={self.names[0]: "a" * 64})
        self.assertEqual(remote.call_count, 1)


if __name__ == "__main__":
    unittest.main()
