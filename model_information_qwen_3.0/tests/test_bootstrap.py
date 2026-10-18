"""The dotted directory resolves to its own package in fresh CPU processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "bootstrap.py"


class BootstrapTests(unittest.TestCase):
    def execute(self, *arguments, cwd=None):
        return subprocess.run([sys.executable, str(BOOTSTRAP), *arguments],
                              cwd=cwd, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                              capture_output=True, text=True, timeout=10)

    def test_alias_resolves_new_package_without_importing_backbone_or_old_package(self):
        source = """import json, sys
from pathlib import Path
from model_information_qwen_v3 import launcher, dialogue_log, server
print(json.dumps({'module':str(Path(launcher.__file__).resolve()),
 'log':str(dialogue_log.DEFAULT_LOG_PATH), 'turns':server.DEFAULT_MAX_TURNS,
 'old_loaded':'model_information_qwen' in sys.modules,
 'torch_loaded':'torch' in sys.modules}))
"""
        result = self.execute("-c", source, cwd="/tmp")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["module"], str(ROOT / "launcher.py"))
        self.assertEqual(payload["log"], str(ROOT / "runs/conversation.log"))
        self.assertEqual(payload["turns"], 4)
        self.assertFalse(payload["old_loaded"])
        self.assertFalse(payload["torch_loaded"])

    def test_entrypoint_help_works_from_another_working_directory(self):
        for entry in ("cli", "launcher", "server", "deploy_pi", "tools.verify_integration",
                      "tools.verify_dialogue_rollout", "tools.verify_pi_handshake"):
            with self.subTest(entry=entry):
                result = self.execute(entry, "--help", cwd="/tmp")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_unittest_entry_can_load_an_aliased_test_module(self):
        result = self.execute("unittest", "model_information_qwen_v3.tests.test_console.ConsoleTests.test_rejects_invalid_tail_values")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_test_file_entry_runs_after_registering_alias(self):
        result = self.execute("tests/test_console.py", "ConsoleTests.test_rejects_invalid_tail_values")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Ran 1 test", result.stderr)

    def test_maintenance_tools_keep_direct_script_help(self):
        for script in ("deploy_pi.py", "tools/verify_integration.py", "tools/verify_dialogue_rollout.py",
                       "tools/verify_pi_handshake.py"):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd="/tmp",
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_script_entry_is_confined_to_this_version(self):
        result = self.execute("../model_information_qwen/launcher.py", "--help")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside this version directory", result.stderr)


if __name__ == "__main__":
    unittest.main()
