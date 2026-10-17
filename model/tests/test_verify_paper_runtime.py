"""Runtime verification reports must stay within the requested model version."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from emostance_meme import paper_pipeline


spec = importlib.util.spec_from_file_location(
    "verify_paper_runtime", Path(__file__).resolve().parents[1] / "tools/verify_paper_runtime.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class RuntimeReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "model"
        self.previous_report = self.root / "runs/paper_v1/final_smoke.json"
        self.previous_report.parent.mkdir(parents=True)
        self.previous_report.write_text("existing V1 report\n", encoding="utf-8")
        self.config_path = self.root / "paper_config_v2.json"

    def test_invalid_config_returns_error_without_model_import_or_report_write(self):
        for config_text, error_type in (
            (None, "FileNotFoundError"),
            ("{broken JSON", "JSONDecodeError"),
            (json.dumps({"run_dir": "../outside", "data_dir": "data"}), "ValueError"),
        ):
            with self.subTest(error_type=error_type):
                if config_text is not None:
                    self.config_path.write_text(config_text, encoding="utf-8")
                error = io.StringIO()
                with patch.object(paper_pipeline, "MODEL_DIR", self.root), \
                     patch.object(verifier, "MODEL_ROOT", self.root), \
                     patch.object(paper_pipeline, "write_json") as write_json, \
                     patch.dict("sys.modules", {"torch": None, "emostance_meme.paper_engine": None}), \
                     redirect_stderr(error):
                    self.assertEqual(verifier.verify(str(self.config_path)), 1)
                self.assertIn(error_type, error.getvalue())
                write_json.assert_not_called()
                self.assertEqual(self.previous_report.read_text(encoding="utf-8"), "existing V1 report\n")
                self.assertFalse((self.root / "runs/paper_v2").exists())

    def test_failure_after_valid_config_writes_only_requested_version_report(self):
        self.config_path.write_text(json.dumps({
            "run_dir": "runs/paper_v2", "data_dir": "data", "do_sample": False,
        }), encoding="utf-8")
        engine = Mock(side_effect=AssertionError("No real model should be initialized"))
        cuda = SimpleNamespace(is_available=Mock(return_value=False))
        modules = {
            "torch": SimpleNamespace(cuda=cuda),
            "emostance_meme.paper_engine": SimpleNamespace(PaperMemeEngine=engine),
            "emostance_meme.http_server": SimpleNamespace(make_handler=Mock()),
            "emostance_meme.service": SimpleNamespace(ChatService=Mock()),
        }
        with patch.object(paper_pipeline, "MODEL_DIR", self.root), \
             patch.object(verifier, "MODEL_ROOT", self.root), \
             patch.dict("sys.modules", modules), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(verifier.verify(str(self.config_path)), 1)
        report = json.loads((self.root / "runs/paper_v2/final_smoke.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertIn("saved sampling configuration", report["error"])
        self.assertEqual(self.previous_report.read_text(encoding="utf-8"), "existing V1 report\n")
        engine.assert_not_called()


if __name__ == "__main__":
    unittest.main()
