"""CPU-only evidence checks for diagnostics and version finalization."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/finalize_model_version.py"
SPEC = importlib.util.spec_from_file_location("finalize_model_version", SCRIPT)
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


class TinyTokenizer:
    unk_token_id = 3
    unk_token = "[UNK]"
    is_fast = True
    vocab_size = 128

    def __init__(self):
        self.calls = []

    def __call__(self, texts, *, truncation, max_length=None, **kwargs):
        self.calls.append((len(texts), truncation))
        result = {"input_ids": [], "special_tokens_mask": []}
        for text in texts:
            content = [int(value) for value in text.split()]
            if truncation:
                content = content[-(max_length - 2):]
            result["input_ids"].append([101] + content + [102])
            result["special_tokens_mask"].append([1] + [0] * len(content) + [1])
        return result


class FinalizeModelVersionTests(unittest.TestCase):
    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_log_requires_final_complete_successful_unittest_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tests.log"
            path.write_text("Started today\n\nRan 7 tests in 0.125s\n\nOK\n")
            evidence = finalizer.parse_unit_test_log(path)
            self.assertEqual(evidence["unit_tests"], 7)
            self.assertEqual(evidence["unit_test_seconds"], .125)
            self.assertEqual(evidence["unit_test_log_sha256"], finalizer.digest(path))
            for text in ("OK\n", "Ran 0 tests in 0.1s\n\nOK\n",
                         "Ran 7 tests in 0.1s\n\nFAILED (failures=1)\n",
                         "Ran 7 tests in 0.1s\n\nOK\nStarted another run\n",
                         "Ran 7 tests in 0.1s\n\nOK\nRan 2 tests in 0.1s\n"):
                path.write_text(text)
                with self.subTest(log=text), self.assertRaises(ValueError):
                    finalizer.parse_unit_test_log(path)

    def test_train_only_batched_diagnostics_measure_actual_truncation_and_unknowns(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "data/train.jsonl"
            train.parent.mkdir()
            train.write_text('\n'.join(json.dumps(row) for row in (
                {"split": "train", "text": "1 3 2 3 4"},
                {"source_split": "train", "text": "1 2"})) + '\n')
            self.write_json(root / "run/planner/tokenizer/tokenizer.json", {"tokenizer": "test"})
            tokenizer = TinyTokenizer()
            result = finalizer.tokenizer_diagnostics(
                {"run_dir": str(root / "run"), "data_dir": str(root / "data"), "planner_max_length": 5},
                tokenizer=tokenizer, serializer=lambda row: row["text"], batch_size=1)
            self.assertEqual(tokenizer.calls, [(1, True), (1, False)] * 2)
            self.assertEqual(tokenizer.truncation_side, "left")
            self.assertEqual(result["source"]["sample_count"], 2)
            self.assertEqual(result["source"]["sha256"], finalizer.digest(train))
            window = result["training_window"]
            self.assertEqual((window["total_tokens"], window["unk_tokens"]), (9, 1))
            self.assertEqual(window["inserted_special_tokens"], 4)
            self.assertEqual(window["non_special_tokens"], 5)
            self.assertAlmostEqual(window["unk_ratio_excluding_inserted_special_tokens"], .2)
            self.assertEqual(window["samples_containing_unk"], 1)
            self.assertEqual((result["before_truncation"]["total_tokens"],
                              result["before_truncation"]["unk_tokens"]), (11, 2))
            self.assertEqual(result["truncation"]["tokens_removed"], 2)
            self.assertEqual(result["truncation"]["samples_truncated_ratio"], .5)
            self.assertFalse((train.parent / "dev.jsonl").exists())
            self.assertFalse((train.parent / "test.jsonl").exists())

    def test_diagnostics_reject_nontraining_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_json(root / "run/planner/tokenizer/tokenizer.json", {})
            (root / "data").mkdir()
            (root / "data/train.jsonl").write_text('{"split":"dev","text":"1 2"}\n')
            with self.assertRaisesRegex(ValueError, "train-only"):
                finalizer.tokenizer_diagnostics(
                    {"run_dir": str(root / "run"), "data_dir": str(root / "data")},
                    tokenizer=TinyTokenizer(), serializer=lambda row: row["text"])

    def test_existing_archive_is_rejected_before_loading_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "V1.0.0"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("unchanged")
            with self.assertRaises(FileExistsError):
                finalizer.finalize_version("missing-config.json", existing, "missing-log.txt")
            self.assertEqual(marker.read_text(), "unchanged")

    def test_archive_inventory_hashes_and_unexpected_files_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "V2.0.0"
            archive.mkdir()
            payload = archive / "weights.bin"
            payload.write_bytes(b"synthetic weights")
            self.write_json(archive / "version_manifest.json", {
                "status": "complete", "version": archive.name, "source_run": str(root / "run"),
                "prefix_tensors_verified_equal": True,
                "files": {payload.name: {"sha256": finalizer.digest(payload), "bytes": payload.stat().st_size}},
                "total_bytes": payload.stat().st_size})
            result = finalizer.verify_archive(archive, root / "run")
            self.assertTrue(result["archive_integrity_verified"])
            payload.write_bytes(b"tampered weights!")
            with self.assertRaisesRegex(ValueError, "integrity"):
                finalizer.verify_archive(archive, root / "run")
            payload.write_bytes(b"synthetic weights")
            (archive / "untracked.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "inventory"):
                finalizer.verify_archive(archive, root / "run")

    def test_baseline_comparison_reports_improvement_regression_and_zero_baseline(self):
        compare = finalizer.build_report.__globals__["baseline_comparison"]
        self.assertIn("降低约 **25.00%**", compare(1.5, 2))
        self.assertIn("上升约 **25.00%**，该指标表现变差", compare(2.5, 2))
        self.assertIn("持平", compare(2, 2))
        self.assertIn("基线为零", compare(1, 0))
        self.assertNotIn("提升有限", compare(1.5, 2))
        with self.assertRaises(ValueError):
            compare(float("nan"), 2)


if __name__ == "__main__":
    unittest.main()
