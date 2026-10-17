"""Restore tiny local packages only; no model/runtime/network dependencies."""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/restore_git_assets.py"
_spec = importlib.util.spec_from_file_location("restore_git_assets", SCRIPT)
restore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(restore)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RestoreGitAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.root.mkdir()
        self.payload = '{"中文":"数据"}\n'.encode("utf-8")
        self.pack = gzip.compress(self.payload, mtime=0)
        self.manifest = {
            "packed_data": [{"source_path": "data_spilt/V2.0.0/train.jsonl",
                             "packed_path": "model/git_data/train.jsonl.gz",
                             "bytes": len(self.payload), "sha256": sha(self.payload),
                             "packed_sha256": sha(self.pack)}],
            "duplicate_data": [{"source_path": "data_spilt/V2.0.0/train.jsonl",
                                "target_path": "model_canshu/V2.0.0/data/train.jsonl",
                                "bytes": len(self.payload), "sha256": sha(self.payload)}],
            "excluded_large_weights": [{"path": "model_canshu/V2.0.0/planner/best.pt",
                                        "bytes": 7, "sha256": sha(b"weights")}],
        }
        self.write("model/git_data/train.jsonl.gz", self.pack)
        self.save_manifest()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def save_manifest(self):
        self.write("model/git_upload_manifest.json", json.dumps(self.manifest).encode("utf-8"))

    def assert_no_temporary_files(self):
        self.assertEqual(list(self.root.rglob("*.restore-*")), [])

    def test_restore_packages_and_duplicates_then_skip_matching_existing_files(self):
        report = restore.restore_assets(self.root)
        self.assertEqual((report["restored_files"], report["unpacked_files"], report["copied_files"]), (2, 1, 1))
        for relative in (self.manifest["packed_data"][0]["source_path"],
                         self.manifest["duplicate_data"][0]["target_path"]):
            self.assertEqual((self.root / relative).read_bytes(), self.payload)
        self.assertEqual(report["missing_weights"], self.manifest["excluded_large_weights"])
        self.assertFalse(report["weights_downloaded"])
        self.assertFalse((self.root / self.manifest["excluded_large_weights"][0]["path"]).exists())
        again = restore.restore_assets(self.root)
        self.assertEqual((again["restored_files"], again["skipped_files"]), (0, 2))
        self.assert_no_temporary_files()

    def test_conflicting_existing_target_is_not_overwritten(self):
        for section, field in (("packed_data", "source_path"), ("duplicate_data", "target_path")):
            with self.subTest(section=section):
                target = self.write(self.manifest[section][0][field], b"keep my data")
                with self.assertRaisesRegex(restore.AssetRestoreError, "Refusing to overwrite"):
                    restore.restore_assets(self.root)
                self.assertEqual(target.read_bytes(), b"keep my data")
                target.unlink()
        self.assert_no_temporary_files()

    def test_packed_hash_is_checked_before_any_target_is_created(self):
        self.write(self.manifest["packed_data"][0]["packed_path"], self.pack + b"corruption")
        with self.assertRaisesRegex(restore.AssetRestoreError, "Packed SHA-256 mismatch"):
            restore.restore_assets(self.root)
        self.assertFalse((self.root / self.manifest["packed_data"][0]["source_path"]).exists())
        self.assert_no_temporary_files()

    def test_unpacked_size_and_sha_mismatch_remove_temporary_outputs(self):
        original = copy.deepcopy(self.manifest)
        for field, value in (("sha256", "0" * 64), ("bytes", len(self.payload) - 1),
                             ("bytes", len(self.payload) + 1)):
            with self.subTest(field=field, value=value):
                self.manifest = copy.deepcopy(original)
                self.manifest["packed_data"][0][field] = value
                self.save_manifest()
                with self.assertRaisesRegex(restore.AssetRestoreError, "byte count|bytes or SHA-256"):
                    restore.restore_assets(self.root)
                self.assertFalse((self.root / self.manifest["packed_data"][0]["source_path"]).exists())
                self.assert_no_temporary_files()

    def test_invalid_gzip_with_valid_package_hash_is_rejected_atomically(self):
        # The latter has a valid gzip header but an invalid DEFLATE block.
        for corrupt in (b"not gzip", b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\x07" + b"\x00" * 8):
            with self.subTest(corrupt=corrupt):
                self.write(self.manifest["packed_data"][0]["packed_path"], corrupt)
                self.manifest["packed_data"][0]["packed_sha256"] = sha(corrupt)
                self.save_manifest()
                with self.assertRaisesRegex(restore.AssetRestoreError, "Cannot decompress"):
                    restore.restore_assets(self.root)
                self.assertFalse((self.root / self.manifest["packed_data"][0]["source_path"]).exists())
                self.assert_no_temporary_files()

    def test_duplicate_source_must_match_expected_size_and_sha(self):
        self.manifest["packed_data"] = []
        self.write(self.manifest["duplicate_data"][0]["source_path"], b"incorrect source")
        self.save_manifest()
        with self.assertRaises(restore.AssetRestoreError):
            restore.restore_assets(self.root)
        self.assertFalse((self.root / self.manifest["duplicate_data"][0]["target_path"]).exists())
        self.assert_no_temporary_files()

    def test_every_manifest_path_rejects_absolute_or_parent_traversal_before_restoring(self):
        original = copy.deepcopy(self.manifest)
        fields = (("packed_data", "source_path"), ("packed_data", "packed_path"),
                  ("duplicate_data", "source_path"), ("duplicate_data", "target_path"),
                  ("excluded_large_weights", "path"))
        for section, field in fields:
            for invalid in ("../outside", "data/../outside", "/absolute", "C:\\outside", "..\\outside"):
                with self.subTest(section=section, field=field, invalid=invalid):
                    self.manifest = copy.deepcopy(original)
                    self.manifest[section][0][field] = invalid
                    self.save_manifest()
                    with self.assertRaisesRegex(restore.AssetRestoreError, "relative|escapes"):
                        restore.restore_assets(self.root)
                    self.assertFalse((self.root / "data_spilt").exists())

    def test_symlinks_cannot_escape_repository(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (self.root / "escape").symlink_to(outside, target_is_directory=True)
        self.manifest["duplicate_data"][0]["target_path"] = "escape/train.jsonl"
        self.save_manifest()
        with self.assertRaisesRegex(restore.AssetRestoreError, "escapes repository"):
            restore.restore_assets(self.root)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.root / "data_spilt").exists())

    def test_existing_weights_are_verified_without_download_or_overwrite(self):
        weight = self.write(self.manifest["excluded_large_weights"][0]["path"], b"weights")
        self.assertEqual(restore.restore_assets(self.root)["missing_weights"], [])
        weight.write_bytes(b"wrong")
        with self.assertRaisesRegex(restore.AssetRestoreError, "Refusing to overwrite"):
            restore.restore_assets(self.root)
        self.assertEqual(weight.read_bytes(), b"wrong")

    def test_missing_pack_fails_clearly_and_does_not_create_target(self):
        (self.root / self.manifest["packed_data"][0]["packed_path"]).unlink()
        with self.assertRaisesRegex(restore.AssetRestoreError, "Missing packaged data"):
            restore.restore_assets(self.root)
        self.assertFalse((self.root / "data_spilt").exists())

    def test_invalid_metadata_is_rejected_before_any_write(self):
        self.manifest["excluded_large_weights"][0]["bytes"] = True
        self.save_manifest()
        with self.assertRaisesRegex(restore.AssetRestoreError, "Invalid byte count"):
            restore.restore_assets(self.root)
        self.assertFalse((self.root / "data_spilt").exists())

    def test_cli_works_directly_outside_repository_and_reports_missing_weights(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--repo-root", str(self.root)],
                                cwd=self.temporary.name, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["restored_files"], 2)
        self.assertEqual(report["missing_weights"], self.manifest["excluded_large_weights"])


if __name__ == "__main__":
    unittest.main()
