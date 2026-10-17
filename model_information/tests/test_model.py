"""Archive-loader tests using tiny fake weights/runtime; no GPU or downloads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from model_information import model


class ArchivedModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "v1.0.0"
        self.root.mkdir()
        self.base = Path(self.temporary.name) / "local_mistral"
        self.base.mkdir()
        (self.base / "config.json").write_text("{}", encoding="utf-8")
        (self.base / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.base / "model.safetensors").write_bytes(b"tiny fake local weights for loader-only tests")
        self.config = {"run_dir": ".", "data_dir": "data", "base_model_path": str(self.base), "device": "cuda", "dtype": "bfloat16"}
        self.write("inference_config.json", self.config)
        for filename in ("manifest.json", "planner/manifest.json", "generator/manifest.json"):
            self.write(filename, {"status": "complete"})
        for filename in ("planner/config.json", "planner/emotion_space.json", "planner/encoder_config/config.json",
                         "planner/tokenizer/tokenizer.json", "planner/tokenizer/tokenizer_config.json", "space.json", "response_index_manifest.json"):
            self.write(filename, {})
        for filename in ("planner/best.pt", "generator/best.pt", "response_vectors.npz"):
            self.write_bytes(filename, b"tiny fake archived weights")
        self.write_bytes("response_anchors.jsonl", b'{"meme_id":"meme-0002"}\n')
        self.write_bytes("data/train.jsonl", b'{"split":"train"}\n')
        self.write_bytes("runtime_source/emostance_meme/__init__.py", b'VERSION = "archived"\n')
        self.write_bytes("runtime_source/emostance_meme/helper.py", b'IDENTITY = "private archived runtime"\n')
        self.write_bytes("runtime_source/emostance_meme/paper_engine.py", b'''from .helper import IDENTITY
import os
class PaperMemeEngine:
    def __init__(self, config):
        self.config = dict(config)
        self.identity = IDENTITY
        self.package = __package__
        self.offline = (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
    def reply(self, messages):
        return {"reply": "fake runtime used only by unit tests", "meme_id": "meme-0002"}
''')
        self.refresh_manifest()
        previous_bytecode = sys.dont_write_bytecode
        self.addCleanup(setattr, sys, "dont_write_bytecode", previous_bytecode)
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.original_modules = set(sys.modules)
        self.addCleanup(self.clean_private_modules)

    def clean_private_modules(self):
        for name in set(sys.modules) - self.original_modules:
            if name.startswith("_archived_emostance_"):
                sys.modules.pop(name, None)

    def write(self, relative, payload):
        self.write_bytes(relative, (json.dumps(payload) + "\n").encode())

    def write_bytes(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def refresh_manifest(self, **overrides):
        entries = {}
        for path in self.root.rglob("*"):
            if path.is_file() and path.name != "version_manifest.json":
                content = path.read_bytes()
                entries[path.relative_to(self.root).as_posix()] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        self.manifest = {"version": "v1.0.0", "status": "complete", "inference_only": True, "files": entries, **overrides}
        self.write("version_manifest.json", self.manifest)

    def test_relative_paths_cpu_default_and_offline_are_resolved_without_mutation(self):
        before = (self.root / "inference_config.json").read_bytes()
        engine = model.load_engine(self.root, device="cpu")
        self.assertEqual(engine.config["run_dir"], str(self.root.resolve()))
        self.assertEqual(engine.config["data_dir"], str((self.root / "data").resolve()))
        self.assertEqual(engine.config["dtype"], "float32")
        self.assertEqual(engine.offline, ("1", "1"))
        self.assertEqual(engine.identity, "private archived runtime")
        self.assertEqual((self.root / "inference_config.json").read_bytes(), before)
        self.assertFalse(list(self.root.rglob("__pycache__")))

    def test_existing_public_package_cannot_replace_archived_private_runtime(self):
        unrelated = types.ModuleType("emostance_meme")
        unrelated.identity = "unrelated workspace package"
        with patch.dict(sys.modules, {"emostance_meme": unrelated}):
            engine = model.load_engine(self.root, device="cpu")
            self.assertIs(sys.modules["emostance_meme"], unrelated)
            self.assertTrue(engine.package.startswith("_archived_emostance_"))
            self.assertEqual(engine.identity, "private archived runtime")

    def test_explicit_cpu_dtype_is_preserved(self):
        engine = model.load_engine(self.root, device="cpu", dtype="bfloat16")
        self.assertEqual(engine.config["dtype"], "bfloat16")

    def test_auto_without_cuda_defaults_to_float32(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            engine = model.load_engine(self.root, device="auto")
        self.assertEqual(engine.config["device"], "cpu")
        self.assertEqual(engine.config["dtype"], "float32")

    def test_same_size_sha_mismatch_is_rejected_before_runtime_import(self):
        path = self.root / "generator/best.pt"
        content = path.read_bytes()
        path.write_bytes(b"X" + content[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch: generator/best.pt"):
            model.load_engine(self.root, device="cpu")

    def test_verify_false_skips_sha_only(self):
        path = self.root / "generator/best.pt"
        path.write_bytes(b"changed but present fake weights")
        self.assertEqual(model.load_engine(self.root, device="cpu", verify=False).identity, "private archived runtime")
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "generator/best.pt"):
            model.load_engine(self.root, device="cpu", verify=False)

    def test_incomplete_version_rejected_even_without_hash_verification(self):
        self.refresh_manifest(status="incomplete")
        with self.assertRaisesRegex(ValueError, "version is not complete"):
            model.load_engine(self.root, device="cpu", verify=False)

    def test_incomplete_stage_is_rejected(self):
        self.write("generator/manifest.json", {"status": "incomplete"})
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, "stage is not complete"):
            model.load_engine(self.root, device="cpu")

    def test_missing_backbone_weights_do_not_fall_back_or_download(self):
        (self.base / "model.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Mistral weights are missing"):
            model.load_engine(self.root, device="cpu")

    def test_missing_sharded_backbone_weight_is_explicit(self):
        (self.base / "model.safetensors").unlink()
        (self.base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "missing-shard.safetensors"}}))
        with self.assertRaisesRegex(FileNotFoundError, "missing-shard"):
            model.load_engine(self.root, device="cpu")

    def test_git_lfs_pointer_is_not_treated_as_model_weights(self):
        (self.base / "model.safetensors").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:abcd\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            model.load_engine(self.root, device="cpu")

    def test_invalid_device_dtype_and_verify_are_explicit(self):
        for keyword in ({"device": "network"}, {"dtype": "int8"}, {"verify": "yes"}):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                model.load_engine(self.root, **keyword)

    def test_manifest_path_escape_is_rejected(self):
        self.manifest["files"]["../outside"] = {"sha256": "0" * 64}
        self.write("version_manifest.json", self.manifest)
        with self.assertRaisesRegex(ValueError, "escapes the model directory"):
            model.load_engine(self.root, device="cpu")

    def test_untracked_runtime_python_cannot_bypass_hash_verification(self):
        self.write_bytes("runtime_source/emostance_meme/untracked.py", b"SIDE_EFFECT = True\n")
        with self.assertRaisesRegex(ValueError, "Runtime source is not covered"):
            model.load_engine(self.root, device="cpu")

    def test_runtime_failure_does_not_return_a_placeholder_engine(self):
        self.write_bytes("runtime_source/emostance_meme/paper_engine.py", b'class PaperMemeEngine:\n    def __init__(self, config):\n        raise RuntimeError("trained weights cannot be loaded")\n')
        self.refresh_manifest()
        with self.assertRaisesRegex(RuntimeError, "trained weights cannot be loaded"):
            model.load_engine(self.root, device="cpu")

    def test_inspection_is_lightweight_and_resolves_paths(self):
        with patch.object(model, "_private_engine_class", side_effect=AssertionError("must not import runtime")):
            result = model.inspect_model_version(self.root)
        self.assertEqual(result["version"], "v1.0.0")
        self.assertEqual(result["run_dir"], str(self.root.resolve()))
        self.assertFalse(result["file_hashes_verified"])

    def test_default_model_directory_can_be_selected(self):
        with patch.object(model, "DEFAULT_MODEL_DIR", self.root):
            self.assertEqual(model.load_engine(device="cpu").identity, "private archived runtime")

    def test_chat_role_prompt_rejected_before_loading_archived_weights(self):
        with patch("model_information.chat_engine.load_chat_settings", return_value={"role_prompt": "你是meme。"}), \
                patch.object(model, "load_engine") as load:
            with self.assertRaisesRegex(ValueError, "role_prompt_enabled=false"):
                model.load_chat_engine(self.root, device="cpu")
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
