"""Reject incomplete or mismatched trained runs before loading GPU weights."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

from model_information_qwen import model


class CompletedQwenRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.data = self.root / "data"
        self.base = self.root / "qwen"
        self.config = {"run_dir": str(self.run), "data_dir": str(self.data),
                       "base_model_path": str(self.base), "device": "cpu"}
        # Quality validation is exercised separately; these tests isolate other
        # required artifacts, index provenance and backbone checks.
        self.quality_patch_original = model._quality_release
        self.quality = patch.object(model, "_quality_release", return_value={}).start()
        self.addCleanup(patch.stopall)
        for name in model._REQUIRED:
            self.write(self.run / name, {} if name.endswith(".json") else b"local trained artifact")
        for name in ("manifest.json", "planner/manifest.json", "generator/manifest.json"):
            self.write(self.run / name, {"status": "complete"})
        self.write(self.data / "train.jsonl", b'{"split":"train"}\n')
        self.index = {"status": "complete", "split": "train",
            "train_file_sha256": model._sha256(self.data / "train.jsonl"),
            "planner_checkpoint_sha256": model._sha256(self.run / "planner/best.pt"),
            "space_sha256": model._sha256(self.run / "space.json")}
        self.write(self.run / "response_index_manifest.json", self.index)
        self.write(self.base / "config.json", {"model_type": "qwen3"})
        for name in ("tokenizer.json", "tokenizer_config.json"):
            self.write(self.base / name, {})
        self.write(self.base / "model.safetensors", b"local Qwen base weights")

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value).encode() if isinstance(value, dict) else value)

    def test_complete_run_and_local_qwen_pass_validation(self):
        model._ready(self.config, verify=True)
        self.quality.assert_called_once_with(self.config)
        model._backbone(self.config)

    def test_failed_chat_quality_blocks_loading_even_with_verify_false(self):
        self.quality.side_effect = ValueError("Qwen chat quality acceptance has not passed")
        with patch.object(model, "_configuration", return_value=(self.root / "config.json", self.config)), \
             patch.object(model, "_backbone", side_effect=AssertionError("must fail before backbone")):
            with self.assertRaisesRegex(ValueError, "quality acceptance"):
                model.load_engine(device="cpu", verify=False)

    def test_quality_guard_delegates_to_explicit_inference_release_validator(self):
        approved = {"dev": {"status": "rejected"}, "holdout": {"status": "rejected"}}
        pipeline = SimpleNamespace(require_inference_release=Mock(return_value=approved),
            require_quality_release=Mock(side_effect=AssertionError("legacy guard cannot authorize selected deployment")))
        # The fixture patches the guard to isolate artifact tests. Temporarily
        # exercise the actual function, using a pipeline with no GPU dependency.
        with patch.object(model, "_runtime", return_value="test_qwen"), \
             patch.object(model.importlib, "import_module", return_value=pipeline):
            actual_guard = self.quality_patch_original
            self.assertIs(actual_guard(self.config), approved)
        pipeline.require_inference_release.assert_called_once_with(self.config)
        pipeline.require_quality_release.assert_not_called()

    def test_missing_quality_reports_block_even_an_explicit_selected_deployment(self):
        for name in ("generator/quality_dev.json", "generator/quality_holdout.json"):
            with self.subTest(name=name):
                path = self.run / name
                previous = path.read_bytes()
                path.unlink()
                with self.assertRaisesRegex(FileNotFoundError, name):
                    model._ready(self.config, verify=False)
                path.write_bytes(previous)

    def test_inspection_preserves_rejected_quality_for_user_selected_run(self):
        self.write(self.run / "manifest.json", {"status": "complete", "quality_status": "rejected"})
        self.write(self.run / "generator/manifest.json", {"status": "complete", "quality_status": "rejected"})
        selection = {"status": "ready", "release_mode": "user_selected_with_known_quality_failures", "quality_status": "rejected",
                     "selected_checkpoint": {"global_step": 52379}}
        with patch.object(model, "_configuration", return_value=(self.root / "inference.json", self.config)), \
             patch.object(model, "_deployment_selection", return_value=selection):
            info = model.inspect_model_version()
        self.assertEqual(info["quality_status"], "rejected")
        self.assertEqual(info["release_status"], "ready")
        self.assertEqual(info["release_mode"], "user_selected_with_known_quality_failures")
        self.assertEqual(info["selected_step"], 52379)

    def test_inspection_reports_manifest_status_for_ordinary_quality_release(self):
        self.write(self.run / "manifest.json", {"status": "complete", "quality_status": "passed"})
        self.write(self.run / "generator/manifest.json", {"status": "complete", "quality_status": "passed", "best_step": 2000})
        with patch.object(model, "_configuration", return_value=(self.root / "paper_config.json", self.config)), \
             patch.object(model, "_deployment_selection", return_value=None):
            info = model.inspect_model_version()
        self.assertEqual(info["quality_status"], "passed")
        self.assertEqual(info["release_status"], "quality_passed")
        self.assertEqual(info["release_mode"], "strict_quality_acceptance")
        self.assertEqual(info["selected_step"], 2000)

    def test_default_config_fallback_warns_and_keeps_release_guard(self):
        default = self.root / "inference_config.json"
        fallback = self.root / "paper_config.json"
        self.write(fallback, {})
        pipeline = SimpleNamespace(load_config=Mock(return_value=self.config))
        self.quality.side_effect = ValueError("Explicit deployment selection or passed quality is required")
        with patch.object(model, "DEFAULT_CONFIG_PATH", default), \
             patch.object(model, "TRAINING_ROOT", self.root), \
             patch.object(model, "_runtime", return_value="test_qwen"), \
             patch.object(model.importlib, "import_module", return_value=pipeline), \
             patch.object(model, "_backbone", side_effect=AssertionError("must not load weights")):
            with self.assertWarnsRegex(RuntimeWarning, "checks remain required"):
                with self.assertRaisesRegex(ValueError, "deployment selection"):
                    model.load_engine(device="cpu", verify=False)
        pipeline.load_config.assert_called_once_with(fallback.resolve())
        self.quality.assert_called_once()

    def test_explicit_missing_config_does_not_fall_back(self):
        default = self.root / "inference_config.json"
        other = self.root / "missing_selected.json"
        self.write(self.root / "paper_config.json", {})
        pipeline = SimpleNamespace(load_config=Mock(side_effect=FileNotFoundError(str(other))))
        with patch.object(model, "DEFAULT_CONFIG_PATH", default), \
             patch.object(model, "TRAINING_ROOT", self.root), \
             patch.object(model, "_runtime", return_value="test_qwen"), \
             patch.object(model.importlib, "import_module", return_value=pipeline), \
             warnings.catch_warnings(record=True) as caught:
            with self.assertRaises(FileNotFoundError):
                model._configuration(config_path=other)
        self.assertEqual(caught, [])
        pipeline.load_config.assert_called_once_with(other.resolve())

    def test_changed_chat_settings_are_rejected_before_model_loading(self):
        config = {**self.config, "max_prompt_tokens":512, "max_new_tokens":128, "do_sample":True,
                  "temperature":.7, "top_p":.8, "no_repeat_ngram_size":4, "system_prompt":"角色指令"}
        settings = dict(max_prompt_tokens=512, max_reply_tokens=128, do_sample=True, temperature=.7,
                        top_p=.8, no_repeat_ngram_size=4, role_prompt="角色指令", asr_enabled=False)
        for changed in ({"max_prompt_tokens":4096}, {"role_prompt":"另一个角色"}, {"asr_enabled":True}):
            with self.subTest(changed=changed), \
                 patch.object(model, "_configuration", return_value=(self.root/"config.json", config)), \
                 patch("model_information_qwen.chat_engine.load_chat_settings", return_value={**settings, **changed}), \
                 patch.object(model, "load_engine", side_effect=AssertionError("must fail before loading weights")):
                with self.assertRaisesRegex(ValueError, "recorded.*evaluation|one reply generation"):
                    model.load_chat_engine(settings_path=self.root/"chat.json")

    def test_no_manifest_or_incomplete_manifest_never_loads_model(self):
        for name in ("manifest.json", "planner/manifest.json", "generator/manifest.json"):
            with self.subTest(name=name):
                self.write(self.run / name, {"status": "running"})
                with patch.object(model, "_configuration", return_value=(self.root / "config.json", self.config)), \
                     patch.object(model, "_backbone", side_effect=AssertionError("must fail before backbone")):
                    with self.assertRaisesRegex(ValueError, "not complete"):
                        model.load_engine(device="cpu", verify=False)
                self.write(self.run / name, {"status": "complete"})

    def test_missing_prefix_cannot_be_served_as_an_untrained_generator(self):
        (self.run / "generator/best.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "generator/best.pt"):
            model._ready(self.config, verify=False)

    def test_dataset_checkpoint_and_space_must_match_response_index(self):
        for key in ("train_file_sha256", "planner_checkpoint_sha256", "space_sha256"):
            with self.subTest(key=key):
                altered = {**self.index, key: "0" * 64}
                self.write(self.run / "response_index_manifest.json", altered)
                with self.assertRaisesRegex(ValueError, key):
                    model._ready(self.config, verify=False)
        self.write(self.run / "response_index_manifest.json", self.index)

    def test_incomplete_index_and_nontraining_bank_are_rejected(self):
        for override in ({"status": "running"}, {"split": "dev"}):
            with self.subTest(override=override):
                self.write(self.run / "response_index_manifest.json", {**self.index, **override})
                with self.assertRaises(ValueError):
                    model._ready(self.config, verify=False)

    def test_completed_run_hash_catches_changed_prefix_with_same_length(self):
        path = self.run / "generator/best.pt"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write(self.run / "manifest.json", {"status": "complete",
            "artifact_sha256": {"generator/best.pt": checksum}})
        path.write_bytes(b"X" + path.read_bytes()[1:])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            model._ready(self.config, verify=True)

    def test_old_backbone_and_lfs_pointer_are_rejected(self):
        self.write(self.base / "config.json", {"model_type": "mistral"})
        with self.assertRaisesRegex(ValueError, "Qwen3 backbone"):
            model._backbone(self.config)
        self.write(self.base / "config.json", {"model_type": "qwen3"})
        self.write(self.base / "model.safetensors", b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            model._backbone(self.config)


if __name__ == "__main__":
    unittest.main()
