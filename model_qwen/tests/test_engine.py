"""Behavior checks for Qwen single generation and honest stage completion."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch  # Load once before scoped sys.modules mocks are restored.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from emostance_qwen.paper_engine import PaperMemeEngine
from emostance_qwen import paper_pipeline as pipeline


class SingleGenerationTest(unittest.TestCase):
    def test_one_generation_and_one_realized_reply_query_without_distribution_scoring(self):
        calls, generated, retrieved = [], [], []

        class Planner:
            def predict(self, history, next_role):
                calls.append((copy.deepcopy(history), next_role))
                if next_role == "B":
                    # Deliberately omit distributions needed by the old reranker.
                    return {"target_vector":np.asarray([1., 0.])}
                return {"source_realized_vector":np.asarray([0., 1.]),
                        "source_vector":np.asarray([99., 99.])}

        class Generator:
            def generate(self, history, next_role, vector, max_new_tokens):
                generated.append((copy.deepcopy(history), next_role, max_new_tokens))
                np.testing.assert_array_equal(vector, [1., 0.])
                return "模型新生成回复"

        class Index:
            def nearest(self, vector):
                retrieved.append(np.asarray(vector))
                np.testing.assert_array_equal(vector, [0., 1.])
                return {"reply":"训练库中的旧回复", "meme_id":"meme-0001"}

        engine = PaperMemeEngine.__new__(PaperMemeEngine)
        # Even direct callers carrying an obsolete field cannot enable a pool.
        engine.config = {"do_sample":True, "num_candidates":4, "max_new_tokens":73}
        engine.planner, engine.generator, engine.index = Planner(), Generator(), Index()
        engine._lock = threading.Lock()
        messages = [{"role":"user", "content":"上一轮"},
                    {"role":"assistant", "content":"上一轮回复"},
                    {"role":"user", "content":"当前消息"}]
        original = copy.deepcopy(messages)
        result = engine.reply(messages)
        self.assertEqual(result, {"reply":"模型新生成回复", "meme_id":"meme-0001"})
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(retrieved), 1)
        self.assertEqual([role for _, role in calls], ["B", "A"])
        self.assertEqual(calls[1][0], calls[0][0]+[{"role":"B", "text":"模型新生成回复"}])
        self.assertEqual(generated[0][2], 73)
        self.assertEqual(messages, original)


class PipelineTest(unittest.TestCase):
    def test_config_resolves_local_paths_and_reads_system_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"prompt.txt").write_text("中文系统指令\n", encoding="utf-8")
            settings = {"run_dir":"runs/full", "data_dir":"../data_spilt/V2.0.0",
                        "base_model_path":"backbones/Qwen3-8B", "planner_model_path":"backbones/deberta",
                        "system_prompt_file":"prompt.txt"}
            (root/"paper_config.json").write_text(json.dumps(settings), encoding="utf-8")
            with patch.object(pipeline, "MODEL_DIR", root):
                config = pipeline.load_config()
            for key in ("run_dir", "data_dir", "base_model_path", "planner_model_path", "system_prompt_file"):
                self.assertTrue(Path(config[key]).is_absolute())
            self.assertEqual(config["system_prompt"], "中文系统指令")
            self.assertIs(config["enable_thinking"], False)

    def test_config_rejects_external_outputs_and_candidate_pool(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {"run_dir":"../outside", "data_dir":"data", "base_model_path":"base",
                        "planner_model_path":"planner"}
            path = root/"paper_config.json"
            with patch.object(pipeline, "MODEL_DIR", root):
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "model_qwen"):
                    pipeline.load_config(path)
                settings.update(run_dir="runs/full", num_candidates=4)
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Candidate generation"):
                    pipeline.load_config(path)
                settings.update(num_candidates=1, enable_thinking=True)
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "enable_thinking"):
                    pipeline.load_config(path)

    def test_partial_generator_training_never_reports_full_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/"runs/full"
            (run/"planner").mkdir(parents=True)
            (run/"planner/manifest.json").write_text('{"status":"complete"}')
            planner = types.ModuleType("emostance_qwen.paper_planner")
            planner.train_planner = lambda *args: {"status":"complete"}
            generator = types.ModuleType("emostance_qwen.paper_generator")
            generator.train_generator = lambda *args, **kwargs: {"status":"incomplete", "global_step":1}
            with patch.object(pipeline, "MODEL_DIR", root), \
                 patch.object(pipeline, "prepare_space", return_value=object()), \
                 patch.object(pipeline, "read_rows", return_value=[{}]), \
                 patch.dict(sys.modules, {planner.__name__:planner, generator.__name__:generator}):
                summary = pipeline.train({"run_dir":str(run), "data_dir":str(root/"data")})
            self.assertEqual(summary["status"], "incomplete")
            self.assertEqual(json.loads((run/"training_status.json").read_text())["status"], "incomplete")
            self.assertFalse((run/"manifest.json").exists())

    def test_planner_interruption_is_recorded_and_generator_does_not_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/"runs/full"
            planner = types.ModuleType("emostance_qwen.paper_planner")
            def interrupt(*args):
                raise KeyboardInterrupt()
            planner.train_planner = interrupt
            with patch.object(pipeline, "MODEL_DIR", root), \
                 patch.object(pipeline, "prepare_space", return_value=object()), \
                 patch.object(pipeline, "read_rows", return_value=[{}]), \
                 patch.dict(sys.modules, {planner.__name__:planner}):
                with self.assertRaises(KeyboardInterrupt):
                    pipeline.train({"run_dir":str(run), "data_dir":str(root/"data")})
            self.assertEqual(json.loads((run/"training_status.json").read_text())["status"], "interrupted")
            self.assertFalse((run/"generator_result.json").exists())

    def test_finalize_binds_trained_artifacts_and_single_generation_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root/"runs/full"
            for stage in ("planner", "generator", "graph"):
                (run/stage).mkdir(parents=True)
                data = {"status":"complete", "num_clusters":2}
                (run/stage/"manifest.json").write_text(json.dumps(data))
            (run/"response_index_manifest.json").write_text('{"status":"complete","split":"train"}')
            (run/"evaluation.json").write_text('{"dev":{"examples":1},"test":{"examples":1}}')
            for name in ("space.json", "planner/best.pt", "generator/best.pt", "generator/quality_dev.json",
                         "generator/quality_holdout.json", "response_vectors.npz", "response_anchors.jsonl"):
                (run/name).write_bytes(name.encode())
            quality = {phase:{"comparison":{"passed":True}} for phase in ("dev", "holdout")}
            with patch.object(pipeline, "MODEL_DIR", root), \
                 patch.object(pipeline, "require_quality_release", return_value=quality):
                manifest = pipeline.finalize_run({"run_dir":str(run)})
            self.assertEqual(manifest["inference"]["generations_per_reply"], 1)
            self.assertEqual(manifest["inference"]["reranking"], "disabled")
            self.assertIs(manifest["inference"]["enable_thinking"], False)
            self.assertEqual(manifest["artifact_sha256"]["generator/best.pt"],
                             hashlib.sha256(b"generator/best.pt").hexdigest())
            (run/"generator/manifest.json").write_text('{"status":"incomplete"}')
            with patch.object(pipeline, "MODEL_DIR", root), self.assertRaisesRegex(ValueError, "has not completed"):
                pipeline.finalize_run({"run_dir":str(run)})


if __name__ == "__main__":
    unittest.main()
