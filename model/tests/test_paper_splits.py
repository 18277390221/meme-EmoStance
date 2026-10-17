"""Real invariants for dialogue grouping and adjacent-turn preparation."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/build_paper_splits.py"
SPEC = importlib.util.spec_from_file_location("build_paper_splits", SCRIPT)
paper_splits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paper_splits)


class PaperSplitTests(unittest.TestCase):
    def dialogue(self, dialogue_id, texts=("第一句", "第二句", "第三句", "第四句")):
        annotations = [{"annotator": f"paper{i}", "meme_id": "m1" if i < 3 else "m2", "confidence": i, "reason": "测试理由"} for i in range(1, 5)]
        return {"dialogue_id": dialogue_id, "original_dialogue_id": "original-" + dialogue_id, "source_split": "test",
                "messages": [{"turn_id": i, "role": "A" if i % 2 == 0 else "B", "text": text,
                              "annotations": [dict(annotation) for annotation in annotations]} for i, text in enumerate(texts)]}

    def test_confidence_weighted_votes_and_reliability(self):
        distribution, weight = paper_splits.soft_labels(self.dialogue("d")["messages"][0]["annotations"])
        self.assertEqual(distribution, {"m1": .3, "m2": .7})
        self.assertEqual(weight, .5)

    def test_slice_both_roles_and_keep_full_history_without_target(self):
        row = self.dialogue("d")
        dialogue, samples, turns = paper_splits.make_records(row, "train", "g")
        self.assertEqual(len(samples), 3)
        self.assertEqual(len(turns), 4)
        self.assertEqual([(sample["source_role"], sample["next_role"]) for sample in samples], [("A", "B"), ("B", "A"), ("A", "B")])
        self.assertEqual([item["text"] for item in samples[2]["history"]], ["第一句", "第二句", "第三句"])
        self.assertEqual(samples[2]["response"], "第四句")
        self.assertEqual(samples[1]["source_text"], samples[1]["history"][-1]["text"])
        self.assertEqual(samples[0]["source_weight"], .5)
        self.assertEqual(samples[0]["target_meme_distribution"], {"m1": .3, "m2": .7})
        self.assertEqual(dialogue["source_split"], "train")
        self.assertEqual(dialogue["original_source_split"], "test")
        self.assertEqual(row["source_split"], "test")
        self.assertEqual(len(turns[0]["annotations"]), 4)

    def test_normalized_prefix_duplicate_transitive_group_preserves_all(self):
        rows = [self.dialogue("short", ("你 好", "谢 谢")), self.dialogue("duplicate", ("你好", "谢谢")),
                self.dialogue("long", ("你好", "谢谢", "后续", "回复")), self.dialogue("other", ("你好", "不同回复"))]
        groups, relations = paper_splits.related_groups(rows)
        self.assertIn(["duplicate", "long", "short"], groups)
        self.assertIn(["other"], groups)
        self.assertEqual(len(relations), 3)
        assignment, group_ids, _ = paper_splits.split_groups(groups)
        self.assertEqual(len(assignment), 4)
        self.assertEqual(len({assignment[key] for key in ["duplicate", "long", "short"]}), 1)
        self.assertEqual(len({group_ids[key] for key in ["duplicate", "long", "short"]}), 1)

    def test_mid_dialogue_subsequence_and_partial_utterance_are_not_prefixes(self):
        rows = [self.dialogue("short", ("你好", "谢谢")), self.dialogue("partial", ("你好啊", "谢谢")),
                self.dialogue("middle", ("开场", "答复", "你好", "谢谢"))]
        groups, relations = paper_splits.related_groups(rows)
        self.assertEqual(len(groups), 3)
        self.assertEqual(relations, [])

    def test_seed_reproducibility_input_order_independence_and_exact_targets(self):
        groups = [[f"d{i:04d}"] for i in range(99)] + [["duplicate1", "duplicate2"]]
        assignment, group_ids, targets = paper_splits.split_groups(groups, seed=42)
        other, other_ids, other_targets = paper_splits.split_groups(list(reversed(groups)), seed=42)
        self.assertEqual((assignment, group_ids, targets), (other, other_ids, other_targets))
        self.assertEqual(targets, {"train": 81, "dev": 10, "test": 10})
        self.assertEqual({split: list(assignment.values()).count(split) for split in paper_splits.SPLITS}, targets)
        self.assertNotEqual(assignment, paper_splits.split_groups(groups, seed=43)[0])

    def test_written_dataset_coverage_hashes_and_new_split_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jsonl"
            rows = [self.dialogue(f"d{i:04d}", (f"问题{i}", f"回复{i}", f"追问{i}")) for i in range(20)]
            paper_splits.write_jsonl(source, rows)
            old_hash = paper_splits.sha256_file(source)
            manifest = paper_splits.build_splits(source, root / "out")
            self.assertEqual(paper_splits.sha256_file(source), old_hash)
            self.assertEqual(manifest["source"]["sha256"], old_hash)
            self.assertTrue(manifest["leakage_audit"]["passed"])
            self.assertEqual(sum(split["dialogues"] for split in manifest["splits"].values()), 20)
            self.assertEqual(sum(split["adjacent_samples"] for split in manifest["splits"].values()), 40)
            self.assertEqual(sum(split["annotations"] for split in manifest["splits"].values()), 240)
            all_ids = set()
            for split in paper_splits.SPLITS:
                output_rows = [json.loads(line) for line in (root / f"out/dialogues/{split}.jsonl").read_text().splitlines()]
                ids = {row["dialogue_id"] for row in output_rows}
                self.assertFalse(all_ids & ids)
                all_ids.update(ids)
                self.assertTrue(all(row["source_split"] == split and row["original_source_split"] == "test" for row in output_rows))
            for relative, metadata in manifest["output_files"].items():
                self.assertEqual(paper_splits.sha256_file(root / "out" / relative), metadata["sha256"])

    def test_loader_rejects_missing_annotator_and_duplicate_turn_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jsonl"
            row = self.dialogue("d")
            row["messages"][0]["annotations"][3]["annotator"] = "paper3"
            paper_splits.write_jsonl(source, [row])
            with self.assertRaisesRegex(ValueError, "paper1, paper2, paper3 and paper4"):
                paper_splits.load_dialogues(source)
            row = self.dialogue("d")
            row["messages"][2]["turn_id"] = 1
            paper_splits.write_jsonl(source, [row])
            with self.assertRaisesRegex(ValueError, "contiguous"):
                paper_splits.load_dialogues(source)

    def test_loader_rejects_bad_confidence_and_blank_text(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.jsonl"
            for confidence in [True, 0, 6, float("nan")]:
                row = self.dialogue("d")
                row["messages"][1]["annotations"][0]["confidence"] = confidence
                paper_splits.write_jsonl(source, [row])
                with self.assertRaisesRegex(ValueError, "confidence"):
                    paper_splits.load_dialogues(source)
            row = self.dialogue("d")
            row["messages"][0]["text"] = " "
            paper_splits.write_jsonl(source, [row])
            with self.assertRaisesRegex(ValueError, "utterance"):
                paper_splits.load_dialogues(source)


if __name__ == "__main__":
    unittest.main()
