import json
import tempfile
import unittest
from pathlib import Path

from emostance_meme.data import DataError, normalize_text, prepare_dataset, save_prepared


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in value) + "\n", encoding="utf-8")
        else:
            path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def conversation(self, dialogue_id="d", source_split="train"):
        return {"dialogue_id": dialogue_id, "source_split": source_split, "messages": [
            {"role": "A", "text": "今天很开心", "meme_id": "meme-1", "confidence": 5},
            {"role": "B", "text": "我也替你开心", "meme_id": "meme-2", "confidence": 4},
            {"role": "A", "text": "一起庆祝吧"},
            {"role": "B", "text": "好呀", "meme_id": "meme-3", "confidence": 3},
        ]}

    def test_full_history_and_separate_turn_supervision(self):
        path = self.write("input.json", [self.conversation()])
        data = prepare_dataset(path)
        samples = data["splits"]["train"]
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["messages"], [{"role": "user", "content": "今天很开心"}])
        self.assertEqual([row["role"] for row in samples[1]["messages"]], ["user", "assistant", "user"])
        self.assertEqual(samples[1]["reply"], "好呀")
        self.assertEqual([row["turn_id"] for row in data["turn_records"]["train"]], [0, 1, 3])
        self.assertAlmostEqual(samples[0]["weight"], .8)

    def test_annotation_only_requires_real_text_never_uses_reason(self):
        annotation = {"dialogue_id": "d", "source_split": "dev", "turns": [
            {"turn_id": 0, "meme_id": "meme-1", "confidence": 3, "reason": "这只是理由，不能当对话"},
            {"turn_id": 1, "meme_id": "meme-2", "confidence": 4, "reason": "不能编造原文"},
        ]}
        path = self.write("labels.jsonl", [annotation])
        with self.assertRaisesRegex(DataError, "Missing original text") as failure:
            prepare_dataset(path)
        self.assertEqual(len(failure.exception.report["missing_text"]), 2)
        self.assertIn("labels.jsonl:1", failure.exception.report["missing_text"][0]["locations"][0])
        text = self.write("text.json", [{"dialogue_id": "d", "messages": [{"role": "A", "text": "早上好"}, {"role": "B", "text": "你好呀"}]}])
        data = prepare_dataset(path, [text])
        self.assertEqual(data["splits"]["train"], [])
        self.assertEqual(data["splits"]["dev"][0]["reply"], "你好呀")
        self.assertEqual(data["splits"]["dev"][0]["meme_id"], "meme-2")

    def test_flat_text_join_and_null_text(self):
        annotation = self.write("labels.json", {"dialogue_id": "d", "source_split": "train", "turns": [{"turn_id": 1, "meme_id": "m", "confidence": 3}]})
        text = self.write("text.jsonl", [
            {"dialogue_id": "d", "turn_id": 0, "role": "user", "content": "谢谢"},
            {"dialogue_id": "d", "turn_id": 1, "role": "assistant", "content": "不客气"},
        ])
        self.assertEqual(prepare_dataset(annotation, text)["splits"]["train"][0]["reply"], "不客气")
        bad = self.write("null.json", {"dialogue_id": "n", "messages": [{"role": "A", "text": "你好"}, {"role": "B", "text": None, "meme_id": "m", "confidence": 1}]})
        with self.assertRaisesRegex(DataError, "Missing original text"):
            prepare_dataset(bad)
        skipped = prepare_dataset([annotation, bad], text, strict=False)
        self.assertEqual(skipped["report"]["skipped_dialogues"][0]["dialogue_id"], "n")
        self.assertEqual(skipped["report"]["missing_text"][0]["turn_id"], 1)

    def test_flat_annotations_without_text_join_original_text(self):
        annotation = self.write("flat_labels.jsonl", [
            {"dialogue_id": "flat", "turn_id": 0, "role": "A", "source_split": "dev", "meme_id": "m1", "confidence": 3, "reason": "不能使用理由补文本"},
            {"dialogue_id": "flat", "turn_id": 1, "role": "B", "source_split": "dev", "meme_id": "m2", "confidence": 4},
        ])
        with self.assertRaisesRegex(DataError, "Missing original text"):
            prepare_dataset(annotation)
        original = self.write("flat_text.jsonl", [
            {"dialogue_id": "flat", "turn_id": 0, "text": "原始问题"},
            {"dialogue_id": "flat", "turn_id": 1, "text": "原始回答"},
        ])
        data = prepare_dataset(annotation, original)
        sample = data["splits"]["dev"][0]
        self.assertEqual(sample["reply"], "原始回答")
        self.assertEqual(sample["messages"], [{"role": "user", "content": "原始问题"}])
        self.assertEqual(sample["meme_id"], "m2")

    def test_annotators_soft_labels_deduplicate_and_conflicts(self):
        first = self.conversation()
        first["annotator"] = "first"
        second = {"dialogue_id": "d", "annotator": "second", "source_split": "train", "turns": [{"turn_id": 1, "meme_id": "other", "confidence": 2}]}
        path = self.write("labels.jsonl", [first, first, second])
        data = prepare_dataset(path)
        sample = data["splits"]["train"][0]
        self.assertAlmostEqual(sample["meme_distribution"]["meme-2"], 2 / 3)
        self.assertAlmostEqual(sample["meme_distribution"]["other"], 1 / 3)
        self.assertAlmostEqual(sample["weight"], .6)
        self.assertEqual(data["report"]["duplicate_annotations"], 3)
        self.assertEqual(len(data["splits"]["train"]), 2)
        second["annotator"] = "first"
        conflict = self.write("conflict.json", [first, second])
        with self.assertRaisesRegex(DataError, "conflicting duplicate annotation"):
            prepare_dataset(conflict)

    def test_legacy_context_rows_group_by_dialogue(self):
        path = self.write("legacy.jsonl", [
            {"dialogue_id": "one", "turn_id": 1, "role": "B", "context": "A: 你 好", "text": "你 好 呀", "meme_id": "m1", "confidence": 4},
            {"dialogue_id": "two", "turn_id": 1, "role": "B", "context": "A: 再 见", "text": "拜 拜", "meme_id": "m2", "confidence": 4},
            {"dialogue_id": "one", "turn_id": 3, "role": "B", "context": "A: 你 好\nB: 你 好 呀\nA: 吃 饭 了 吗", "text": "吃 了", "meme_id": "m3", "confidence": 3},
        ])
        data = prepare_dataset(path, dev_ratio=0, test_ratio=0)
        one = [sample for sample in data["splits"]["train"] if sample["dialogue_id"] == "one"]
        self.assertEqual(one[1]["messages"], [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}, {"role": "user", "content": "吃饭了吗"}])
        two = [sample for sample in data["splits"]["train"] if sample["dialogue_id"] == "two"]
        self.assertEqual(two[0]["messages"], [{"role": "user", "content": "再见"}])

    def test_split_stable_no_dialogue_leakage_and_preserve_eval(self):
        rows = [self.conversation(f"dialogue-{i}", "pool") for i in range(100)]
        rows += [self.conversation("held-dev", "dev"), self.conversation("held-test", "test")]
        data = prepare_dataset(self.write("input.json", rows))
        repeated = prepare_dataset(self.write("reverse.json", list(reversed(rows))))
        self.assertEqual(data["splits"], repeated["splits"])
        ids = {name: {row["dialogue_id"] for row in samples} for name, samples in data["splits"].items()}
        self.assertTrue(all(ids.values()))
        self.assertFalse(ids["train"] & ids["dev"] or ids["train"] & ids["test"] or ids["dev"] & ids["test"])
        self.assertIn("held-dev", ids["dev"])
        self.assertIn("held-test", ids["test"])
        self.assertEqual(len(ids["train"] | ids["dev"] | ids["test"]), 102)

    def test_conflicting_splits_and_malformed_roles_rejected(self):
        path = self.write("splits.json", [self.conversation("d", "dev"), self.conversation("d", "train")])
        with self.assertRaisesRegex(DataError, "conflicting explicit source splits"):
            prepare_dataset(path)
        row = self.conversation()
        row["messages"][2]["role"] = "B"
        path = self.write("roles.json", row)
        with self.assertRaisesRegex(DataError, "malformed alternation"):
            prepare_dataset(path)

    def test_confidence_and_empty_label_validation(self):
        for confidence in (0, -1, 6, float("nan"), float("inf"), True, None, "4"):
            with self.subTest(confidence=confidence):
                row = self.conversation()
                row["messages"][1]["confidence"] = confidence
                with self.assertRaisesRegex(DataError, "confidence"):
                    prepare_dataset(self.write("bad.json", row))
        row = self.conversation()
        row["messages"][1]["meme_id"] = ""
        with self.assertRaisesRegex(DataError, "nonempty meme_id"):
            prepare_dataset(self.write("bad.json", row))
        row = self.conversation()
        data = prepare_dataset(self.write("valid.json", row), max_confidence=10)
        self.assertEqual(data["splits"]["train"][0]["weight"], .4)

    def test_unannotated_assistant_gets_no_fake_emotion_label(self):
        path = self.write("unlabeled.json", {"dialogue_id": "d", "source_split": "train", "messages": [{"role": "A", "text": "你好"}, {"role": "B", "text": "你好呀"}]})
        data = prepare_dataset(path)
        sample = data["splits"]["train"][0]
        self.assertIsNone(sample["meme_id"])
        self.assertEqual(sample["meme_distribution"], {})
        self.assertEqual(sample["weight"], 0)
        self.assertEqual(data["turn_records"]["train"], [])

    def test_directory_discovery_and_saved_outputs(self):
        path = self.write("inputs/sub/input.jsonl", [self.conversation()])
        self.write("inputs/report.json", {"not": "dataset"})
        data = prepare_dataset(path.parent.parent)
        output = self.root / "prepared"
        save_prepared(data, output)
        self.assertEqual(len((output / "train.jsonl").read_text().splitlines()), 2)
        self.assertEqual(len((output / "turn_records/train.jsonl").read_text().splitlines()), 3)
        self.assertEqual(json.loads((output / "report.json").read_text())["counts"]["train"]["dialogues"], 1)

    def test_empty_directory_and_text_normalization(self):
        empty = self.root / "empty"
        empty.mkdir()
        (empty / ".gitkeep").touch()
        with self.assertRaisesRegex(DataError, "No JSON/JSONL"):
            prepare_dataset(empty)
        self.assertEqual(normalize_text("幸 好 2 3 号 又 可 以 放 假 了 ~"), "幸好23号又可以放假了~")
        self.assertEqual(normalize_text("hello world"), "hello world")


if __name__ == "__main__":
    unittest.main()
