"""Business safeguards for the canonical four-annotator dialogue export."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("export_cleaned", Path(__file__).resolve().parents[1] / "tools/export_cleaned_dialogues.py")
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class CleanedExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "original/data"
        self.source.mkdir(parents=True)
        self.output = self.root / "v1.0.0/cleaned_dialogues.jsonl"
        self.raw = [{"dialogue_id": "lccc-dev-00000000", "original_dialogue_id": "lccc-dev-00000004",
                     "source_split": "dev", "turns": ["你 好 ， 新 年 好", "幸 好 2 3 号 又 可 以 放 假 了 ~", "I love cats"]}]
        self.write("dialogues_pool.jsonl", self.raw)
        self.write("meme_catalog.jsonl", [{"meme_id": "meme-0002"}, {"meme_id": "meme-0062"}])
        for annotator in exporter.ANNOTATORS:
            self.write(f"annotations_dialogue_pool_{annotator}.jsonl", [{
                **{key: self.raw[0][key] for key in ("dialogue_id", "original_dialogue_id", "source_split")},
                "annotator": annotator, "model": "deepseek-pro", "prompt_set": "dialogue_v1",
                "turns": [{"turn_id": i, "meme_id": "meme-0062" if i == 1 else "meme-0002",
                           "confidence": 4, "reason": "原始理由", "source": "dialogue_request"} for i in range(3)]}])
        # A malformed legacy file must not be discovered or parsed.
        (self.source / "annotations_pool_paper1.jsonl").write_text("invalid legacy JSON", encoding="utf-8")

    def write(self, name, rows):
        (self.source / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    def mutate_annotation(self, action, annotator="paper4"):
        name = f"annotations_dialogue_pool_{annotator}.jsonl"
        rows = exporter.read_jsonl(self.source / name)
        action(rows)
        self.write(name, rows)

    def test_merge_retains_four_labels_original_id_and_source_without_legacy(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        manifest = exporter.export_dataset(self.source, self.output)
        row = exporter.read_jsonl(self.output)[0]
        self.assertEqual(row["original_dialogue_id"], "lccc-dev-00000004")
        self.assertEqual(row["source_split"], "dev")
        self.assertEqual([turn["role"] for turn in row["messages"]], ["A", "B", "A"])
        self.assertEqual([turn["text"] for turn in row["messages"]], ["你好，新年好", "幸好23号又可以放假了~", "I love cats"])
        for message in row["messages"]:
            self.assertEqual([label["annotator"] for label in message["annotations"]], list(exporter.ANNOTATORS))
            self.assertTrue(all(label["source"] == "dialogue_request" for label in message["annotations"]))
            self.assertTrue(all(label["annotation_provenance"]["model"] == "deepseek-pro" for label in message["annotations"]))
        self.assertEqual((manifest["dialogues"], manifest["turns"], manifest["annotations"]), (1, 3, 12))
        self.assertEqual(manifest["output_sha256"], exporter.sha256_file(self.output))
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, before)
        self.assertEqual(exporter.export_dataset(self.source, self.output), manifest)

    def test_missing_annotator_dialogue_aborts_without_output(self):
        self.write("annotations_dialogue_pool_paper4.jsonl", [])
        with self.assertRaisesRegex(ValueError, "incomplete dialogue coverage"):
            exporter.export_dataset(self.source, self.output)
        self.assertFalse(self.output.parent.exists())

    def test_missing_turn_label_aborts_without_fabricating_annotation(self):
        self.mutate_annotation(lambda rows: rows[0]["turns"].pop())
        with self.assertRaisesRegex(ValueError, "missing or extra turn annotations"):
            exporter.export_dataset(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_unknown_meme_and_invalid_confidence_are_rejected(self):
        self.mutate_annotation(lambda rows: rows[0]["turns"][0].update(meme_id="meme-9999"))
        with self.assertRaisesRegex(ValueError, "unknown meme_id"):
            exporter.export_dataset(self.source, self.output)
        self.mutate_annotation(lambda rows: rows[0]["turns"][0].update(meme_id="meme-0002", confidence=True))
        with self.assertRaisesRegex(ValueError, "confidence must be an integer"):
            exporter.export_dataset(self.source, self.output)

    def test_original_id_conflict_is_not_silently_merged(self):
        self.mutate_annotation(lambda rows: rows[0].update(original_dialogue_id="other"))
        with self.assertRaisesRegex(ValueError, "original_dialogue_id mismatch"):
            exporter.export_dataset(self.source, self.output)

    def test_existing_different_output_is_preserved(self):
        self.output.parent.mkdir()
        self.output.write_bytes(b"precious existing version\n")
        with self.assertRaises(FileExistsError):
            exporter.export_dataset(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b"precious existing version\n")
        self.assertFalse(self.output.with_suffix(".manifest.json").exists())

    def test_output_cannot_modify_original_source_directory(self):
        with self.assertRaisesRegex(ValueError, "outside the original"):
            exporter.export_dataset(self.source, self.source / "new.jsonl")


if __name__ == "__main__":
    unittest.main()
