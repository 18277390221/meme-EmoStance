"""Exercise destructive-cleaning safeguards using disposable synthetic data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("dataset_cleaner", Path(__file__).resolve().parents[1] / "tools/clean_dataset.py")
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)

export_spec = importlib.util.spec_from_file_location("dataset_exporter", Path(__file__).resolve().parents[1] / "tools/export_cleaned_dialogues.py")
exporter = importlib.util.module_from_spec(export_spec)
export_spec.loader.exec_module(exporter)


class DatasetCleaningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        raw = [{"dialogue_id": f"lccc-train-{i:08d}", "source_split": "train", "turns": ["你好", "回复"]} for i in range(1, 6)]
        raw.insert(1, {"dialogue_id": "lccc-dev-00000100", "source_split": "dev", "turns": ["你好", "回复"]})
        cleaner.write_rows(self.data / "dialogues_pool.jsonl", raw)
        cleaner.write_rows(self.data / "meme_catalog.jsonl", [{"meme_id": "meme-0001"}, {"meme_id": "meme-0002"}])
        for annotator in cleaner.ANNOTATORS:
            rows, legacy = [], []
            for row in raw:
                identifier = row["dialogue_id"]
                if identifier == "lccc-train-00000004" or (identifier == "lccc-train-00000003" and annotator == "paper4"):
                    continue
                turns = [{"turn_id": i, "meme_id": f"meme-{i + 1:04d}", "confidence": 4, "reason": "正常理由"} for i in range(2)]
                if identifier == "lccc-train-00000002" and annotator == "paper1":
                    turns[0]["reason"] = "乱码\ufffd"
                response = json.dumps({"turns": turns}, ensure_ascii=False)
                if identifier == "lccc-dev-00000100":
                    alternate = json.loads(response)
                    alternate["turns"][0]["reason"] = "仅措辞改写"
                    response = "```json\n" + response + "\n```\n" + json.dumps(alternate)
                if identifier == "lccc-train-00000005" and annotator == "paper3":
                    alternate = json.loads(response)
                    alternate["turns"][0]["meme_id"] = "meme-0002"
                    response += "\n修正：" + json.dumps(alternate)
                rows.append({"dialogue_id": identifier, "source_split": row["source_split"], "annotator": annotator,
                             "requested_turn_ids": [0, 1], "turns": turns, "raw_response_text": response})
                if identifier in {"lccc-train-00000001", "lccc-dev-00000100"}:
                    label = turns[1]
                    legacy.append({"dialogue_id": identifier, "turn_id": 1, "role": "B", "text": "回复", "context": "A: 你好",
                                   "annotator": annotator, **label, "raw_response_text": json.dumps({k: label[k] for k in ("meme_id", "confidence", "reason")})})
            cleaner.write_rows(self.data / f"annotations_dialogue_pool_{annotator}.jsonl", rows)
            cleaner.write_rows(self.data / f"annotations_pool_{annotator}.jsonl", legacy)
        (self.root / "statistics.json").write_text('{"before":true}')
        (self.root / "data.7z").write_bytes(b"original archive preserved")

    def test_dry_run_distinguishes_real_errors_from_reason_rewrites(self):
        original = (self.data / "dialogues_pool.jsonl").read_bytes()
        report = cleaner.run(self.root)
        self.assertEqual(report["retained_dialogues"], 2)
        self.assertEqual(report["deleted_dialogues"], 4)
        self.assertEqual(report["deletion_categories"], {"invalid_annotation": 2, "missing_annotators": 1, "no_annotations": 1})
        self.assertEqual(report["retained_reason_only_rewrites"], ["lccc-dev-00000100"])
        self.assertEqual((self.data / "dialogues_pool.jsonl").read_bytes(), original)
        self.assertFalse((self.root / "backups").exists())

    def test_apply_preserves_originals_and_updates_all_references(self):
        original = {path.name: path.read_bytes() for path in self.data.iterdir()}
        archive = (self.root / "data.7z").read_bytes()
        report = cleaner.run(self.root, apply=True)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["after"]["dialogues"], 2)
        self.assertEqual(report["after"]["turns"], 4)
        backup = Path(report["backup_directory"])
        for name, value in original.items():
            self.assertEqual((backup / "data" / name).read_bytes(), value)
        self.assertEqual(json.loads((backup / "statistics.json").read_text()), {"before": True})
        self.assertEqual((self.root / "data.7z").read_bytes(), archive)
        after = cleaner.audit(self.data)
        self.assertFalse(after["issues"])
        expected = {"lccc-train-00000000", "lccc-dev-00000000"}
        self.assertEqual(set(after["dialogues"]), expected)
        for rows in after["new"].values():
            self.assertEqual({row["dialogue_id"] for row in rows}, expected)
            self.assertTrue(all("original_dialogue_id" in row for row in rows))
        for rows in after["legacy"].values():
            self.assertEqual({row["dialogue_id"] for row in rows}, expected)
        mapping = cleaner.load_rows(self.root / "dialogue_id_mapping.jsonl")
        self.assertEqual(len(mapping), 2)
        self.assertEqual(len(cleaner.load_rows(self.root / "deleted_dialogues.jsonl")), 4)

    def test_modern_only_export_cleans_and_exports_without_creating_legacy_files(self):
        for annotator in cleaner.ANNOTATORS:
            (self.data / f"annotations_pool_{annotator}.jsonl").unlink()
        report = cleaner.run(self.root, apply=True)
        self.assertEqual(report["after"]["dialogues"], 2)
        self.assertEqual(report["after"]["legacy_rows_per_annotator"], {annotator: 0 for annotator in cleaner.ANNOTATORS})
        self.assertFalse(list(self.data.glob("annotations_pool_*.jsonl")))
        self.assertFalse(list((Path(report["backup_directory"]) / "data").glob("annotations_pool_*.jsonl")))
        manifest = exporter.export_dataset(self.data, self.root / "canonical/cleaned_dialogues.jsonl")
        self.assertEqual(manifest["dialogues"], 2)
        self.assertEqual(manifest["turns"], 4)
        self.assertEqual(manifest["annotations"], 16)

    def test_present_invalid_legacy_annotation_still_rejects_entire_dialogue(self):
        path = self.data / "annotations_pool_paper1.jsonl"
        rows = cleaner.load_rows(path)
        rejected_id = rows[0]["dialogue_id"]
        rows[0]["confidence"] = 0
        cleaner.write_rows(path, rows)
        audited = cleaner.audit(self.data)
        self.assertIn("invalid_confidence", {issue["code"] for issue in audited["issues"][rejected_id]})
        self.assertNotIn(rejected_id, {row["dialogue_id"] for row in audited["kept"]})
        report = cleaner.run(self.root, apply=True)
        self.assertEqual(report["after"]["dialogues"], 1)
        self.assertTrue(path.is_file())
        self.assertEqual(len(cleaner.load_rows(path)), 1)
        self.assertEqual(cleaner.load_rows(path)[0]["original_dialogue_id"], "lccc-dev-00000100")

    def test_unidentifiable_invalid_json_stops_without_modifying_data(self):
        path = self.data / "annotations_dialogue_pool_paper1.jsonl"
        path.write_text('{invalid json\n')
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            cleaner.run(self.root, apply=True)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.root / "backups").exists())

    def test_exact_prefix_rules_keep_whitespace_variants_branches_and_equal_records(self):
        def row(identifier, turns):
            return {"dialogue_id": identifier, "source_split": "train", "turns": turns}
        rows = [row('short', ['A', 'B']), row('equal', ['A', 'B']),
                row('middle', ['A', 'B', 'C']), row('long', ['A', 'B', 'C', 'D']),
                row('branch', ['A', 'B', 'E']), row('space', [' A', 'B']),
                row('internal', ['B', 'C'])]
        pairs = cleaner.prefix_pairs(rows)
        short_ids = {pair['short_dialogue_id'] for pair in pairs}
        self.assertEqual(short_ids, {'short', 'equal', 'middle'})
        self.assertNotIn(('short', 'equal'), {(p['short_dialogue_id'], p['long_dialogue_id']) for p in pairs})
        self.assertNotIn('space', short_ids)
        self.assertNotIn('internal', short_ids)
        remaining = [row for row in rows if row['dialogue_id'] not in short_ids]
        self.assertFalse(cleaner.prefix_pairs(remaining))

    def test_equal_length_duplicate_dialogues_are_not_prefixes(self):
        data = cleaner.audit(self.data)
        self.assertEqual(len(data['kept']), 2)
        self.assertFalse(cleaner.prefix_pairs(data['kept']))

    def test_apply_exact_prefix_removes_short_turns_and_synchronizes_four_annotators(self):
        identifier = 'lccc-dev-00000101'
        raw = cleaner.load_rows(self.data / 'dialogues_pool.jsonl')
        raw.append({'dialogue_id': identifier, 'source_split': 'dev', 'turns': ['你好', '回复', '继续', '回答']})
        cleaner.write_rows(self.data / 'dialogues_pool.jsonl', raw)
        for annotator in cleaner.ANNOTATORS:
            path = self.data / f'annotations_dialogue_pool_{annotator}.jsonl'
            rows = cleaner.load_rows(path)
            turns = [{'turn_id': i, 'meme_id': f'meme-{i % 2 + 1:04d}', 'confidence': 4, 'reason': '正常理由'} for i in range(4)]
            rows.append({'dialogue_id': identifier, 'source_split': 'dev', 'annotator': annotator, 'requested_turn_ids': [0, 1, 2, 3],
                         'turns': turns, 'raw_response_text': json.dumps({'turns': turns})})
            cleaner.write_rows(path, rows)
        report = cleaner.run(self.root, apply=True, remove_exact_prefixes=True)
        self.assertEqual(report['deletion_categories']['exact_dialogue_prefix'], 2)
        self.assertEqual(report['after']['dialogues'], 1)
        self.assertEqual(report['after']['turns'], 4)
        after = cleaner.audit(self.data)
        self.assertEqual(after['raw'][0]['turns'], ['你好', '回复', '继续', '回答'])
        self.assertTrue(all(len(rows) == 1 for rows in after['new'].values()))
        self.assertFalse(cleaner.prefix_pairs(after['raw']))
        self.assertTrue((self.root / 'prefix_cleaning_report.json').exists())


if __name__ == "__main__":
    unittest.main()
