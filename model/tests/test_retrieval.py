import json
import math
import tempfile
import unittest
from pathlib import Path

from emostance_meme.retrieval import ResponseIndex


class RetrievalTests(unittest.TestCase):
    def records(self):
        return [
            {"reply": "太开心了", "meme_id": "meme-happy", "vector": [1, 0], "dialogue_id": "a", "turn_id": 1},
            {"reply": "抱抱你", "meme_id": "meme-sad", "vector": [0, 1], "dialogue_id": "b", "turn_id": 1},
            {"reply": "这可不好", "meme_id": "meme-angry", "vector": [-1, 0], "dialogue_id": "c", "turn_id": 1},
        ]

    def test_cosine_nearest_returns_anchor_meme_and_distance(self):
        index = ResponseIndex(self.records())
        result = index.nearest([10, 1])
        self.assertEqual(result["meme_id"], "meme-happy")
        self.assertEqual(result["reply"], "太开心了")
        self.assertAlmostEqual(result["distance"], 1 - 10 / math.sqrt(101))
        self.assertEqual(index.search([1, 0], 3)[-1]["distance"], 2)

    def test_euclidean_differs_from_cosine(self):
        records = [{"reply": "far", "meme_id": "m1", "vector": [100, 0]}, {"reply": "near", "meme_id": "m2", "vector": [1, 1]}]
        self.assertEqual(ResponseIndex(records, metric="cosine").nearest([1, 0])["meme_id"], "m1")
        euclidean = ResponseIndex(records, metric="euclidean").nearest([1, 0])
        self.assertEqual(euclidean["meme_id"], "m2")
        self.assertEqual(euclidean["distance"], 1)

    def test_ties_deterministic_across_input_order(self):
        records = [{"reply": "b", "meme_id": "m2", "vector": [1, 0]}, {"reply": "a", "meme_id": "m1", "vector": [1, 0]}]
        self.assertEqual(ResponseIndex(records).nearest([1, 0])["meme_id"], "m1")
        self.assertEqual(ResponseIndex(reversed(records)).nearest([1, 0])["meme_id"], "m1")

    def test_invalid_vectors_dimensions_and_k(self):
        index = ResponseIndex(self.records())
        for vector in ([], [0, 0], [1], [1, 2, 3], [math.nan, 1], [math.inf, 1], [True, 1], None):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    index.nearest(vector)
        with self.assertRaisesRegex(ValueError, "dimension"):
            ResponseIndex(self.records() + [{"reply": "extra", "meme_id": "m", "vector": [1, 1, 1]}])
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                index.search([1, 0], k)
        self.assertEqual(len(index.search([1, 0], 50)), 3)

    def test_no_empty_bank_no_empty_meme_and_no_eval_leakage(self):
        with self.assertRaisesRegex(ValueError, "without labeled"):
            ResponseIndex([])
        for change in ({"meme_id": None}, {"reply": ""}, {"split": "dev"}, {"source_split": "test"}):
            row = {**self.records()[0], **change}
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    ResponseIndex([row])

    def test_persistence_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            index = ResponseIndex(self.records(), "euclidean")
            index.save(path)
            restored = ResponseIndex.load(path)
            self.assertEqual(restored.metric, "euclidean")
            self.assertEqual(index.nearest([1, .5]), restored.nearest([1, .5]))
            bad = json.loads(path.read_text())
            bad["dimension"] = 99
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "dimension"):
                ResponseIndex.load(path)

    def test_large_finite_vectors_remain_finite_for_cosine(self):
        index = ResponseIndex([{"reply": "a", "meme_id": "m", "vector": [1e300, 1e300]}])
        distance = index.nearest([1e-300, 1e-300])["distance"]
        self.assertTrue(math.isfinite(distance))
        self.assertAlmostEqual(distance, 0)

    def test_returned_vector_is_a_copy(self):
        index = ResponseIndex(self.records())
        index.nearest([1, 0])["vector"][0] = -1
        self.assertEqual(index.nearest([1, 0])["vector"], [1, 0])

    def test_numpy_float32_query_and_records(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is optional for the standard-library retrieval layer")
        records = self.records()
        for record in records:
            record["vector"] = np.asarray(record["vector"], dtype=np.float32)
        index = ResponseIndex(records)
        result = index.nearest(np.asarray([1, .1], dtype=np.float32))
        self.assertEqual(result["meme_id"], "meme-happy")
        self.assertTrue(math.isfinite(result["distance"]))


if __name__ == "__main__":
    unittest.main()
