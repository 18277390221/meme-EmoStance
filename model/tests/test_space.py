"""Meaningful checks for confidence weighting and train-only weak supervision."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from emostance_meme.space import EmotionSpace


def turn(meme: str, weight: float = 1.0) -> dict:
    return {"reply": "训练原文", "meme_distribution": {meme: 1.0}, "weight": weight, "role": "B"}


class EmotionSpaceTests(unittest.TestCase):
    def test_confidence_changes_centroid_without_using_text_or_label_names(self) -> None:
        records = [turn("opaque-a", 9.0), turn("opaque-a", 1.0)]
        space = EmotionSpace.fit(records, np.eye(2), vector_dim=2, shrinkage=0)
        expected = np.asarray([9.0, 1.0]) / np.linalg.norm([9.0, 1.0])
        np.testing.assert_allclose(space.centroids[0], expected)
        renamed = [dict(record, reply="完全不同的原文", meme_distribution={"other-key": 1}) for record in records]
        other = EmotionSpace.fit(renamed, np.eye(2), vector_dim=2, shrinkage=0)
        np.testing.assert_array_equal(space.centroids, other.centroids)

    def test_soft_votes_project_linearly_and_prototypes_respect_confidence(self) -> None:
        records = [
            turn("a", 4),
            turn("b", 1),
            {"meme_distribution": {"a": 1, "b": 3}, "weight": 2},
        ]
        space = EmotionSpace.fit(records, np.asarray([[1, 0], [0, 1], [1, 1]]), vector_dim=2, num_clusters=2)
        q_a, v_a = space.targets({"a": 1})
        q_b, v_b = space.targets({"b": 1})
        q_mix, v_mix = space.targets({"a": 1, "b": 3})
        np.testing.assert_allclose(q_mix, 0.25 * q_a + 0.75 * q_b)
        np.testing.assert_allclose(v_mix, 0.25 * v_a + 0.75 * v_b)
        self.assertTrue((space.membership > 0).all())
        np.testing.assert_allclose(space.membership.sum(axis=1), 1)
        qs = np.stack([q_a, q_b, q_mix])
        vs = np.stack([v_a, v_b, v_mix])
        weights = np.asarray([4, 1, 2])
        expected = ((weights[:, None] * qs).T @ vs) / (weights @ qs)[:, None]
        np.testing.assert_allclose(space.prototypes, expected)

    def test_same_train_input_and_seed_produce_identical_artifacts(self) -> None:
        rng = np.random.default_rng(19)
        features = rng.normal(size=(30, 15))
        records = [turn(f"key-{index % 6}") for index in range(30)]
        first = EmotionSpace.fit(records, features, vector_dim=4, num_clusters=3, seed=7)
        second = EmotionSpace.fit(records, features, vector_dim=4, num_clusters=3, seed=7)
        for key in ("centroids", "membership", "prototypes"):
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
        self.assertEqual(first.vector_dim, 4)
        self.assertEqual(first.num_clusters, 3)

    def test_held_out_feature_changes_cannot_modify_training_artifacts(self) -> None:
        train = np.asarray([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=float)
        all_features = np.concatenate([train, [[1000, -999, 888]]])
        records = [turn("a"), turn("b"), turn("a")]
        first = EmotionSpace.fit(records, all_features[:3], vector_dim=2)
        all_features[3] *= -1234
        second = EmotionSpace.fit(records, all_features[:3], vector_dim=2)
        np.testing.assert_array_equal(first.centroids, second.centroids)
        np.testing.assert_array_equal(first.membership, second.membership)
        with self.assertRaisesRegex(ValueError, "Unknown meme"):
            first.targets({"test-only-meme": 1})

    def test_unknown_ids_and_malformed_distributions_fail_explicitly(self) -> None:
        space = EmotionSpace.fit([turn("a")], np.asarray([[1.0, 2.0]]))
        for distribution in ({"a": 1, "unseen": 1}, {"unseen": 0, "a": 1}):
            with self.assertRaisesRegex(ValueError, "Unknown meme"):
                space.targets(distribution)
        for distribution in ({}, {"a": 0}, {"a": -1}, {"a": float("nan")}):
            with self.assertRaises(ValueError):
                space.targets(distribution)

    def test_single_sample_single_label_and_zero_features_remain_finite(self) -> None:
        for features in (np.ones((1, 20)), np.zeros((1, 20)), np.zeros((3, 20))):
            records = [turn("only")] * len(features)
            space = EmotionSpace.fit(records, features, vector_dim=4, num_clusters=8)
            self.assertGreaterEqual(space.vector_dim, 1)
            self.assertEqual(space.num_clusters, 1)
            q, v = space.targets({"only": 4})
            np.testing.assert_array_equal(q, [1])
            self.assertTrue(np.isfinite(v).all())
            self.assertTrue(np.isfinite(space.prototypes).all())

    def test_identical_features_and_tiny_temperature_do_not_empty_regions(self) -> None:
        records = [turn(f"m-{index}") for index in range(5)]
        space = EmotionSpace.fit(records, np.ones((5, 3)), num_clusters=3, temperature=1e-12)
        self.assertEqual(space.num_clusters, 3)
        self.assertTrue(np.isfinite(space.prototypes).all())
        np.testing.assert_allclose(space.membership.sum(axis=1), 1)

    def test_save_load_roundtrip_and_invalid_artifacts(self) -> None:
        space = EmotionSpace.fit([turn("a"), turn("b")], np.eye(2), num_clusters=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "space.json"
            space.save(path)
            restored = EmotionSpace.load(path)
            self.assertEqual(restored.meme_ids, space.meme_ids)
            for key in ("centroids", "membership", "prototypes"):
                np.testing.assert_array_equal(getattr(restored, key), getattr(space, key))
            broken = json.loads(path.read_text())
            broken["membership"][0] = [0.0, 0.0]
            path.write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError, "probability"):
                EmotionSpace.load(path)

    def test_nonfinite_and_invalid_dimensions_fail_before_fitting(self) -> None:
        records = [turn("a")]
        for features in (np.ones((0, 2)), np.ones((1, 0)), np.ones((2, 2)), np.asarray([[np.inf]])):
            with self.assertRaises(ValueError):
                EmotionSpace.fit(records, features)
        for options in ({"vector_dim": 0}, {"num_clusters": 0}, {"shrinkage": -1}, {"temperature": 0}):
            with self.assertRaises(ValueError):
                EmotionSpace.fit(records, np.ones((1, 2)), **options)
        with self.assertRaisesRegex(ValueError, "confidence"):
            EmotionSpace.fit([turn("a", 0)], np.ones((1, 2)))


if __name__ == "__main__":
    unittest.main()
