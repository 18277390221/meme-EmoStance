import unittest

import numpy as np

from emostance_meme.training import _source_lookup, _transition


class TinySpace:
    num_clusters = 2
    def targets(self, distribution):
        vector = np.array([distribution.get('a', 0), distribution.get('b', 0)], dtype=float)
        return vector, vector


class TrainingWiringTests(unittest.TestCase):
    def test_repeated_user_text_keeps_correct_turn_transition(self):
        records = [
            {'dialogue_id': 'd', 'turn_id': 0, 'role': 'user', 'reply': '同一句话', 'meme_distribution': {'a': 1}},
            {'dialogue_id': 'd', 'turn_id': 2, 'role': 'user', 'reply': '同一句话', 'meme_distribution': {'b': 1}},
        ]
        samples = [
            {'dialogue_id': 'd', 'turn_id': 1, 'meme_distribution': {'a': 1}, 'weight': 1},
            {'dialogue_id': 'd', 'turn_id': 3, 'meme_distribution': {'b': 1}, 'weight': 1},
        ]
        transition = _transition(TinySpace(), samples, _source_lookup(records))
        self.assertGreater(transition[0, 0], transition[0, 1])
        self.assertGreater(transition[1, 1], transition[1, 0])
        np.testing.assert_allclose(transition.sum(1), 1)

    def test_missing_source_annotation_stays_smoothed(self):
        transition = _transition(TinySpace(), [{'dialogue_id': 'd', 'turn_id': 1, 'meme_distribution': {'a': 1}}], {})
        np.testing.assert_allclose(transition, [[.5, .5], [.5, .5]])


if __name__ == '__main__':
    unittest.main()
