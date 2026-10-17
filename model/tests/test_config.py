import json
import tempfile
import unittest
from pathlib import Path

from emostance_meme.config import Config, MODEL_ROOT, local_backbone, work_path


class ConfigTests(unittest.TestCase):
    def test_default_config_and_model_only_outputs(self):
        config = Config.load()
        self.assertEqual(config.idle_timeout, 10.0)
        self.assertEqual(work_path('runs/test'), MODEL_ROOT / 'runs/test')
        with self.assertRaises(ValueError):
            work_path('../dataset/output')

    def test_missing_model_shard_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'test': 'missing.safetensors'}}))
            with self.assertRaisesRegex(ValueError, '分片缺失'):
                local_backbone(root)

    def test_invalid_training_and_split_settings(self):
        for kwargs in [{'head_epochs': 0}, {'head_epochs': 1.5}, {'head_batch_size': True}, {'annotations': 'file.jsonl'}, {'max_reply_tokens': 1}, {'idle_timeout': float('nan')}, {'dev_ratio': .8, 'test_ratio': .3}, {'distance': 'text_similarity'}]:
            with self.assertRaises(ValueError):
                Config(**kwargs).validate()


if __name__ == '__main__':
    unittest.main()
