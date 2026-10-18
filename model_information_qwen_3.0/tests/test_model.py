"""The V3 model swap requires explicit authorization and complete loss evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from model_information_qwen_v3 import model, release


class CompletedLossQwenRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run, self.data, self.base = self.root/'run', self.root/'data', self.root/'qwen'
        self.config_path = self.root/'app/inference_config.json'
        self.settings_path = self.root/'app/chat_config.json'
        self.release_path = self.root/'app/inference_release.json'
        self.prompt = '保持原角色，简短中文回复。'
        self.write(self.root/'prompt.txt', self.prompt.encode())
        self.config = dict(version='qwen3_8b_v3_loss_ce', model_version='3.0.0', data_version='2.0.0',
            run_dir=str(self.run), data_dir=str(self.data), base_model_path=str(self.base),
            planner_model_path=str(self.root/'planner'), system_prompt_file=str(self.root/'prompt.txt'),
            system_prompt=self.prompt, device='cpu', dtype='float32', prefix_epochs=3,
            max_prompt_tokens=512, max_new_tokens=128, do_sample=True, temperature=.7, top_p=.8,
            top_k=20, no_repeat_ngram_size=4, enable_thinking=False,
            chat_quality_enabled=False, generator_selection_mode='dev_response_ce',
            training_objective='pure_reply_ce')
        trained = copy.deepcopy(self.config)
        self.config['serving_max_prompt_tokens'] = 4096
        self.write(self.config_path, self.config)
        settings = dict(role_prompt=str(self.root/'prompt.txt'), asr_prompt=str(self.root/'prompt.txt'),
            asr_enabled=False, asr_history_messages=6, asr_max_new_tokens=192,
            max_prompt_tokens=4096, max_reply_tokens=128, do_sample=True,
            temperature=.7, top_p=.8, no_repeat_ngram_size=4)
        self.write(self.settings_path, settings)
        for name in model._REQUIRED:
            self.write(self.run/name, {} if name.endswith('.json') else b'trained local artifact')
        self.write(self.run/'graph/manifest.json', dict(status='complete'))
        for split in ('train', 'dev', 'test'):
            self.write(self.data/f'{split}.jsonl', b'{"split":"'+split.encode()+b'"}\n')
        self.write(self.run/'planner/manifest.json', dict(status='complete'))
        self.write(self.run/'generator/dev_subset.json', dict(split='dev'))
        self.generator = dict(status='complete', generator_selection_mode='dev_response_ce',
            selection_criterion='dev_response_ce', quality_status='not_evaluated', chat_quality_status='not_evaluated',
            version=self.config['version'], model_version='3.0.0', data_version='2.0.0',
            best_dev_response_ce=3.918008093610924, best_step=150000, best_epoch=3, global_step=157137,
            train_examples=52379, epochs_complete=3, epochs_requested=3,
            checkpoint_sha256=release._sha(self.run/'generator/best.pt'))
        self.write(self.run/'generator/manifest.json', self.generator)
        self.selection = dict(version=self.config['version'], model_version='3.0.0', data_version='2.0.0',
            best_checkpoint_sha256=self.generator['checkpoint_sha256'],
            dev_subset_sha256=release._sha(self.run/'generator/dev_subset.json'),
            best_dev_response_ce=3.918008093610924, best_step=150000, best_epoch=3, global_step=157137,
            control='planner_predicted', weighting='reply_eos_token_weighted', selection_uses_test=False,
            chat_quality_evaluated=False)
        self.write(self.run/'generator/loss_selection.json', self.selection)
        self.evaluation = dict(version=self.config['version'], model_version='3.0.0', data_version='2.0.0',
            evaluation_mode='response_ce_only', checkpoint_selection='dev_response_ce',
            checkpoint_sha256=self.generator['checkpoint_sha256'], chat_quality_status='not_evaluated',
            test_used_for_training=False, test_used_for_checkpoint_selection=False)
        for split in ('dev', 'test'):
            self.evaluation[split] = dict(source_file_sha256=release._sha(self.data/f'{split}.jsonl'),
                generator=dict(token_ce=dict(gold_control=3.8, predicted_control=3.9)))
        self.write(self.run/'evaluation.json', self.evaluation)
        self.index = dict(status='complete', split='train',
            train_file_sha256=release._sha(self.data/'train.jsonl'),
            planner_checkpoint_sha256=release._sha(self.run/'planner/best.pt'),
            space_sha256=release._sha(self.run/'space.json'))
        self.write(self.run/'response_index_manifest.json', self.index)
        names = ('generator/best.pt', 'planner/best.pt', 'space.json', 'generator/dev_subset.json',
                 'generator/loss_selection.json', 'evaluation.json', 'response_vectors.npz', 'response_anchors.jsonl')
        self.manifest = dict(version=self.config['version'], model_version='3.0.0', data_version='2.0.0',
            status='ready_for_review', training_status='complete', inference_enabled=False,
            inference_release_status='not_authorized', quality_status='not_evaluated',
            chat_quality_status='not_evaluated', selection_criterion='dev_response_ce', config=trained,
            artifact_sha256={name:release._sha(self.run/name) for name in names})
        self.write(self.run/'manifest.json', self.manifest)
        self.write(self.base/'config.json', dict(model_type='qwen3'))
        self.write(self.base/'model.safetensors', b'local base weights')
        self.write(self.base/'tokenizer.json', {})
        self.write(self.base/'tokenizer_config.json', {})
        for target, name, value in ((release, 'DEFAULT_RELEASE_PATH', self.release_path),
                                    (release, 'DEFAULT_SETTINGS_PATH', self.settings_path),
                                    (model, 'DEFAULT_CONFIG_PATH', self.config_path)):
            mock = patch.object(target, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        pipeline = release._pipeline()
        mock = patch.object(pipeline, 'MODEL_DIR', self.root)
        mock.start()
        self.addCleanup(mock.stop)
        mock = patch.object(release, '_EXPECTED_CHECKPOINT_SHA256', self.generator['checkpoint_sha256'])
        mock.start()
        self.addCleanup(mock.stop)
        self.record = release.build_inference_release(self.config, self.config_path, self.settings_path,
            authorization_text=release.AUTHORIZATION_TEXT)
        self.write(self.release_path, self.record)

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(value, ensure_ascii=False).encode() if isinstance(value, dict) else value)

    def guard(self, config=None):
        return release.require_inference_release(config or self.config, self.config_path, self.settings_path)

    def test_authorized_completed_loss_run_passes_without_quality_reports(self):
        self.assertEqual(self.guard()['release_mode'], 'user_selected_loss_only')
        self.assertFalse((self.run/'generator/quality_dev.json').exists())
        self.assertFalse((self.run/'generator/quality_holdout.json').exists())
        model._ready(self.config, verify=False, config_path=self.config_path, settings_path=self.settings_path)
        model._backbone(self.config)
        self.assertEqual(json.loads((self.run/'manifest.json').read_text()), self.manifest)

    def test_inspection_reports_real_loss_selection_and_not_evaluated_quality(self):
        with patch.object(model, '_configuration', return_value=(self.config_path, self.config)):
            info = model.inspect_model_version()
        self.assertEqual(info['version'], '3.0.0')
        self.assertEqual(info['status'], 'ready_for_review')
        self.assertEqual(info['release_status'], 'ready')
        self.assertEqual(info['quality_status'], 'not_evaluated')
        self.assertEqual(info['selected_step'], 150000)
        self.assertEqual(info['best_dev_response_ce'], 3.918008093610924)
        self.assertFalse(info['dialogue_policy']['enabled'])

    def test_missing_or_changed_authorization_blocks_before_backbone_even_verify_false(self):
        for bad in (None, {**self.record, 'quality_status':'passed'},
                    {**self.record, 'authorization':dict(user_selection=False, text=release.AUTHORIZATION_TEXT)}):
            with self.subTest(record=bad):
                if bad is None:
                    self.release_path.unlink()
                else:
                    self.write(self.release_path, bad)
                with patch.object(model, '_configuration', return_value=(self.config_path, self.config)), \
                     patch.object(model, '_backbone', side_effect=AssertionError('weights must not load')):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        model.load_engine(device='cpu', verify=False, settings_path=self.settings_path)
        self.write(self.release_path, self.record)

    def test_all_bound_artifacts_are_mandatory_with_verify_false(self):
        for name in release._ARTIFACTS:
            with self.subTest(name=name):
                path = self.run/name
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaises(ValueError):
                    self.guard()
                path.write_bytes(original)

    def test_partial_training_or_other_version_cannot_receive_authorization(self):
        for update in ({'status':'incomplete'}, {'epochs_complete':2}, {'global_step':157136},
                       {'model_version':'2.0.0'}, {'quality_status':'passed'}):
            with self.subTest(update=update):
                self.write(self.run/'generator/manifest.json', {**self.generator, **update})
                with self.assertRaises(ValueError):
                    release.build_inference_release(self.config, self.config_path, self.settings_path,
                        authorization_text=release.AUTHORIZATION_TEXT)
        self.write(self.run/'generator/manifest.json', self.generator)

    def test_other_user_text_or_checkpoint_selection_is_not_authorized(self):
        with self.assertRaisesRegex(ValueError, 'explicit user selection'):
            release.build_inference_release(self.config, self.config_path, authorization_text='please train')
        for update in ({'best_step':149000}, {'selection_uses_test':True}, {'chat_quality_evaluated':True}):
            self.write(self.run/'generator/loss_selection.json', {**self.selection, **update})
            with self.assertRaises(ValueError):
                self.guard()
        self.write(self.run/'generator/loss_selection.json', self.selection)

    def test_serving_prompt_limit_is_separate_from_unchanged_training_limit(self):
        self.assertEqual(self.config['max_prompt_tokens'], 512)
        self.assertEqual(self.config['serving_max_prompt_tokens'], 4096)
        for update in ({'max_prompt_tokens':4096}, {'serving_max_prompt_tokens':512}, {'top_k':10},
                       {'system_prompt':'changed'}, {'prefix_length':9}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.guard({**self.config, **update})
        self.assertEqual(self.guard({**self.config, 'device':'cuda:1', 'dtype':'bfloat16'}), self.record)

    def test_settings_and_resolved_prompt_hashes_are_bound(self):
        for path in (self.settings_path, self.root/'prompt.txt', self.config_path):
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.guard()
            path.write_bytes(original)

    def test_changed_online_settings_are_rejected_before_model_loading(self):
        from model_information_qwen_v3.chat_engine import load_chat_settings
        settings = load_chat_settings(self.settings_path)
        for update in ({'max_prompt_tokens':512}, {'role_prompt':'changed'}, {'asr_enabled':True}):
            with self.subTest(update=update), \
                 patch.object(model, '_configuration', return_value=(self.config_path, self.config)), \
                 patch('model_information_qwen_v3.chat_engine.load_chat_settings', return_value={**settings, **update}), \
                 patch.object(model, 'load_engine', side_effect=AssertionError('must reject before GPU')):
                with self.assertRaisesRegex(ValueError, 'Chat settings'):
                    model.load_chat_engine(settings_path=self.settings_path)

    def test_new_default_config_never_falls_back_to_old_version(self):
        missing = self.root/'missing.json'
        with patch.object(model, 'DEFAULT_CONFIG_PATH', missing):
            with self.assertRaises(FileNotFoundError):
                model._configuration()

    def test_raw_engine_reuses_reply_without_calling_old_quality_constructor(self):
        class OldEngine:
            def __init__(self, config):
                raise AssertionError('old quality constructor cannot authorize V3')
            def reply(self, messages):
                return {'reply':'raw reply retained'}
        old = SimpleNamespace(PaperMemeEngine=OldEngine, PaperResponseIndex=Mock(return_value='index'))
        planner = SimpleNamespace(load_planner=Mock(return_value='planner'))
        generator = SimpleNamespace(PaperGenerator=Mock(return_value='generator'))
        modules = {'test_runtime.paper_engine':old, 'test_runtime.paper_planner':planner,
                   'test_runtime.paper_generator':generator}
        with patch.object(release, 'require_inference_release', return_value=self.record) as guard, \
             patch.object(model, '_runtime', return_value='test_runtime'), \
             patch.object(model.importlib, 'import_module', side_effect=modules.__getitem__), \
             patch('torch.set_num_threads'):
            raw = model._create_raw_engine(self.config, self.config_path, self.settings_path)
        guard.assert_called_once()
        self.assertEqual(raw.reply([]), {'reply':'raw reply retained'})
        self.assertEqual(raw.planner, 'planner')
        generator.PaperGenerator.assert_called_once_with(self.config, checkpoint=self.run/'generator/best.pt')

    def test_non_qwen_backbone_and_lfs_pointer_are_rejected(self):
        self.write(self.base/'config.json', dict(model_type='mistral'))
        with self.assertRaisesRegex(ValueError, 'Qwen3 backbone'):
            model._backbone(self.config)
        self.write(self.base/'config.json', dict(model_type='qwen3'))
        self.write(self.base/'model.safetensors', b'version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
            model._backbone(self.config)


if __name__ == '__main__':
    unittest.main()
