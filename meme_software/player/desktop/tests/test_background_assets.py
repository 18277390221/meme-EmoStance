"""Background variants keep server meme IDs and original audio unchanged."""
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import wave

from PIL import Image

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP.parent))
from directory_library import DirectoryAssets, DirectoryPlayer
from launch import PlayerError


def gif(color):
    stream = BytesIO()
    Image.new('RGB', (4, 3), color).save(stream, format='GIF', duration=120)
    return stream.getvalue()


def wav():
    stream = BytesIO()
    with wave.open(stream, 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b'\0\0' * 1600)
    return stream.getvalue()


class BackgroundAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / 'sucai_meme/meme_gif'
        self.background = self.original.parent / 'meme_gif_background'
        self.original.mkdir(parents=True)
        self.background.mkdir()
        self.original_bytes = gif('red')
        self.audio_bytes = wav()
        self.stems = {'meme-0001': 'meme-0001-猫1号', 'meme-0002': 'meme-0002-狗-10'}
        for stem in self.stems.values():
            (self.original / (stem + '.gif')).write_bytes(self.original_bytes)
            (self.original / (stem + '.wav')).write_bytes(self.audio_bytes)
            for number in range(1, 11):
                (self.background / (stem + '-' + str(number) + '.gif')).write_bytes(
                    gif((number * 20, 40, 80)))

    def assets(self, choice=None):
        return DirectoryAssets(self.original, choice=choice or (lambda values: 1))

    def test_suffix_one_and_ten_keep_exact_meme_id_and_original_audio(self):
        assets = self.assets()
        self.assertEqual(tuple(range(1, 11)), assets.available_background_ids)
        for number in (1, 10):
            with self.subTest(background_id=number):
                row = assets.read('meme-0001', background_id=number)
                self.assertEqual('meme-0001', row['id'])
                self.assertEqual('猫1号', row['label'])
                self.assertEqual(number, row['background_id'])
                self.assertEqual(self.stems['meme-0001'] + '-' + str(number) + '.gif', row['visual_name'])
                self.assertEqual(self.stems['meme-0001'] + '.wav', row['audio_name'])
                self.assertEqual(self.audio_bytes, row['audio_blob'])
                self.assertNotEqual(self.original_bytes, row['visual_blob'])
                self.assertEqual(self.background, Path(row['source_path']).parent)

    def test_one_choice_per_conversation_and_fixed_across_meme_ids(self):
        calls = []
        choices = iter((1, 4, 10))
        def choice(values):
            calls.append(values)
            return next(choices)
        assets = self.assets(choice)
        self.assertEqual(1, assets.background_id)  # Initial/manual display.
        self.assertEqual(4, assets.begin_conversation('dialogue-1'))
        self.assertEqual(4, assets.begin_conversation('dialogue-1'))
        self.assertEqual(2, len(calls))
        self.assertEqual([4, 4], [assets.read(key)['background_id'] for key in self.stems])
        self.assertEqual(10, assets.begin_conversation('dialogue-2'))
        self.assertEqual(3, len(calls))
        self.assertEqual('dialogue-2', assets.conversation_id)

    def test_explicit_event_background_survives_new_conversation(self):
        selections = iter((1, 4, 10))
        assets = self.assets(lambda values: next(selections))
        original_event_background = assets.begin_conversation('first')
        assets.begin_conversation('second')
        self.assertEqual(10, assets.read('meme-0001')['background_id'])
        self.assertEqual(4, assets.read('meme-0001', original_event_background)['background_id'])

    def test_complete_backgrounds_only_are_eligible(self):
        (self.background / (self.stems['meme-0002'] + '-3.gif')).unlink()
        assets = self.assets()
        self.assertNotIn(3, assets.available_background_ids)
        with self.assertRaisesRegex(PlayerError, '缺少部分表情'):
            assets.read('meme-0001', 3)

    def test_deleted_current_variant_fails_without_switching_background(self):
        assets = self.assets()
        assets.begin_conversation('dialogue')
        (self.background / (self.stems['meme-0002'] + '-1.gif')).unlink()
        with self.assertRaisesRegex(PlayerError, '缺少部分表情'):
            assets.read('meme-0001')
        self.assertEqual(1, assets.background_id)
        with self.assertRaises(PlayerError):
            assets.begin_conversation('dialogue')
        self.assertEqual(1, assets.background_id)

    def test_missing_directory_never_uses_original_gif(self):
        for path in self.background.iterdir():
            path.unlink()
        self.background.rmdir()
        with self.assertRaisesRegex(PlayerError, '不会回退'):
            self.assets()

    def test_no_complete_background_fails_even_if_original_gifs_exist(self):
        for path in self.background.iterdir():
            path.unlink()
        with self.assertRaisesRegex(PlayerError, '不会回退'):
            self.assets()

    def test_duplicate_id_and_background_are_rejected(self):
        duplicate = self.background / 'meme-0001-z-1.gif'
        duplicate.write_bytes(gif('blue'))
        with self.assertRaisesRegex(PlayerError, '重复文件'):
            self.assets()

    def test_original_stem_mismatch_cannot_select_an_unrelated_image(self):
        (self.background / (self.stems['meme-0001'] + '-1.gif')).unlink()
        (self.background / 'meme-0001-unrelated-1.gif').write_bytes(gif('blue'))
        with self.assertRaisesRegex(PlayerError, '名称与原表情不对应'):
            self.assets()

    def test_symlink_variant_is_rejected(self):
        path = self.background / (self.stems['meme-0001'] + '-1.gif')
        path.unlink()
        path.symlink_to(self.original / (self.stems['meme-0001'] + '.gif'))
        with self.assertRaisesRegex(PlayerError, '符号链接'):
            self.assets()

    def test_symlink_audio_is_rejected(self):
        path = self.original / (self.stems['meme-0001'] + '.wav')
        path.unlink()
        path.symlink_to(self.original / (self.stems['meme-0002'] + '.wav'))
        with self.assertRaisesRegex(PlayerError, '符号链接'):
            self.assets()

    def test_original_duplicate_id_is_rejected(self):
        (self.original / 'meme-0001-other.gif').write_bytes(self.original_bytes)
        with self.assertRaisesRegex(PlayerError, '编号重复'):
            self.assets()

    def test_excluded_original_and_background_remain_excluded(self):
        (self.original / 'meme-0056-excluded.gif').write_bytes(self.original_bytes)
        for number in range(1, 11):
            (self.background / ('meme-0056-excluded-' + str(number) + '.gif')).write_bytes(gif('blue'))
        assets = self.assets()
        self.assertEqual(set(self.stems), {row['id'] for row in assets.assets()})
        with self.assertRaisesRegex(PlayerError, '已剔除'):
            assets.read('meme-0056')

    def test_invalid_background_id_is_rejected(self):
        assets = self.assets()
        for number in (0, 11, '1', True):
            with self.subTest(background_id=number), self.assertRaises(PlayerError):
                assets.read('meme-0001', number)

    def test_metadata_points_to_background_and_keeps_original_label(self):
        assets = self.assets(lambda values: 10)
        rows = assets.assets()
        self.assertEqual(set(self.stems), {row['id'] for row in rows})
        self.assertTrue(all(row['background_id'] == 10 for row in rows))
        self.assertTrue(all(Path(row['source_path']).parent == self.background for row in rows))
        self.assertEqual('狗-10', rows[1]['label'])
        self.assertEqual(Path(rows[0]['source_path']).stat().st_mtime_ns, rows[0]['updated_at'])

    def test_background_selection_may_repeat_on_a_new_conversation(self):
        assets = self.assets()
        self.assertEqual(1, assets.begin_conversation('first'))
        self.assertEqual(1, assets.begin_conversation('second'))

    def test_background_revision_changes_without_changing_conversation(self):
        candidates = []
        def choose(values):
            candidates.append(tuple(values))
            return min(values)
        assets = self.assets(choose)
        first = assets.begin_conversation('dialogue', 0)
        second = assets.begin_conversation('dialogue', 1)
        self.assertNotEqual(first, second)
        self.assertNotIn(first, candidates[-1])
        self.assertEqual(('dialogue',1), (assets.conversation_id,assets.background_revision))
        self.assertEqual(second, assets.begin_conversation('dialogue', 1))
        self.assertEqual(first, assets.begin_conversation('dialogue', 0))
        self.assertEqual(second, assets.background_id)
        self.assertEqual(1, assets.background_revision)
        self.assertEqual(3, len(candidates))

    def test_unknown_old_revision_cannot_change_active_background(self):
        assets = self.assets(min)
        selected = assets.begin_conversation('dialogue', 3)
        with self.assertRaisesRegex(PlayerError, '过期'):
            assets.begin_conversation('dialogue', 2)
        self.assertEqual((selected,3), (assets.background_id,assets.background_revision))

    def test_only_one_complete_background_can_remain_on_revision_change(self):
        for path in self.background.iterdir():
            if not path.name.endswith('-1.gif'):
                path.unlink()
        assets = self.assets(min)
        self.assertEqual(1, assets.begin_conversation('dialogue', 0))
        self.assertEqual(1, assets.begin_conversation('dialogue', 1))
        self.assertEqual(1, assets.background_revision)

    def test_invalid_background_revisions_are_rejected_before_selection(self):
        assets = self.assets(min)
        assets.begin_conversation('dialogue', 1)
        original = assets.background_id
        for value in (None, True, False, -1, 1.0, '1'):
            with self.subTest(value=value), self.assertRaises(PlayerError):
                assets.begin_conversation('other', value)
            self.assertEqual((original,'dialogue',1),
                             (assets.background_id,assets.conversation_id,assets.background_revision))

    def test_unchanged_directories_do_not_rescan_all_variants(self):
        assets = self.assets()
        with mock.patch.object(Path, 'iterdir', side_effect=AssertionError('Unnecessary index scan')):
            assets.begin_conversation('dialogue')
            self.assertEqual(2, len(assets.assets()))
            self.assertEqual(1, assets.read('meme-0001')['background_id'])

    def test_variant_bytes_are_not_cached_with_the_index(self):
        assets = self.assets()
        path = self.background / (self.stems['meme-0001'] + '-1.gif')
        replacement = gif('yellow')
        path.write_bytes(replacement)
        self.assertEqual(replacement, assets.read('meme-0001')['visual_blob'])

    def test_directory_player_exposes_background_snapshots_and_original_ids(self):
        player_root = self.root / 'player'
        (player_root / 'website').mkdir(parents=True)
        (player_root / 'website/library.json').write_text(json.dumps({'items': [], 'scenarios': []}))
        player = DirectoryPlayer(player_root, self.original, choice=lambda values: 10)
        self.assertEqual(10, player.background_id)
        self.assertEqual(10, player.begin_conversation('dialogue'))
        self.assertEqual('dialogue', player.conversation_id)
        self.assertEqual(1, player.read_native_asset('meme-0001', background_id=1)['background_id'])
        self.assertEqual(set(self.stems), {row['id'] for row in player.assets()})
        visual_type, visual_blob = player.media('meme-0001', 'visual')
        self.assertEqual('image/gif', visual_type)
        self.assertNotEqual(self.original_bytes, visual_blob)
        audio_type, audio_blob = player.media('meme-0001', 'audio')
        self.assertEqual(('audio/wav', self.audio_bytes), (audio_type, audio_blob))


if __name__ == '__main__':
    unittest.main()
