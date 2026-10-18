"""Safe filename-based catalogue metadata using the delivered music names."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

from meme_sing.catalog import MusicCatalog, filename_metadata


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / 'music'
        self.directory.mkdir()

    def music(self, filename):
        path = self.directory / filename
        path.write_bytes(b'ID3localmusic')
        return path

    def test_actual_weighted_filename_and_original_path_are_preserved(self):
        name = 'cut_akulamatata - 米的随波逐流_浪漫5，悲伤2，欢乐3_5.mp3'
        path = self.music(name)
        catalog = MusicCatalog(self.directory)
        self.assertIsInstance(catalog.tracks, tuple)
        track = catalog.tracks[0]
        self.assertEqual(path, track.path)
        self.assertEqual(name, track.filename)
        self.assertEqual('cut_akulamatata - 米的随波逐流', track.title)
        self.assertEqual({'浪漫': .5, '悲伤': .2, '欢乐': .3}, track.tags)
        self.assertIs(track, catalog.resolve(track.id))
        with self.assertRaises(FrozenInstanceError):
            track.id = 'changed'

    def test_amnesia_missing_underscore_extracts_only_emotion_tail(self):
        name = 'cut_是只大鱼鸭 - Amnesia哈基米（完整版）沉思6，悲伤3，欣喜1_10.mp3'
        self.music(name)
        track = MusicCatalog(self.directory).tracks[0]
        self.assertEqual('cut_是只大鱼鸭 - Amnesia哈基米（完整版）', track.title)
        self.assertEqual({'沉思': .6, '悲伤': .3, '欣喜': .1}, track.tags)

    def test_unweighted_labels_have_equal_weights(self):
        self.assertEqual(('akulamatata - Color-X', {'欢快': .5, '轻松': .5}),
                         filename_metadata('akulamatata - Color-X_欢快，轻松_6.mp3'))
        self.assertEqual(('歌曲', {'伤心': 1.}), filename_metadata('歌曲_伤心_7.mp3'))

    def test_mixed_weights_and_duplicate_labels_are_normalized(self):
        title, tags = filename_metadata('歌曲_开心3,轻松、开心_6.mp3')
        self.assertEqual('歌曲', title)
        self.assertEqual({'开心': .8, '轻松': .2}, tags)

    def test_repeated_trailing_number_is_not_track_identity(self):
        self.music('第一首_轻松，休闲_6.mp3')
        self.music('第二首_轻松，休闲_6.mp3')
        first, second = MusicCatalog(self.directory).tracks
        self.assertNotEqual(first.id, second.id)
        self.assertRegex(first.id, r'^music-[a-f0-9]{24}$')
        again = MusicCatalog(self.directory)
        self.assertEqual({row.filename: row.id for row in again.tracks},
                         {row.filename: row.id for row in (first, second)})

    def test_tagless_filename_remains_a_title(self):
        self.assertEqual(('My Song', {}), filename_metadata('My Song.mp3'))
        self.assertEqual(('一首歌曲', {}), filename_metadata('一首歌曲.mp3'))

    def test_metadata_and_unsupported_files_do_not_become_tracks(self):
        self.music('歌曲_轻松_6.mp3')
        (self.directory / '.gitkeep').write_bytes(b'\n')
        (self.directory / 'sucai_meme.lnk').write_bytes(b'windows shortcut')
        (self.directory / 'cover.png').write_bytes(b'png')
        self.assertEqual(1, len(MusicCatalog(self.directory).tracks))

    def test_supported_regular_audio_formats(self):
        for suffix in ('MP3', 'wav', 'ogg', 'flac'):
            self.music('歌曲_开心_1.' + suffix)
        self.assertEqual(4, len(MusicCatalog(self.directory).tracks))

    def test_missing_empty_and_not_a_directory_are_rejected(self):
        for directory in (self.root / 'missing', self.directory):
            with self.subTest(directory=directory), self.assertRaises(ValueError):
                MusicCatalog(directory)
        plain = self.root / 'file'
        plain.write_bytes(b'file')
        with self.assertRaises(ValueError):
            MusicCatalog(plain)

    def test_empty_audio_file_is_rejected(self):
        (self.directory / '歌曲_开心_1.mp3').touch()
        with self.assertRaisesRegex(ValueError, '非空'):
            MusicCatalog(self.directory)

    def test_audio_named_directory_is_rejected(self):
        (self.directory / 'folder.mp3').mkdir()
        with self.assertRaises(ValueError):
            MusicCatalog(self.directory)

    def test_symlink_file_and_catalogue_are_rejected(self):
        outside = self.root / 'outside.mp3'
        outside.write_bytes(b'ID3outside')
        (self.directory / '歌曲_轻松_6.mp3').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            MusicCatalog(self.directory)
        shortcut = self.root / 'shortcut'
        shortcut.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, '符号链接'):
            MusicCatalog(shortcut)

    def test_resolve_revalidates_current_file_and_rejects_unknown_id(self):
        path = self.music('歌曲_轻松_6.mp3')
        catalog = MusicCatalog(self.directory)
        for identity in ('../outside.mp3', None, [], 'music-doesnotexist'):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                catalog.resolve(identity)
        path.unlink()
        with self.assertRaises(ValueError):
            catalog.resolve(catalog.tracks[0].id)

    def test_zero_weight_metadata_is_rejected(self):
        self.music('歌曲_轻松0，快乐0_6.mp3')
        with self.assertRaisesRegex(ValueError, '全部为零'):
            MusicCatalog(self.directory)


if __name__ == '__main__':
    unittest.main()
