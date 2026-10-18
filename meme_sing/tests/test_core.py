"""Local intents, confirmed dialogue boundaries and weighted music matching."""
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from meme_sing.core import (CONTEXT_TTL_SECONDS, MusicContext, StopCommandDetector,
                            is_music_request, is_stop_request)


@dataclass(frozen=True)
class Track:
    id: str
    tags: dict
    title: str = "test"
    filename: str = "test.mp3"
    path: Path = Path("test.mp3")


class IntentTests(unittest.TestCase):
    def test_actual_music_requests(self):
        for text in ("我想听音乐", "来首歌", "放一首轻松的歌", "能给我播放歌曲吗", "唱首歌吧", "听歌", "来点音乐",
                     "不要悲伤的歌，放轻松的音乐", "我不想听伤感的歌，来点轻松的音乐"):
            with self.subTest(text=text):
                self.assertTrue(is_music_request(text))

    def test_music_discussion_negation_and_quoted_commands_are_not_requests(self):
        for text in ("我不想听音乐", "不要放歌", "别唱歌", "你喜欢听歌吗", "音乐是什么", "他说想听音乐", "如果我想听音乐再放", '请读出“放首歌”', "你好", None):
            with self.subTest(text=text):
                self.assertFalse(is_music_request(text))

    def test_stop_commands_and_partial_suffix(self):
        for text in ("停止", "停", "暂停", "别放了", "不听了", "请停止播放音乐", "把音乐停了", "停一下吧",
                     "停下", "停下吧", "关音乐", "关闭音乐", "关掉音乐", "不要放歌了", "不用放了", "不想听了",
                     "我想停止音乐", "Meme请先暂停", "我不想听了", "停止播放音乐", "停止播放歌曲",
                     "暂停一下", "停止这首歌", "音乐停一下", "把歌关了", "别唱了"):
            with self.subTest(text=text):
                self.assertTrue(is_stop_request(text))
                self.assertFalse(is_music_request(text), "A stop sentence must never start a song while idle")
        self.assertFalse(is_stop_request("一些前面的识别内容请停止"))
        self.assertTrue(is_stop_request("一些前面的识别内容请停止", partial=True))

    def test_negation_mentions_and_conditions_do_not_stop(self):
        for text in ("不要停止", "别停", "别停止音乐", "不要暂停", "不想暂停", "不要停止播放音乐", "我没有说停止", "他说停止", "他说停", "停止是什么意思", '歌曲叫“停止”', "如果我说停止", "等唱完再停止", "继续播放", "音乐已经停止了", "歌曲停止了", None):
            with self.subTest(text=text):
                self.assertFalse(is_stop_request(text))
                self.assertFalse(is_stop_request(text, partial=True))


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.
        self.context = MusicContext(clock=lambda: self.now)
        self.sad = Track("sad", {"悲伤": 1.})
        self.happy = Track("happy", {"欢乐": 1.})
        self.calm = Track("calm", {"轻松": 1.})
        self.catalog = SimpleNamespace(tracks=(self.sad, self.happy, self.calm))

    def test_successful_user_evidence_beats_assistant_and_explicit_style_overrides(self):
        self.context.observe_reply("a", "我今天很难过", "我们开心一点吧", "meme-0008", ("欢乐",))
        first = self.context.choose(self.catalog, "想听音乐")
        self.assertIs(first.track, self.sad)
        self.assertGreater(first.moods["悲伤"], first.moods["欢乐"])
        self.assertIs(self.context.choose(self.catalog, "放一首轻松的歌").track, self.calm)
        self.assertIn("本地规则", first.reason)

    def test_unhappy_is_sad_but_negated_sadness_is_not_evidence(self):
        self.context.observe_reply("a", "我不开心", "我在这里")
        self.assertIs(self.context.choose(self.catalog, "来首歌").track, self.sad)
        self.context.observe_reply("b", "我不是难过", "知道了")
        result = self.context.choose(self.catalog, "来首歌")
        self.assertNotIn("悲伤", result.moods)
        self.assertIs(result.track, self.calm)

    def test_new_true_conversation_clears_old_mood_and_time_only_expires_evidence(self):
        self.context.observe_reply("old", "很伤心", "陪着你")
        self.context.observe_reply("new", "今天好开心", "很好")
        self.assertIs(self.context.choose(self.catalog, "来首歌").track, self.happy)
        self.now = CONTEXT_TTL_SECONDS
        result = self.context.choose(self.catalog, "来首歌")
        self.assertEqual(self.context.conversation_id, "new")
        self.assertEqual(result.moods, {"中性": 1.})
        self.assertIs(result.track, self.calm)

    def test_eight_pair_limit_forgets_older_emotion_and_controls_do_not_count(self):
        self.context.observe_reply("a", "我很悲伤", "嗯")
        for _ in range(8):
            self.context.observe_reply("a", "你好", "在的")
        self.context.observe_reply("a", "想听音乐", "悲伤的音乐")
        self.context.observe_reply("a", "停止", "很悲伤")
        result = self.context.choose(self.catalog, "来首歌")
        self.assertEqual(result.moods, {"中性": 1.})

    def test_catalog_tags_weighted_families_and_meme_tag_hint(self):
        nostalgic = Track("nostalgic", {"怀念": .7, "悲伤": .3})
        mixed = Track("mixed", {"怀念": .2, "悲伤": .8})
        self.context.observe_reply("a", "你好", "嗯", meme_tags={"怀旧": 1.})
        result = self.context.choose(SimpleNamespace(tracks=(mixed, nostalgic)), "来点音乐")
        self.assertIs(result.track, nostalgic)
        self.assertEqual(result.moods, {"怀念": 1.})

    def test_negative_requested_style_does_not_override_positive_style(self):
        self.assertIs(self.context.choose(self.catalog, "不要悲伤的歌，放轻松的音乐").track, self.calm)
        self.assertIs(self.context.choose(self.catalog, "来一首伤感的歌").track, self.sad)

    def test_avoid_repeating_when_alternative_exists_but_allow_single_track(self):
        pair = SimpleNamespace(tracks=(Track("one", {"愉悦": 1.}), Track("two", {"开心": 1.})))
        with patch("meme_sing.core.random.choice", side_effect=lambda values: values[0]):
            self.assertEqual([self.context.choose(pair, "欢乐的歌").track.id for _ in range(3)], ["one", "two", "one"])
        single = SimpleNamespace(tracks=(self.sad,))
        self.assertIs(self.context.choose(single, "来首歌").track, self.sad)
        self.assertIs(self.context.choose(single, "来首歌").track, self.sad)

    def test_empty_catalog_and_invalid_limits_fail_clearly(self):
        with self.assertRaises(ValueError):
            self.context.choose(SimpleNamespace(tracks=()), "来首歌")
        for value in (0, True, 1.5):
            with self.assertRaises(ValueError):
                MusicContext(max_turns=value)

    def test_real_catalog_filename_weights_work_without_model_dependencies(self):
        from meme_sing.catalog import MusicCatalog
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "song_悲伤7，怀念3_7.mp3").write_bytes(b"local-test-mp3")
            (root / "another_轻松，愉悦_6.mp3").write_bytes(b"local-test-mp3")
            catalog = MusicCatalog(root)
            self.context.observe_reply("a", "我很难过", "陪着你")
            result = self.context.choose(catalog, "想听音乐")
            self.assertEqual(result.track.filename, "song_悲伤7，怀念3_7.mp3")
            self.assertEqual(result.track.tags, {"悲伤": .7, "怀念": .3})


class StopDetectorTests(unittest.TestCase):
    def test_two_spaced_hits_fire_once_and_reset_allows_next_playback(self):
        detector = StopCommandDetector()
        self.assertFalse(detector.feed("停止", now=0))
        self.assertFalse(detector.feed("停止", now=.1))
        self.assertTrue(detector.feed("停止", now=.2))
        self.assertFalse(detector.feed("停止", now=.4))
        detector.reset()
        self.assertFalse(detector.feed("暂停", now=.5))
        self.assertTrue(detector.feed("暂停", now=.7))

    def test_intervening_non_command_or_large_gap_breaks_stability(self):
        detector = StopCommandDetector()
        self.assertFalse(detector.feed("停止", now=0))
        self.assertFalse(detector.feed("不要停止", now=.1))
        self.assertFalse(detector.feed("停止", now=.2))
        self.assertTrue(detector.feed("停止", now=.4))
        detector.reset()
        self.assertFalse(detector.feed("歌词停止", now=1))
        self.assertFalse(detector.feed("请停止", now=2))
        self.assertFalse(detector.feed("请停止", now=4))
        self.assertTrue(detector.feed("请停止", now=4.2))


if __name__ == "__main__":
    unittest.main()
