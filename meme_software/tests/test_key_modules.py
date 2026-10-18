"""Pure logic checks for the published Pi snapshot; no mic/network/media."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'player/desktop'), str(ROOT / 'player')]
from conversation_turn import ConversationTurn
from meme_huanxing.commands import classify_command
from meme_sing.core import is_music_request, is_stop_request
from meme_sing.catalog import filename_metadata
from server_link import ServerLink, parse_reply
from voice_recovery import VoiceRecovery, configured_capture_available


class KeyModules(unittest.TestCase):
    def test_addressed_sleep_wake_and_ordinary_speech(self):
        self.assertEqual(classify_command('小me，去睡觉吧'), 'sleep')
        self.assertEqual(classify_command('小me，别睡了'), 'wake')
        for text in ['我想睡觉', '睡觉吧', '小me不要去睡觉', '如果小me醒醒会怎样']:
            self.assertIsNone(classify_command(text))

    def test_music_controls_reject_negation(self):
        self.assertTrue(is_music_request('给我放首开心的歌'))
        self.assertFalse(is_music_request('我不想听音乐'))
        self.assertTrue(is_stop_request('停止音乐'))
        self.assertFalse(is_stop_request('不要停止音乐'))

    def test_music_filename_weight_normalization(self):
        title, tags = filename_metadata('示例歌曲_浪漫5，愉悦5_01.mp3')
        self.assertEqual(title, '示例歌曲')
        self.assertEqual(tags, {'浪漫': .5, '愉悦': .5})

    def turn(self):
        actions = []
        sent = []
        def send(text):
            sent.append(text)
            return 'fixture-request'
        turn = ConversationTurn(send, lambda: actions.append('pause'), lambda: actions.append('resume'))
        return turn, actions, sent

    def test_readiness_and_single_final_submission(self):
        turn, actions, sent = self.turn()
        turn.enable()
        self.assertNotIn('resume', actions)
        turn.set_ready(True)
        self.assertIn('resume', actions)
        self.assertEqual(turn.submit_final('合成完整句'), 'fixture-request')
        self.assertIsNone(turn.submit_final('重复回调'))
        self.assertEqual(sent, ['合成完整句'])

    def test_standby_transition_is_local_and_wake_pauses(self):
        turn, actions, sent = self.turn()
        turn.set_ready(True)
        turn.enable()
        self.assertTrue(turn.standby_prepare())
        self.assertTrue(turn.standby_started())
        self.assertEqual(turn.phase, 'standby')
        self.assertIsNone(turn.submit_final('待机期间普通话语'))
        self.assertTrue(turn.standby_waking())
        self.assertEqual(actions[-1], 'pause')
        self.assertEqual(sent, [])

    def test_protocol2_request_requires_ready_and_contains_end_marker(self):
        link = ServerLink()
        self.assertIsNone(link.submit('合成请求'))
        link.connected = link.model_ready = True
        key = link.submit('合成请求')
        frame = link.outgoing.get_nowait()
        self.assertEqual(frame, {'type': 'request', 'id': key, 'text': '合成请求', 'end': True})
        self.assertIsNone(link.submit('第二个未确认请求'))
        self.assertEqual(link.outgoing.qsize(), 0)

    def test_reply_validation_rejects_echo_and_invalid_id(self):
        self.assertIsNone(parse_reply({'text': '普通回显'}))
        with self.assertRaises(ValueError):
            parse_reply({'meme_id': '../bad', 'reply_text': '合成回复'})
        self.assertEqual(parse_reply({'meme_id': 'meme-0004', 'reply_text': '合成回复'}),
                         {'meme_id': 'meme-0004', 'reply_text': '合成回复'})

    def test_recovery_cancel_invalidates_pending_restart(self):
        now = [0.]
        recovery = VoiceRecovery(clock=lambda: now[0])
        self.assertTrue(recovery.fail({'reason': 'audio_device_unavailable', 'recoverable': True}))
        now[0] = 2.
        ticket = recovery.poll(device_available=None)
        self.assertIsNotNone(ticket)
        recovery.cancel()
        self.assertFalse(recovery.attempt_started(ticket))

    def test_absent_device_does_not_spend_restart_budget(self):
        now = [0.]
        recovery = VoiceRecovery(clock=lambda: now[0])
        recovery.fail({'reason': 'audio_device_unavailable', 'recoverable': True})
        for seconds in [2., 10., 100.]:
            now[0] = seconds
            self.assertIsNone(recovery.poll(device_available=False))
        self.assertEqual(recovery.attempts, 0)

    def test_stable_alsa_name_resolves_new_card_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'proc/asound').mkdir(parents=True)
            (root / 'dev/snd').mkdir(parents=True)
            (root / 'proc/asound/cards').write_text(' 4 [UACDemoV10   ]: USB-Audio\n')
            (root / 'dev/snd/pcmC4D0c').touch()
            self.assertTrue(configured_capture_available('plughw:CARD=UACDemoV10,DEV=0',
                                                         root / 'proc', root / 'dev'))


if __name__ == '__main__':
    unittest.main()
