"""Wake-only capture owns no model turn and wakes only after screen completion."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_turn import ConversationTurn


class StandbyTurnTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.turn = ConversationTurn(lambda text: self.calls.append(('send', text)) or 'request',
            lambda: self.calls.append(('pause',)), lambda: self.calls.append(('resume',)),
            speech_start=lambda: self.calls.append(('lease',)),
            speech_cancel=lambda: self.calls.append(('cancel',)),
            playback_done=lambda *a, **kw: self.calls.append(('done',)))
        self.turn.set_ready(True)
        self.turn.enable()
        self.calls.clear()

    def enter(self):
        self.assertTrue(self.turn.standby_prepare())
        self.assertTrue(self.turn.standby_started())

    def test_sleep_cancels_speech_lease_without_model_request(self):
        self.turn.speech_started()
        self.assertTrue(self.turn.standby_prepare())
        self.assertEqual([('lease',), ('cancel',), ('pause',)], self.calls)
        self.assertEqual('standby_preparing', self.turn.phase)

    def test_standby_accepts_control_capture_but_never_chat_or_media(self):
        self.enter()
        self.assertEqual('standby', self.turn.phase)
        self.assertEqual(('resume',), self.calls[-1])
        self.assertIsNone(self.turn.submit_final('睡觉声音或其他话'))
        self.assertFalse(self.turn.can_start_media())
        self.assertFalse(self.turn.music_prepare())
        with self.assertRaises(ValueError):
            self.turn.respond({'request_id': 'desktop-new'}, lambda body: body, {})
        self.assertFalse(any(call[0] in ('send', 'done') for call in self.calls))

    def test_sleep_cannot_start_while_waiting_for_reply(self):
        self.turn.submit_final('正常聊天')
        self.assertFalse(self.turn.standby_prepare())
        self.assertEqual('waiting', self.turn.phase)

    def test_wake_blocks_capture_until_complete_then_cooldown(self):
        self.enter()
        self.assertTrue(self.turn.standby_waking())
        resumes = self.calls.count(('resume',))
        self.turn.enable()
        self.turn.tick(now=1000)
        self.assertEqual('standby_waking', self.turn.phase)
        self.assertEqual(resumes, self.calls.count(('resume',)))
        self.turn.standby_finished()
        deadline = self.turn.resume_at
        self.turn.tick(now=deadline - .01)
        self.assertEqual(resumes, self.calls.count(('resume',)))
        self.turn.tick(now=deadline)
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual(resumes + 1, self.calls.count(('resume',)))

    def test_abort_releases_mode_without_automatically_opening_microphone(self):
        self.enter()
        resumes = self.calls.count(('resume',))
        self.turn.standby_abort()
        self.turn.tick(now=1000)
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual(resumes, self.calls.count(('resume',)))
        self.turn.enable()
        self.assertEqual(resumes + 1, self.calls.count(('resume',)))

    def test_disable_and_disconnect_preserve_mode_until_abort(self):
        self.enter()
        self.turn.disable()
        self.turn.set_ready(False)
        self.assertEqual('standby', self.turn.phase)
        self.assertFalse(self.turn.standby_started())
        self.turn.standby_abort()
        self.assertEqual('idle', self.turn.phase)
        self.assertIsNone(self.turn.resume_at)

    def test_wake_request_while_awake_cannot_start_wake_handover(self):
        self.assertFalse(self.turn.standby_waking())
        self.turn.standby_finished()
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual([], self.calls)

    def test_rearm_control_window_does_not_open_a_recording_lease(self):
        self.enter()
        self.turn.standby_rearm()
        self.assertEqual(2, self.calls.count(('resume',)))
        self.turn.speech_started()
        self.assertNotIn(('lease',), self.calls)


if __name__ == '__main__':
    unittest.main()
