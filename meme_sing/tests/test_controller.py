"""Song routing never submits lyrics or local controls to the model server."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'meme_software/player/desktop'))
from conversation_turn import ConversationTurn
from meme_sing.controller import MusicController


class FakePlayer:
    def __init__(self):
        self.events = []
        self.starts = []
        self.stop_calls = 0
        self.closed = False
        self.playing = False
        self.current_token = None
    def start(self, track):
        token = 'token-' + str(len(self.starts) + 1)
        self.starts.append(track)
        self.playing = True
        self.current_token = token
        self.events.append({'type': 'started', 'token': token, 'track_id': track.id})
        return token
    def stop(self):
        self.stop_calls += 1
    def drain(self):
        events, self.events = self.events, []
        for event in events:
            if event.get('token') == self.current_token and event['type'] in ('finished', 'stopped', 'error'):
                self.playing = False
        return events
    def is_playing(self):
        return self.playing
    def close(self):
        self.closed = True


class FakeContext:
    def __init__(self):
        self.observed = []
        self.requests = []
    def choose(self, catalog, text):
        self.requests.append(text)
        return SimpleNamespace(track=SimpleNamespace(id='music-1', title='小猫之歌'))
    def observe_reply(self, *args, **kwargs):
        self.observed.append((args, kwargs))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.paused = True
        self.turn = ConversationTurn(lambda text: self.calls.append(('request', text)) or 'request-1',
                                     lambda: self.calls.append(('pause',)),
                                     lambda: self.calls.append(('resume',)),
                                     speech_start=lambda: self.calls.append(('speech_start',)),
                                     speech_cancel=lambda: self.calls.append(('speech_cancel',)),
                                     playback_done=lambda key, failed=False: self.calls.append(('done', key)))
        self.turn.enable()
        self.turn.set_ready(True)
        self.player, self.context = FakePlayer(), FakeContext()
        self.music = MusicController(self.turn, '/local/music',
            lambda enabled, token: self.calls.append(('control', enabled, token)),
            lambda: self.paused, player=self.player,
            catalog_factory=lambda directory: object(), context=self.context)

    def start_music(self):
        self.turn.speech_started()
        self.assertIsNone(self.music.handle_final('我想听一首歌'))
        self.music.poll()
        self.assertEqual('playing', self.music.state)
        return self.music.token

    def finish_stop(self, token):
        self.player.events.append({'type': 'stopped', 'token': token, 'track_id': 'music-1'})
        self.music.poll()

    def test_song_cancels_server_speech_lease_and_does_not_create_request(self):
        self.start_music()
        self.assertIn(('speech_cancel',), self.calls)
        self.assertFalse(any(call[0] in ('request', 'done') for call in self.calls))
        self.assertEqual('music', self.turn.phase)
        self.assertTrue(self.turn.listen_requested)
        self.assertFalse(self.turn.media_active)  # Meme output ownership is separate.

    def test_prepare_waits_for_microphone_closed_ack(self):
        self.paused = False
        self.music.handle_final('放首歌')
        self.music.poll()
        self.assertEqual([], self.player.starts)
        self.assertFalse(self.turn.listen_requested)
        self.paused = True
        self.music.poll()
        self.assertEqual(1, len(self.player.starts))
        self.assertTrue(self.turn.listen_requested)

    def test_gif_reply_or_manual_media_cannot_interrupt_music(self):
        self.start_music()
        self.assertFalse(self.turn.can_start_media())
        with self.assertRaises(ValueError):
            self.turn.respond({'request_id': 'desktop-other'}, lambda body: body, set())

    def test_music_speech_is_consumed_and_rearmed_without_server_lease(self):
        token = self.start_music()
        count = len(self.calls)
        self.turn.speech_started()
        self.music.music_utterance(token)
        self.music.handle_final('这是扬声器里的歌词')
        self.assertFalse(any(call[0] in ('request', 'speech_start') for call in self.calls[count:]))
        self.assertEqual(2, self.calls[count:].count(('resume',)))

    def test_stop_waits_for_owned_process_exit_before_normal_listening(self):
        token = self.start_music()
        self.music.music_stop(token)
        self.assertEqual('stopping', self.music.state)
        self.assertEqual('music_stopping', self.turn.phase)
        self.assertFalse(self.turn.listen_requested)
        self.assertFalse(self.turn.can_start_media())
        self.finish_stop(token)
        self.assertEqual('idle', self.music.state)
        self.assertEqual('cooldown', self.turn.phase)
        self.turn.tick(now=float('inf'))
        self.assertEqual('listening', self.turn.phase)
        self.assertFalse(any(call[0] == 'done' for call in self.calls))

    def test_stop_when_not_playing_is_normal_chat(self):
        self.assertEqual('request-1', self.music.handle_final('停止'))
        self.assertIn(('request', '停止'), self.calls)
        self.assertEqual('waiting', self.turn.phase)

    def test_local_api_play_uses_same_routing_and_stop_idle_never_becomes_chat(self):
        result = self.music.command({'action': 'stop'})
        self.assertEqual('idle', result['state'])
        self.assertFalse(any(call[0] == 'request' for call in self.calls))
        result = self.music.command({'action': 'play'})
        self.assertEqual('preparing', result['state'])
        self.music.poll()
        self.assertTrue(self.music.status()['playing'])
        self.assertEqual('stopping', self.music.command({'action': 'stop'})['state'])
        self.assertFalse(any(call[0] == 'request' for call in self.calls))

    def test_local_api_rejects_paths_files_invalid_intents_and_busy_state(self):
        for body in ({'action': 'play', 'path': '/tmp/song.mp3'},
                     {'action': 'play', 'track_id': 'music-1'},
                     {'action': 'delete'}, {'action': 'play', 'text': '普通聊天'},
                     {'action': 'stop', 'text': '停止'}, []):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.music.command(body)
        self.music.set_ready(False)
        with self.assertRaises(ValueError):
            self.music.command({'action': 'play'})
        self.music.set_ready(True)
        self.music.command({'action': 'play'})
        with self.assertRaises(ValueError):
            self.music.command({'action': 'play'})

    def test_exited_backend_before_poll_does_not_swallow_normal_stop(self):
        self.start_music()
        self.player.playing = False  # Physical playback ended before UI drains its event.
        self.assertFalse(self.music.status()['playing'])
        self.assertIsNone(self.music.handle_final('停止播放音乐'))
        self.assertEqual('停止播放音乐', self.music.pending_normal)
        self.assertEqual(0, self.player.stop_calls)
        self.turn.tick(now=float('inf'))
        self.music.poll()
        self.assertIn(('request', '停止播放音乐'), self.calls)
        self.assertEqual('waiting', self.turn.phase)
        self.assertEqual('停止播放音乐', self.music.pending_chat['request-1'])

    def test_partial_stop_after_physical_exit_never_stops_or_submits_chat(self):
        token = self.start_music()
        self.player.playing = False
        self.music.music_stop(token)
        self.assertEqual(0, self.player.stop_calls)
        self.assertEqual('idle', self.music.state)
        self.assertFalse(any(call[0] == 'request' for call in self.calls))

    def test_natural_end_and_error_both_restore_normal_turn_state(self):
        for kind in ('finished', 'error'):
            with self.subTest(kind=kind):
                token = self.start_music()
                self.player.events.append({'type': kind, 'token': token, 'message': '本地声音失败'})
                self.music.poll()
                self.assertEqual('idle', self.music.state)
                self.assertEqual('cooldown', self.turn.phase)
                self.turn.tick(now=float('inf'))
                self.assertTrue(self.turn.listen_requested)

    def test_old_playback_events_and_stop_do_not_affect_a_new_song(self):
        first = self.start_music()
        self.music.music_stop(first)
        self.finish_stop(first)
        self.turn.tick(now=float('inf'))
        second = self.start_music()
        self.assertNotEqual(first, second)
        self.player.events.extend([{'type': 'finished', 'token': first},
                                   {'type': 'error', 'token': first, 'message': 'old error'}])
        self.music.music_stop(first)
        self.music.music_utterance(first)
        self.music.poll()
        self.assertEqual(second, self.music.token)
        self.assertEqual('playing', self.music.state)

    def test_disconnect_stops_song_and_keeps_microphone_paused(self):
        token = self.start_music()
        self.music.set_ready(False)
        self.assertEqual(1, self.player.stop_calls)
        self.finish_stop(token)
        self.turn.tick(now=float('inf'))
        self.assertEqual('connecting', self.turn.phase)
        self.assertFalse(self.turn.listen_requested)

    def test_voice_disable_and_close_stop_owned_song(self):
        self.start_music()
        self.music.disable()
        self.assertFalse(self.turn.enabled)
        self.assertEqual(1, self.player.stop_calls)
        self.music.close()
        self.assertTrue(self.player.closed)
        self.assertIsNone(self.music.handle_final('放首歌'))

    def test_missing_library_reports_local_error_without_breaking_chat(self):
        self.music.catalog_factory = lambda directory: (_ for _ in ()).throw(ValueError('目录不存在'))
        self.music.handle_final('放首歌')
        events = self.music.poll()
        self.assertEqual('error', events[-1]['type'])
        self.assertIn('目录不存在', self.music.status()['last_error'])
        self.turn.tick(now=float('inf'))
        self.assertEqual('request-1', self.music.handle_final('你好'))

    def test_successful_reply_correlates_request_text_once(self):
        self.music.handle_final('今天非常难过')
        reply = {'id': 'request-1', 'conversation_id': 'conversation-1',
                 'reply_text': '我陪你', 'meme_id': 'meme-0001'}
        self.music.observe_reply(reply)
        self.music.observe_reply(reply)
        args, kwargs = self.context.observed[0]
        self.assertEqual(('conversation-1', '今天非常难过', '我陪你'), args)
        self.assertEqual('meme-0001', kwargs['meme_id'])
        self.assertEqual(1, len(self.context.observed))

    def test_cancel_during_prepare_ignores_late_started(self):
        self.music.handle_final('放首歌')
        # Start is normally followed by a queued 'started'; delay its processing.
        self.music.pending_song = None
        self.music.token = self.player.start(SimpleNamespace(id='music-1', title='歌'))
        self.music.track_title = '歌'
        self.music.set_ready(False)
        self.music.poll()
        self.assertEqual('stopping', self.music.state)
        self.assertFalse(any(call[0:2] == ('control', True) for call in self.calls))


if __name__ == '__main__':
    unittest.main()
