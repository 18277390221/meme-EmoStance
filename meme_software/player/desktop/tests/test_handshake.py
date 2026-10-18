"""Headless protocol and turn regressions; no model, sound device or display needed."""
from pathlib import Path
import sys
import unittest

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
from conversation_turn import ConversationTurn
from server_link import ServerLink, parse_reply


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.turn = ConversationTurn(
            lambda text: self.calls.append(('send', text)) or 'req-1',
            lambda: self.calls.append(('pause',)),
            lambda: self.calls.append(('resume',)), cooldown=.25,
            speech_start=lambda: self.calls.append(('speech_start',)),
            speech_cancel=lambda: self.calls.append(('speech_cancel',)),
            playback_done=lambda key, failed=False: self.calls.append(('done', key, failed)))

    def start(self):
        self.turn.enable()
        self.turn.set_ready(True)

    def test_enable_waits_for_ready(self):
        self.turn.enable()
        self.assertEqual('connecting', self.turn.phase)
        self.assertNotIn(('resume',), self.calls)
        self.turn.set_ready(True)
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual([('resume',)], self.calls)

    def test_readiness_before_enable_does_not_open_mic(self):
        self.turn.set_ready(True)
        self.assertEqual([], self.calls)
        self.turn.enable()
        self.assertEqual([('resume',)], self.calls)

    def test_disconnect_pauses_but_preserves_pending_id(self):
        self.start()
        self.turn.submit_final('你好')
        self.turn.set_ready(False)
        self.assertEqual('req-1', self.turn.request_id)
        self.assertEqual('waiting', self.turn.phase)
        self.assertEqual(('pause',), self.calls[-1])
        count = self.calls.count(('resume',))
        self.turn.set_ready(True)
        self.assertEqual(count, self.calls.count(('resume',)))

    def test_pause_before_send_and_one_request(self):
        self.start()
        self.turn.submit_final('你好')
        self.assertEqual([('pause',), ('send', '你好')], self.calls[-2:])
        self.assertIsNone(self.turn.submit_final('再说一句'))
        self.assertEqual(1, sum(c[0] == 'send' for c in self.calls))

    def test_actual_speech_start_only_once_no_start_on_resume(self):
        self.start()
        self.assertNotIn(('speech_start',), self.calls)
        self.turn.speech_started()
        self.turn.speech_started()
        self.assertEqual(1, self.calls.count(('speech_start',)))

    def test_cancel_releases_recording_and_resumes(self):
        self.start()
        self.turn.speech_started()
        self.turn.speech_cancelled()
        self.assertIn(('speech_cancel',), self.calls)
        self.assertEqual(('resume',), self.calls[-1])

    def test_wait_prepare_play_all_keep_microphone_closed(self):
        self.start()
        self.turn.submit_final('你好')
        self.turn.respond({'request_id': 'server-req-1'}, lambda body: body, set())
        self.assertEqual('req-1', self.turn.request_id)
        self.assertEqual('preparing', self.turn.phase)
        self.turn.media_started()
        self.assertEqual('playing', self.turn.phase)
        self.assertFalse(self.turn.listen_requested)
        self.assertIsNone(self.turn.submit_final('扬声器回声'))
        self.assertEqual(1, self.calls.count(('resume',)))
        self.turn.media_finished(now=10)
        self.assertIn(('done', 'req-1', False), self.calls)
        self.assertEqual('cooldown', self.turn.phase)
        self.turn.tick(now=10.24)
        self.assertEqual(1, self.calls.count(('resume',)))
        self.turn.tick(now=10.25)
        self.assertEqual('listening', self.turn.phase)
        self.assertEqual(2, self.calls.count(('resume',)))

    def test_playback_cancel_sends_failed_once(self):
        self.start()
        self.turn.submit_final('你好')
        self.turn.respond({'request_id': 'server-req-1'}, lambda body: body, set())
        self.turn.media_cancelled()
        self.turn.media_cancelled()
        self.assertEqual(1, self.calls.count(('done', 'req-1', True)))

    def test_play_finishes_while_disconnected_does_not_resume(self):
        self.start()
        self.turn.submit_final('你好')
        self.turn.respond({'request_id': 'server-req-1'}, lambda body: body, set())
        self.turn.media_started()
        self.turn.set_ready(False)
        self.turn.media_finished(now=10)
        self.turn.tick(now=11)
        self.assertEqual('connecting', self.turn.phase)
        self.assertEqual(1, self.calls.count(('resume',)))

    def test_stale_reply_and_duplicate_do_not_replay(self):
        self.start()
        self.turn.submit_final('你好')
        with self.assertRaises(ValueError):
            self.turn.respond({'request_id': 'server-old'}, lambda body: body, set())
        body = {'request_id': 'server-req-1'}
        self.turn.respond(body, lambda body: body, set())
        self.turn.media_started()
        count = self.calls.count(('pause',))
        self.turn.respond(body, lambda body: 'already-played', {'server-req-1'})
        self.assertEqual(count, self.calls.count(('pause',)))
        self.assertEqual('playing', self.turn.phase)

    def test_failed_submit_releases_speech_and_resumes(self):
        self.start()
        self.turn.send = lambda text: None
        self.assertIsNone(self.turn.submit_final('你好'))
        self.assertIn(('speech_cancel',), self.calls)
        self.assertEqual('listening', self.turn.phase)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.ready = []
        self.link = ServerLink(on_ready=self.ready.append)
        self.frames = []

    def hello(self, epoch='server-1'):
        return {'type': 'hello_ok', 'protocol_version': 2, 'model_ready': True,
                'model_version': 'V2.0.0', 'server_instance_id': epoch,
                'session_id': self.link.session_id, 'max_inflight': 1}

    def connect(self):
        self.link._handshake(self.hello(), self.frames.append)

    def request(self):
        self.connect()
        key = self.link.submit('你好')
        self.assertIsNotNone(key)
        return key

    def test_not_ready_cannot_submit(self):
        self.assertIsNone(self.link.submit('你好'))
        self.assertTrue(self.link.outgoing.empty())

    def test_ready_fields_are_mandatory(self):
        for field, value in [('model_ready', False), ('protocol_version', 1), ('server_instance_id', ''),
                             ('max_inflight', 8), ('session_id', 'wrong')]:
            message = self.hello()
            message[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.link._handshake(message, self.frames.append)
        self.assertFalse(self.link.connected)

    def test_final_has_end_flag_and_only_one_request(self):
        key = self.request()
        frame = self.link.outgoing.get_nowait()
        self.assertEqual({'type': 'request', 'id': key, 'text': '你好', 'end': True}, frame)
        self.assertIsNone(self.link.submit('第二句'))

    def test_accepted_does_not_pop_waiter_or_emit_reply(self):
        key = self.request()
        self.link.drain()
        self.link._received({'type': 'accepted', 'id': key, 'status': 'processing'}, self.frames.append)
        self.assertIn(key, self.link.waiting)
        self.assertTrue(self.link.waiting[key]['accepted'])
        self.assertEqual(['accepted'], [e['type'] for e in self.link.drain()])

    def test_control_ack_does_not_pop_waiter(self):
        key = self.request()
        self.link._received({'type': 'control_ack', 'control': 'speech_start'}, self.frames.append)
        self.assertIn(key, self.link.waiting)

    def test_same_epoch_reconnect_queries_original_id_never_resends_text(self):
        key = self.request()
        self.link._ready(False)
        self.frames.clear()
        self.link._handshake(self.hello(), self.frames.append)
        self.assertEqual([{'type': 'request_status', 'id': key}], self.frames)
        self.assertIn(key, self.link.waiting)

    def test_new_epoch_discards_old_request_without_resending(self):
        key = self.request()
        self.link._ready(False)
        self.frames.clear()
        self.link._handshake(self.hello('server-2'), self.frames.append)
        self.assertEqual({}, self.link.waiting)
        self.assertFalse(any(f['type'] in ('request', 'request_status') for f in self.frames))
        restart = [e for e in self.link.drain() if e['type'] == 'server_restarted']
        self.assertEqual([key], restart[0]['ids'])

    def test_result_plays_once_and_blocks_next_until_done(self):
        key = self.request()
        self.link.drain()
        frame = {'type': 'result', 'id': key, 'meme_id': 'meme-0001', 'reply_text': '你好呀'}
        self.link._received(frame, self.frames.append)
        self.link._received(frame, self.frames.append)
        self.assertEqual(1, sum(e['type'] == 'reply' for e in self.link.drain()))
        self.assertEqual(key, self.link.awaiting_playback)
        self.assertIsNone(self.link.submit('第二句'))
        self.link.playback_done(key)
        self.assertIsNotNone(self.link.submit('第二句'))

    def test_conversation_id_is_forwarded_without_changing_the_meme_id(self):
        key = self.request()
        self.link.drain()
        self.link._received({'type': 'result', 'id': key, 'meme_id': 'meme-0001',
                             'reply_text': '你好呀', 'conversation_id': 'conversation-1'}, self.frames.append)
        reply = self.link.drain()[-1]
        self.assertEqual('reply', reply['type'])
        self.assertEqual('conversation-1', reply['conversation_id'])
        self.assertEqual('meme-0001', reply['meme_id'])
        self.assertEqual('conversation-1', self.link.conversation_id)

    def test_legacy_result_keeps_one_conversation_for_same_server_boot(self):
        key = self.request()
        self.link.drain()
        frame = {'type': 'result', 'meme_id': 'meme-0001', 'reply_text': '你好呀'}
        self.link._received(dict(frame, id=key), self.frames.append)
        first = self.link.drain()[-1]['conversation_id']
        self.link.playback_done(key)
        key = self.link.submit('再说一句')
        self.link._ready(False)
        self.link._handshake(self.hello(), self.frames.append)
        self.link.drain()
        self.link._received(dict(frame, id=key), self.frames.append)
        self.assertEqual(first, self.link.drain()[-1]['conversation_id'])
        self.assertTrue(first.startswith('legacy-'))

    def test_invalid_conversation_id_fails_playback_instead_of_selecting_background(self):
        key = self.request()
        self.link._received({'type': 'result', 'id': key, 'meme_id': 'meme-0001',
                             'reply_text': '你好呀', 'conversation_id': '../bad'}, self.frames.append)
        self.assertEqual('playback_failed', self.link.playback_receipts[key])
        self.assertIsNone(self.link.conversation_id)
        self.assertEqual('reply_error', self.link.drain()[-1]['type'])

    def test_nested_reply_keeps_outer_conversation_identity(self):
        reply = parse_reply({'conversation_id': 'conversation-1',
                             'text': '{"meme_id":"meme-0001","reply_text":"你好呀"}'})
        self.assertEqual('conversation-1', reply['conversation_id'])

    def test_background_revision_is_optional_strict_and_forwarded(self):
        base = {'meme_id':'meme-0001', 'reply_text':'你好呀', 'conversation_id':'conversation-1'}
        self.assertEqual(0, parse_reply(base)['background_revision'])
        self.assertEqual(3, parse_reply(dict(base, background_revision=3))['background_revision'])
        for value in (True, False, -1, 1.0, '1', None, [], {}):
            for nested in (False, True):
                with self.subTest(value=value, nested=nested), self.assertRaises(ValueError):
                    payload = dict(base, background_revision=value)
                    parse_reply({'text':__import__('json').dumps(payload)} if nested else payload)
        with self.assertRaises(ValueError):
            parse_reply({'background_revision':1,
                         'text':'{"meme_id":"meme-0001","reply_text":"你好","background_revision":0}'})
        key = self.request()
        self.link.drain()
        self.link._received(dict(base, type='result', id=key, background_revision=2), self.frames.append)
        event = self.link.drain()[-1]
        self.assertEqual(2, event['background_revision'])
        self.assertEqual(2, self.link.background_revision)

    def test_old_receipts_cannot_replay_or_lower_background_revision(self):
        first = self.request()
        frame = {'type':'result', 'id':first, 'meme_id':'meme-0001',
                 'reply_text':'第一句', 'conversation_id':'conversation-1', 'background_revision':0}
        self.link._received(frame, self.frames.append)
        self.link.playback_done(first)
        second = self.link.submit('第二句')
        self.link._received(dict(frame, id=second, reply_text='第二句', background_revision=1), self.frames.append)
        self.link.drain()
        self.link._received(frame, self.frames.append)
        self.assertEqual([], self.link.drain())
        self.assertEqual(1, self.link.background_revision)
        self.link.playback_done(second)
        third = self.link.submit('第三句')
        self.link._received(dict(frame, id=third), self.frames.append)
        events = self.link.drain()
        self.assertFalse(any(event['type']=='reply' for event in events))
        self.assertEqual('reply_error', events[-1]['type'])
        self.assertEqual('playback_failed', self.link.playback_receipts[third])
        self.assertEqual(('conversation-1',1), (self.link.conversation_id,self.link.background_revision))

    def test_server_restart_clears_background_revision_history(self):
        key = self.request()
        self.link._received({'type':'result', 'id':key, 'meme_id':'meme-0001', 'reply_text':'你好',
                             'conversation_id':'conversation-1', 'background_revision':3}, self.frames.append)
        self.link._handshake(self.hello('new-server'), self.frames.append)
        self.assertEqual(0, self.link.background_revision)
        self.assertEqual({}, self.link._conversation_revisions)

    def test_playback_receipt_survives_disconnect_until_control_ack(self):
        key = self.request()
        self.link._received({'type': 'result', 'id': key, 'meme_id': 'meme-0001',
                             'reply_text': '你好呀'}, self.frames.append)
        self.link._ready(False)
        self.link.playback_done(key)
        self.frames.clear()
        self.link._handshake(self.hello(), self.frames.append)
        self.assertIn({'type': 'playback_done', 'id': key}, self.frames)
        self.link._received({'type': 'control_ack', 'control': 'playback_done', 'id': key}, self.frames.append)
        self.assertEqual({}, self.link.playback_receipts)

    def test_malformed_result_releases_server_playback_lease(self):
        key = self.request()
        self.link._received({'type': 'result', 'id': key, 'meme_id': '../bad',
                             'reply_text': '你好'}, self.frames.append)
        self.assertEqual('playback_failed', self.link.playback_receipts[key])
        self.assertIsNone(self.link.awaiting_playback)
        self.assertTrue(any(e['type'] == 'reply_error' for e in self.link.drain()))

    def test_unknown_request_releases_waiter_without_resend(self):
        key = self.request()
        self.link.outgoing.get_nowait()
        self.link._received({'type': 'error', 'id': key, 'code': 'unknown_request'}, self.frames.append)
        self.assertEqual({}, self.link.waiting)
        self.assertTrue(self.link.outgoing.empty())
        event = self.link.drain()[-1]
        self.assertEqual('server_error', event['type'])
        self.assertIn('未自动重发', event['message'])


class CaptureTests(unittest.TestCase):
    def test_endpoint_final_is_sent_after_alsa_context_closes(self):
        from capture_turn import MicrophoneGate, listen_once
        gate = MicrophoneGate()
        events, closed = [], []

        class Capture:
            def __init__(self, *_args, **_kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                closed.append(True)
            def chunks(self):
                yield b'\0\0' * 320
                yield b'\0\0' * 320
            def request_finish(self):
                pass
            def stats(self):
                return {}

        class Decoder:
            utterance_seen = True
            def __init__(self, finalized):
                self.finalized = finalized
            def feed(self, *_args):
                self.finalized('完整句子')
            def stats(self):
                return {}

        def send(kind, **values):
            if kind in ('final', 'paused'):
                self.assertTrue(closed, kind + ' must follow microphone close')
            events.append((kind, values))

        listen_once(gate, gate.epoch, ['fake-arecord'], Decoder, send, Capture)
        kinds = [kind for kind, _values in events]
        self.assertLess(kinds.index('paused'), kinds.index('final'))
        self.assertEqual(1, kinds.count('final'))

    def test_pause_drops_unfinished_sentence_and_emits_cancel_after_close(self):
        from capture_turn import MicrophoneGate, listen_once
        gate = MicrophoneGate()
        events, closed = [], []

        class Capture:
            def __init__(self, *_args, **_kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                closed.append(True)
            def chunks(self):
                yield b'\0\0' * 320
            def request_finish(self):
                pass
            def stats(self):
                return {}

        class Decoder:
            utterance_seen = True
            def __init__(self, finalized):
                self.finalized = finalized
            def feed(self, *_args):
                gate.command('pause')
            def finish(self):
                raise AssertionError('must not force-finalize an interrupted phrase')
            def stats(self):
                return {}

        def send(kind, **values):
            if kind == 'utterance_cancelled':
                self.assertTrue(closed)
            events.append(kind)

        listen_once(gate, gate.epoch, ['fake-arecord'], Decoder, send, Capture)
        self.assertIn('utterance_cancelled', events)
        self.assertNotIn('final', events)


class WelcomeTests(unittest.TestCase):
    def test_initial_media_without_server_request_keeps_microphone_paused(self):
        calls = []
        turn = ConversationTurn(lambda text: 'id', lambda: calls.append('pause'),
                                lambda: calls.append('resume'))
        turn.media_started()
        turn.set_ready(True)
        turn.enable()
        self.assertEqual('playing', turn.phase)
        self.assertNotIn('resume', calls)
        turn.media_finished(now=10)
        turn.tick(now=11)
        self.assertEqual(['pause', 'pause', 'resume'], calls)

    def test_old_boot_playback_completion_is_not_sent_to_new_boot(self):
        link = ServerLink()
        frames = []
        def hello(epoch):
            return {'type': 'hello_ok', 'protocol_version': 2, 'model_ready': True,
                    'model_version': 'V2.0.0', 'server_instance_id': epoch,
                    'session_id': link.session_id, 'max_inflight': 1}
        link._handshake(hello('one'), frames.append)
        key = link.submit('你好')
        link._received({'type': 'result', 'id': key, 'meme_id': 'meme-0001',
                        'reply_text': '你好呀'}, frames.append)
        link._ready(False)
        link._handshake(hello('two'), frames.append)
        self.assertFalse(link.playback_done(key))
        self.assertEqual({}, link.playback_receipts)

    def test_unknown_playback_receipt_is_discarded(self):
        link = ServerLink()
        link.playback_receipts['expired'] = 'playback_done'
        link._received({'type': 'error', 'id': 'expired', 'code': 'unknown_request'}, lambda frame: None)
        self.assertEqual({}, link.playback_receipts)


if __name__ == '__main__':
    unittest.main()
