"""A response keeps its visual background while workers, resize and replay run."""
import ast
from io import BytesIO
import json
from pathlib import Path
import sys
import threading
import tempfile
import time
import unittest
import wave
from types import SimpleNamespace

from PIL import Image

DESKTOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP.parent))
from launch import Player, PlayerError
from native_core import read_asset
from render_pipeline import RenderEngine
from directory_library import DirectoryPlayer
from server_link import ServerLink


class EventSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.player = object.__new__(Player)
        self.player.lock = threading.RLock()
        self.player.event = {'revision': 0, 'type': 'idle'}
        self.player.requests = {}
        self.player.library = lambda: {'items': [{'id': 'meme-0001'}]}
        self.player.background_id = 3
        self.player.conversation_id = 'conversation-1'
        self.player.background_revision = 0
        self.body = {'request_id': 'req-1', 'meme_id': 'meme-0001', 'reply_text': '你好呀'}

    def test_initial_manual_event_uses_existing_background(self):
        event = self.player.respond(self.body)
        self.assertEqual(3, event['background_id'])
        self.assertEqual('conversation-1', event['conversation_id'])
        self.assertEqual(0, event['background_revision'])

    def test_queued_event_freezes_revision_and_duplicate_cannot_replace_it(self):
        first = self.player.respond(dict(self.body, background_revision=1))
        self.player.background_revision = 2
        self.player.background_id = 8
        self.assertEqual(1, first['background_revision'])
        self.assertEqual(3, first['background_id'])
        self.assertEqual(first, self.player.respond(self.body))
        with self.assertRaises(PlayerError):
            self.player.respond(dict(self.body, background_revision=2))

    def test_duplicate_after_new_conversation_keeps_original_event(self):
        original = self.player.respond(self.body)
        self.player.background_id = 8
        self.player.conversation_id = 'conversation-2'
        self.assertEqual(original, self.player.respond(self.body))
        self.assertEqual(original, self.player.event)

    def test_replay_can_explicitly_freeze_old_conversation_background(self):
        original = self.player.respond(self.body)
        self.player.background_id = 8
        self.player.conversation_id = 'conversation-2'
        replay = self.player.respond(dict(self.body, request_id='replay-1',
                                         background_id=original['background_id'],
                                         conversation_id=original['conversation_id']))
        self.assertEqual(3, replay['background_id'])
        self.assertEqual('conversation-1', replay['conversation_id'])

    def test_same_request_cannot_change_explicit_background(self):
        self.player.respond(dict(self.body, background_id=3))
        with self.assertRaises(PlayerError):
            self.player.respond(dict(self.body, background_id=8))

    def test_invalid_background_ids_are_rejected(self):
        for value in (True, False, 0, 11, '3', 3.0):
            with self.subTest(value=value), self.assertRaises(PlayerError):
                self.player.respond(dict(self.body, background_id=value))

    def test_invalid_background_revision_does_not_create_an_event(self):
        for value in (True, False, -1, 1.0, '1', None):
            with self.subTest(value=value), self.assertRaises(PlayerError):
                self.player.respond(dict(self.body, background_revision=value))
        self.assertEqual({'revision':0,'type':'idle'}, self.player.event)


class DesktopSnapshotTests(unittest.TestCase):
    def method(self, name, **values):
        tree = ast.parse((DESKTOP/'app.py').read_text())
        node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        namespace = dict(values)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), namespace)
        return namespace[name]

    def test_replay_retains_displayed_revision_instead_of_latest_selection(self):
        calls = []
        desktop = SimpleNamespace(key='meme-0001', reply='第一句',
            current_event=dict(background_id=3, conversation_id='dialogue', background_revision=0),
            request=lambda *args:calls.append(args))
        self.method('replay')(desktop)
        self.assertEqual([('meme-0001','第一句',3,'dialogue',0)], calls)

    def test_health_distinguishes_displayed_revision_from_network_revision(self):
        status = lambda: {}
        desktop = SimpleNamespace(
            turn=SimpleNamespace(lock=threading.RLock(), phase='playing', enabled=True, media_active=True),
            link=SimpleNamespace(lock=threading.RLock(), connected=True, model_ready=True, model_version='3.0.0',
                waiting={}, awaiting_playback='req', playback_receipts={}, background_revision=2),
            speech=SimpleNamespace(running=lambda:True, paused=True), speech_state='paused', speech_error='',
            voice_recovery=SimpleNamespace(status=status), music=SimpleNamespace(status=status),
            audio_output=SimpleNamespace(status=status), standby=SimpleNamespace(status=status),
            current_event=dict(conversation_id='dialogue',background_id=3,background_revision=1),
            current_visual_name='meme-0001-cat-3.gif')
        result = self.method('communication_status', args=SimpleNamespace(end_silence=.7),
                             player=SimpleNamespace(background_id=8,background_revision=2))(desktop)
        self.assertEqual((1,2), (result['background_revision'],result['server_background_revision']))
        self.assertEqual(('dialogue',3), (result['conversation_id'],result['background_id']))


class RenderSnapshotTests(unittest.TestCase):
    def setUp(self):
        data = BytesIO()
        Image.new('RGB', (2, 2), 'red').save(data, format='GIF')
        self.blob = data.getvalue()
        self.calls = []

        def load(key, background_id=None):
            self.calls.append((key, background_id))
            return {'visual_blob': self.blob, 'audio_blob': b'original-wav',
                    'duration_seconds': .05, 'visual_name': 'meme-0001-cat-%s.gif' % background_id,
                    'updated_at': 123}

        self.renderer = RenderEngine(load)
        self.addCleanup(self.renderer.close)

    def ready(self, event, reflow=False):
        generation = self.renderer.request(event, (100, 100), reflow=reflow)
        until = time.monotonic() + 2
        while time.monotonic() < until:
            for result in self.renderer.take():
                if result['type'] == 'render_error':
                    self.fail(result['message'])
                if result['type'] == 'ready' and result['generation'] == generation:
                    return result
            time.sleep(.005)
        self.fail('background renderer did not prepare a frame')

    def test_worker_uses_frozen_background_and_original_audio(self):
        event = {'meme_id': 'meme-0001', 'background_id': 3, 'conversation_id': 'conversation-1'}
        result = self.ready(event)
        event['background_id'] = 8
        self.assertEqual([('meme-0001', 3)], self.calls)
        self.assertEqual(3, result['event']['background_id'])
        self.assertEqual(b'original-wav', result['audio'])
        self.assertEqual('meme-0001-cat-3.gif', result['visual_name'])

    def test_background_change_replaces_cache_even_when_gif_bytes_are_equal(self):
        first = self.ready({'meme_id': 'meme-0001', 'background_id': 3})
        second = self.ready({'meme_id': 'meme-0001', 'background_id': 8})
        self.assertNotEqual(first['cache_key'], second['cache_key'])
        self.assertEqual([('meme-0001', 3), ('meme-0001', 8)], self.calls)

    def test_resize_and_replay_reuse_same_background_cache(self):
        event = {'meme_id': 'meme-0001', 'background_id': 3}
        first = self.ready(event)
        resized = self.ready(event, reflow=True)
        replayed = self.ready(event)
        self.assertEqual(first['cache_key'], resized['cache_key'])
        self.assertEqual(first['cache_key'], replayed['cache_key'])
        self.assertEqual([('meme-0001', 3)] * 3, self.calls)

    def test_native_loader_passes_explicit_background(self):
        class Directory:
            def read_native_asset(inner, key, background_id=None):
                return key, background_id
        self.assertEqual(('meme-0001', 3), read_asset(Directory(), 'meme-0001', background_id=3))


class ReplyMediaIntegrationTests(unittest.TestCase):
    def test_network_reply_to_frozen_event_preserves_background_and_original_wav(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / 'sucai_meme/meme_gif'
            background = original.parent / 'meme_gif_background'
            original.mkdir(parents=True)
            background.mkdir()
            data = BytesIO()
            Image.new('RGB', (2, 2), 'red').save(data, format='GIF')
            stem = 'meme-0001-cat'
            (original / (stem + '.gif')).write_bytes(data.getvalue())
            sound = BytesIO()
            with wave.open(sound, 'wb') as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16000)
                audio.writeframes(b'\0\0' * 160)
            (original / (stem + '.wav')).write_bytes(sound.getvalue())
            for number in range(1, 11):
                data = BytesIO()
                Image.new('RGB', (2, 2), (number * 20, 0, 0)).save(data, format='GIF')
                (background / (stem + '-%s.gif' % number)).write_bytes(data.getvalue())
            player_root = root / 'player'
            (player_root / 'website').mkdir(parents=True)
            (player_root / 'website/library.json').write_text(json.dumps({'items': [], 'scenarios': []}))
            choices = iter((1, 4, 10))
            player = DirectoryPlayer(player_root, original, choice=lambda values: next(choices))
            link = ServerLink()
            link._handshake({'type': 'hello_ok', 'protocol_version': 2, 'model_ready': True,
                             'model_version': 'V2.0.0', 'server_instance_id': 'boot-1',
                             'session_id': link.session_id, 'max_inflight': 1}, lambda message: None)
            key = link.submit('你好')
            link.drain()
            link._received({'type': 'result', 'id': key, 'meme_id': 'meme-0001',
                            'reply_text': '你好呀', 'conversation_id': 'conversation-1'}, lambda message: None)
            reply = link.drain()[-1]
            selected = player.begin_conversation(reply['conversation_id'])
            event = player.respond({'meme_id': reply['meme_id'], 'reply_text': reply['reply_text'],
                                    'request_id': 'server-' + reply['id'],
                                    'conversation_id': reply['conversation_id'], 'background_id': selected})
            player.begin_conversation('conversation-2')
            self.assertEqual(10, player.background_id)
            row = read_asset(player, event['meme_id'], background_id=event['background_id'])
            self.assertEqual('meme-0001', row['id'])
            self.assertEqual(4, row['background_id'])
            self.assertEqual(stem + '-4.gif', row['visual_name'])
            self.assertEqual(sound.getvalue(), row['audio_blob'])
            self.assertEqual(stem + '.wav', row['audio_name'])


class DirectoryResponseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        original = root / 'sucai_meme/meme_gif'
        background = original.parent / 'meme_gif_background'
        original.mkdir(parents=True)
        background.mkdir()
        image = BytesIO()
        Image.new('RGB', (2, 2), 'red').save(image, format='GIF')
        for key in ('meme-0001', 'meme-0002'):
            (original / (key + '-cat.gif')).write_bytes(image.getvalue())
            for number in range(1, 11):
                (background / (key + '-cat-%s.gif' % number)).write_bytes(image.getvalue())
        player_root = root / 'player'
        (player_root / 'website').mkdir(parents=True)
        (player_root / 'website/library.json').write_text(json.dumps({'items': [], 'scenarios': []}))
        self.choices = []
        selections = iter((1, 4, 10))
        def choose(values):
            self.choices.append(tuple(values))
            return next(selections)
        self.player = DirectoryPlayer(player_root, original, choice=choose)

    def body(self, request_id, conversation_id, key='meme-0001'):
        return {'request_id': request_id, 'conversation_id': conversation_id,
                'meme_id': key, 'reply_text': '你好呀'}

    def test_manual_conversation_selects_once_across_different_meme_ids(self):
        first = self.player.respond(self.body('first', 'conversation-1'))
        second = self.player.respond(self.body('second', 'conversation-1', 'meme-0002'))
        third = self.player.respond(self.body('third', 'conversation-2'))
        self.assertEqual((4, 4, 10), (first['background_id'], second['background_id'], third['background_id']))
        self.assertEqual(3, len(self.choices))  # Welcome plus two new conversations.
        self.assertEqual(('meme-0001', 'meme-0002'), (first['meme_id'], second['meme_id']))

    def test_duplicate_old_request_does_not_change_new_conversation(self):
        body = self.body('first', 'conversation-1')
        first = self.player.respond(body)
        self.player.respond(self.body('second', 'conversation-2'))
        self.assertEqual(first, self.player.respond(body))
        self.assertEqual(10, self.player.background_id)
        self.assertEqual('conversation-2', self.player.conversation_id)
        self.assertEqual(3, len(self.choices))

    def test_revision_change_freezes_events_and_old_cache_keeps_new_selection(self):
        body = dict(self.body('first', 'conversation-1'), background_revision=0)
        first = self.player.respond(body)
        second = self.player.respond(dict(self.body('second', 'conversation-1'), background_revision=1))
        self.assertEqual((4,10), (first['background_id'],second['background_id']))
        self.assertEqual((0,1), (first['background_revision'],second['background_revision']))
        self.assertEqual(first, self.player.respond(body))
        self.assertEqual(('conversation-1',1,10),
                         (self.player.conversation_id,self.player.background_revision,self.player.background_id))
        self.assertEqual(second, self.player.event)

    def test_explicit_replay_old_background_does_not_change_current_conversation(self):
        first = self.player.respond(self.body('first', 'conversation-1'))
        self.player.respond(self.body('second', 'conversation-2'))
        replay = self.player.respond(dict(first, request_id='replay'))
        self.assertEqual(4, replay['background_id'])
        self.assertEqual('conversation-1', replay['conversation_id'])
        self.assertEqual(10, self.player.background_id)
        self.assertEqual('conversation-2', self.player.conversation_id)
        row = read_asset(self.player, replay['meme_id'], background_id=replay['background_id'])
        self.assertEqual('meme-0001-cat-4.gif', row['visual_name'])
        self.assertEqual(3, len(self.choices))

    def test_invalid_manual_reply_cannot_change_selected_background(self):
        self.player.respond(self.body('first', 'conversation-1'))
        for changes in ({'meme_id': 'meme-0999'}, {'reply_text': ''}, {'request_id': None},
                        {'conversation_id': '../bad'}, {'background_id': None}):
            with self.subTest(changes=changes), self.assertRaises(PlayerError):
                self.player.respond(dict(self.body('bad', 'conversation-2'), **changes))
            self.assertEqual(4, self.player.background_id)
            self.assertEqual('conversation-1', self.player.conversation_id)
        self.assertEqual(2, len(self.choices))


if __name__ == '__main__':
    unittest.main()
