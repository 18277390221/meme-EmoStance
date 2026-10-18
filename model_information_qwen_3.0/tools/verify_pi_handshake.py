"""Probe the installed Pi over its existing SSH setup without copying files.

--check reads local player health and only relevant process PIDs.
--exercise-playback requests one existing local asset, never model inference.
--exercise-request runs one independent Pi-to-server version 2 text exchange.
Reports omit credentials, recognized user text, model replies and process argv.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

if __package__ in (None, "") and "--remote" not in sys.argv:
    import runpy
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "bootstrap.py"))["register_package"]()


HEALTH_BOOL_FIELDS = ('voice_enabled', 'media_active', 'server_connected', 'model_ready',
                      'playback_pending', 'asr_running', 'microphone_paused')
HEALTH_COUNT_FIELDS = ('protocol_version', 'pending_requests', 'pending_playback_receipts')
PHASES = {'idle', 'connecting', 'listening', 'waiting', 'preparing', 'playing', 'cooldown'}
ASSET_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')


def sanitized_health(payload):
    """Allow-list status fields; never echo arbitrary HTTP payload content."""
    result = {'health_available': isinstance(payload, dict) and payload.get('status') == 'ready'}
    state = payload.get('communication', {}) if isinstance(payload, dict) else {}
    if not isinstance(state, dict):
        return result
    for key in HEALTH_BOOL_FIELDS:
        if isinstance(state.get(key), bool):
            result[key] = state[key]
    for key in HEALTH_COUNT_FIELDS:
        if type(state.get(key)) is int and 0 <= state[key] <= 100000:
            result[key] = state[key]
    if state.get('phase') in PHASES:
        result['phase'] = state['phase']
    version = state.get('model_version')
    if isinstance(version, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', version):
        result['model_version'] = version
    background_id = state.get('background_id')
    if type(background_id) is int and 1 <= background_id <= 10:
        result['background_id'] = background_id
    conversation_id = state.get('conversation_id')
    if isinstance(conversation_id, str) and ASSET_ID.fullmatch(conversation_id):
        result['conversation_id'] = conversation_id
    visual = state.get('selected_visual')
    if isinstance(visual, str) and re.fullmatch(r'meme-\d{4}(?:-[^/\\\x00-\x1f]+)?-(?:[1-9]|10)\.gif', visual):
        result['selected_visual'] = visual
    return result


def http_json(port, path, body=None):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode('utf-8')
    request = Request(f'http://127.0.0.1:{port}' + path, data=data,
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=3) as response:
        if response.status != 200:
            raise ValueError('http_status')
        return json.loads(response.read(2 * 1024 * 1024))


def process_ids(software_dir, proc_root=Path('/proc')):
    """Select this player's app/worker and their arecord/aplay descendants."""
    desktop = Path(software_dir).resolve() / 'player/desktop'
    records = {}
    for path in proc_root.iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = {}
            for line in (path / 'status').read_text().splitlines():
                if line.startswith(('Uid:', 'PPid:')):
                    key, value = line.split(':', 1)
                    status[key] = int(value.split()[0])
            if status.get('Uid') != os.getuid():
                continue
            comm = (path / 'comm').read_text().strip()
            if not (comm.startswith('python') or comm in ('arecord', 'aplay')):
                continue
            argv = (path / 'cmdline').read_bytes().decode('utf-8', 'replace').split('\0')
            cwd = (path / 'cwd').resolve()
            category = comm if comm in ('arecord', 'aplay') else None
            for name, label in (('app.py', 'app'), ('stt_worker.py', 'stt')):
                expected = desktop / name
                if any(argument and (Path(argument) if Path(argument).is_absolute() else cwd / argument).resolve() == expected
                       for argument in argv[1:] if not argument.startswith('-')):
                    category = label
            records[int(path.name)] = (status.get('PPid', -1), category)
        except (OSError, ValueError):
            continue
    selected = {pid for pid, (_parent, label) in records.items() if label in ('app', 'stt')}
    changed = True
    while changed:
        before = len(selected)
        selected.update(pid for pid, (parent, label) in records.items()
                        if parent in selected and label in ('arecord', 'aplay'))
        changed = before != len(selected)
    return {label: sorted(pid for pid in selected if records[pid][1] == label)
            for label in ('app', 'stt', 'arecord', 'aplay')}


def snapshot(options):
    try:
        report = sanitized_health(http_json(options['player_port'], '/api/health'))
    except (OSError, ValueError, TimeoutError):
        report = {'health_available': False}
    report['processes'] = process_ids(options['software_dir'])
    return report


def select_asset(payload):
    rows = payload if isinstance(payload, list) else []
    keys = [row.get('id') for row in rows if isinstance(row, dict)
            and isinstance(row.get('id'), str) and ASSET_ID.fullmatch(row['id'])]
    if not keys:
        raise ValueError('no_valid_local_asset')
    return 'meme-0062' if 'meme-0062' in keys else keys[0]


def playback_exercise(options):
    before = snapshot(options)
    report = {'action': 'exercise-playback', 'before': before, 'playback_verified': False,
              'listening_restored': False}
    if not (before.get('model_ready') is True and before.get('phase') == 'listening'
            and before.get('microphone_paused') is False and not before.get('pending_requests')
            and not before.get('media_active')):
        return {**report, 'error': 'player_not_ready_for_controlled_playback'}
    key = select_asset(http_json(options['player_port'], '/api/assets'))
    http_json(options['player_port'], '/api/response',
              {'meme_id': key, 'reply_text': '握手联调完成', 'request_id': 'desktop-verify-' + uuid.uuid4().hex})
    report['meme_id'] = key
    started, observations = time.monotonic(), []
    played = False
    while time.monotonic() - started < options['timeout']:
        state = snapshot(options)
        phase = state.get('phase')
        observation = {'phase': phase, 'microphone_paused': state.get('microphone_paused'),
                       'arecord_running': bool(state['processes']['arecord']),
                       'aplay_running': bool(state['processes']['aplay'])}
        if not observations or observation != observations[-1]:
            observations.append(observation)
        if phase == 'playing':
            played = True
            if state.get('microphone_paused') is True and not state['processes']['arecord']:
                report['playback_verified'] = True
            if state['processes']['arecord']:
                report['microphone_active_during_playback'] = True
        if played and phase == 'listening' and state.get('microphone_paused') is False and state['processes']['arecord']:
            report['listening_restored'] = True
            break
        time.sleep(.1)
    report.update(observations=observations[:128], after=state,
                  elapsed_seconds=round(time.monotonic() - started, 2))
    return report


def local_token(software_dir):
    token = os.environ.get('ZT_CHAT_TOKEN', '').strip()
    if not token:
        path = Path.home() / 'zerotier_chat/.env'
        config_path = Path(software_dir) / 'player/data/speech-config.json'
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(config, dict) and isinstance(config.get('token_file'), str):
                path = Path(config['token_file']).expanduser()
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('ZT_CHAT_TOKEN='):
                token = line.split('=', 1)[1].strip()
                break
    if len(token) < 16:
        raise ValueError('local_auth_unavailable')
    return token


def request_exercise(options):
    key, session = uuid.uuid4().hex, uuid.uuid4().hex
    report = {'action': 'exercise-request', 'request_id': key, 'model_ready': False,
              'accepted_before_result': False, 'reply_nonempty': False, 'playback_acknowledged': False,
              'frame_order': []}
    started = time.monotonic()
    with socket.create_connection((options['server_host'], options['server_port']), timeout=5) as connection:
        connection.settimeout(options['timeout'])
        stream = connection.makefile('rb')
        def send(message):
            connection.sendall((json.dumps(message, ensure_ascii=False) + '\n').encode('utf-8'))
        def receive():
            data = stream.readline(1024 * 1024 + 1)
            if not data or len(data) > 1024 * 1024 or not data.endswith(b'\n'):
                raise ValueError('invalid_protocol_frame')
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError('invalid_protocol_frame')
            return payload
        try:
            send({'type': 'hello', 'protocol_version': 2, 'name': 'meme-player-verify',
                  'session_id': session, 'token': local_token(options['software_dir'])})
            hello = receive()
            report['model_ready'] = (hello.get('type') == 'hello_ok' and hello.get('protocol_version') == 2
                                     and hello.get('model_ready') is True and hello.get('session_id') == session)
            version = hello.get('model_version')
            if isinstance(version, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', version):
                report['model_version'] = version
            if not report['model_ready']:
                return {**report, 'error': 'model_not_ready'}
            send({'type': 'speech_start'})
            send({'type': 'request', 'id': key, 'text': '你好，今天心情怎么样？', 'end': True})
            accepted = False
            while time.monotonic() - started < options['timeout']:
                message = receive()
                kind = message.get('type')
                if kind == 'ping':
                    send({'type': 'pong'})
                elif kind == 'control_ack':
                    control = message.get('control')
                    if control in ('speech_start', 'playback_done'):
                        report['frame_order'].append('control_ack:' + control)
                    if control == 'playback_done' and message.get('id') == key:
                        report['playback_acknowledged'] = True
                        break
                elif message.get('id') == key and kind == 'accepted':
                    accepted = True
                    report['frame_order'].append('accepted')
                elif message.get('id') == key and kind == 'result':
                    report['frame_order'].append('result')
                    report['accepted_before_result'] = accepted
                    reply = message.get('reply_text', message.get('reply'))
                    report['reply_nonempty'] = isinstance(reply, str) and bool(reply.strip())
                    meme_id = message.get('meme_id')
                    if isinstance(meme_id, str) and ASSET_ID.fullmatch(meme_id):
                        report['meme_id'] = meme_id
                    send({'type': 'playback_done', 'id': key})
                elif message.get('id') == key and kind == 'error':
                    report['frame_order'].append('error')
                    report['error'] = 'server_request_failed'
                    break
            report['elapsed_seconds'] = round(time.monotonic() - started, 2)
            return report
        finally:
            stream.close()


def remote_run(options):
    try:
        if options['action'] == 'exercise-playback':
            return playback_exercise(options)
        if options['action'] == 'exercise-request':
            return request_exercise(options)
        return {'action': 'check', **snapshot(options)}
    except (OSError, ValueError, TypeError, TimeoutError, KeyError):
        # Avoid reflecting exception text, credentials or user-visible HTTP bodies.
        return {'action': options.get('action'), 'error': 'probe_failed'}


def ssh_probe(settings, options):
    source = Path(__file__).read_text(encoding='utf-8')
    command = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=2',
               '-o', 'StrictHostKeyChecking=yes', '-o', 'IdentitiesOnly=yes',
               '-o', f'UserKnownHostsFile={settings.known_hosts}', '-i', str(settings.ssh_key),
               '--', f"{settings.network['PI_SSH_USER']}@{settings.network['PI_ZT_IP']}",
               'python3 -c ' + shlex.quote(source) + ' --remote']
    result = subprocess.run(command, input=json.dumps(options), text=True, capture_output=True,
                            timeout=options['timeout'] + 15)
    if result.returncode:
        raise RuntimeError('pi_probe_ssh_failed')
    report = json.loads(result.stdout)
    if not isinstance(report, dict):
        raise ValueError('pi_probe_invalid_report')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--check', action='store_true', help='Read health and process PIDs only (default)')
    actions.add_argument('--exercise-playback', action='store_true', help='Play one local asset, no inference')
    actions.add_argument('--exercise-request', action='store_true', help='Run one independent Pi-to-server text turn')
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--player-port', type=int, default=8784)
    parser.add_argument('--network-config', type=Path)
    parser.add_argument('--config', type=Path, default=None,
                        help='Selected Qwen model configuration; no model is loaded by this probe')
    parser.add_argument('--model-dir', type=Path, default=None)
    parser.add_argument('--remote', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.remote:
        print(json.dumps(remote_run(json.load(sys.stdin)), ensure_ascii=False))
        return 0
    if not 1 <= args.player_port <= 65535 or not 5 <= args.timeout <= 120:
        parser.error('player-port must be 1..65535; timeout must be 5..120 seconds')
    from model_information_qwen_v3.launcher import load_settings
    from model_information_qwen_v3.model import DEFAULT_CONFIG_PATH, PROJECT_ROOT
    args.network_config = args.network_config or PROJECT_ROOT / 'zerotier_chat/network.env'
    args.config, args.token_file = args.config or DEFAULT_CONFIG_PATH, None
    settings = load_settings(args)
    action = ('exercise-playback' if args.exercise_playback else
              'exercise-request' if args.exercise_request else 'check')
    options = {'action': action, 'timeout': args.timeout, 'player_port': args.player_port,
               'software_dir': settings.software_dir, 'server_host': settings.host, 'server_port': settings.port}
    try:
        report = ssh_probe(settings, options)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        report = {'action': action, 'error': 'pi_probe_unavailable'}
    print(json.dumps(report, ensure_ascii=False))
    return int('error' in report)


if __name__ == '__main__':
    raise SystemExit(main())
