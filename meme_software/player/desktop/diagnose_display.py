#!/usr/bin/env python3
"""Read-only display diagnostics; stdlib only, no sudo/install/network."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def collect():
    result = {'format': 'meme-display-diagnostics-v1', 'system': platform.system(),
              'machine': platform.machine(), 'framebuffers': [], 'drm': [],
              'display_session_present': bool(os.environ.get('DISPLAY')),
              'session_type': os.environ.get('XDG_SESSION_TYPE', 'unknown'),
              'scope': 'Desktop/driver advertised modes only; not measured physical TFT refresh rate.'}
    for root, fields, name in [(Path('/sys/class/graphics'), ('name', 'virtual_size', 'bits_per_pixel', 'modes'), 'framebuffers'),
                               (Path('/sys/class/drm'), ('status', 'enabled', 'modes'), 'drm')]:
        if not root.is_dir():
            continue
        for device in sorted(root.iterdir()):
            item = {'device': device.name}
            for field in fields:
                try:
                    item[field] = (device / field).read_text(encoding='utf-8').strip()[:2000]
                except OSError:
                    pass
            if len(item) > 1:
                result[name].append(item)
    commands = [('xrandr', '--current')] if result['display_session_present'] else []
    commands += [('vcgencmd', 'get_throttled')]
    for command in commands:
        executable = shutil.which(command[0])
        if executable:
            try:
                run = subprocess.run([executable, *command[1:]], capture_output=True, text=True, timeout=4)
                result[command[0]] = {'exit_code': run.returncode,
                                     'output': (run.stdout + run.stderr)[:10000]}
            except (OSError, subprocess.TimeoutExpired):
                result[command[0]] = '不可用或超时，未修改设置'
    if not result['display_session_present']:
        result['note'] = '此会话没有DISPLAY；不能据此判断显示屏离线。需要时从Pi自身桌面终端运行本脚本。'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    content = json.dumps(collect(), ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(content, encoding='utf-8', newline='\n')
    print(content, end='')
