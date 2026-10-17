#!/usr/bin/env python3
"""Install only this application's desktop/menu entries; no sudo or autostart."""
from pathlib import Path
import subprocess
import sys


def main():
    if sys.platform != 'linux':
        raise SystemExit('桌面快捷方式仅用于Linux；Windows测试请直接运行app.py。')
    root = Path(__file__).resolve().parent
    # Desktop Entry Exec uses its own quoting, not shell expansion.
    def quote(value):
        return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$').replace('%', '%%') + '"'
    entry = ('[Desktop Entry]\nType=Application\nVersion=1.0\nName=Meme猫播放器\n'
             'Comment=表情、配套声音与底部字幕\n'
             'Exec=/usr/bin/python3 ' + quote(root / 'app.py') + '\n'
             'Icon=' + str(root / 'cat.svg') + '\nTerminal=false\nCategories=AudioVideo;\nStartupNotify=true\n')
    applications = Path.home() / '.local/share/applications'
    applications.mkdir(parents=True, exist_ok=True)
    paths = [applications / 'meme-cat-native.desktop']
    try:
        desktop = subprocess.check_output(['xdg-user-dir', 'DESKTOP'], text=True).strip()
        folder = Path(desktop)
        if desktop and folder.is_dir() and folder != Path.home():
            paths.append(folder / 'meme-cat-native.desktop')
    except (OSError, subprocess.CalledProcessError):
        print('未找到桌面目录；仍可从应用菜单打开。')
    # Refuse to replace an unrelated desktop entry.
    for path in paths:
        if path.exists() and path.read_text(encoding='utf-8') != entry:
            raise SystemExit('已有同名且内容不同的快捷方式，未覆盖：' + str(path))
    for path in paths:
        path.write_text(entry, encoding='utf-8', newline='\n')
        path.chmod(0o755)
        print('已创建：' + str(path))
    print('桌面或应用菜单打开“Meme猫播放器”；若系统要求信任快捷方式，请本人确认。')


if __name__ == '__main__':
    main()
