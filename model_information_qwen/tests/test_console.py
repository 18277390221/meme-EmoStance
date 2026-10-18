"""Exercise terminal replay/follow against isolated files and fake polling."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from communication.common import FRAME_LIMIT
from model_information_qwen import console


def dialogue(number=0, *, reply=False, text="你好，Meme"):
    payload = {"device": "meme-player", "request_id": str(number)}
    direction = "服务器 → 树莓派" if reply else "树莓派 → 服务器"
    payload.update({"reply": text, "meme_id": "meme-0001"} if reply else {"text": text})
    return f"[2026-10-18T12:00:00+08:00] [{direction}] " + json.dumps(payload, ensure_ascii=False)


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "conversation.log"
        self.lock = self.root / "server.lock"

    def append(self, data):
        with self.path.open("ab") as stream:
            stream.write(data)

    def follow_changes(self, changes, *, tail=20):
        """Apply a deterministic filesystem update at each idle poll."""
        output = []
        changes = list(changes)
        alive = True
        def sleep(_seconds):
            nonlocal alive
            if changes:
                changes.pop(0)(output)
            else:
                alive = False
        console.follow_file(self.path, lambda: alive, tail=tail, emit=output.append, sleep=sleep)
        self.assertEqual(changes, [], "Viewer exited before all simulated writes")
        return output

    def test_replays_recent_complete_business_lines_and_keeps_chinese_readable(self):
        lines = [dialogue(number, reply=bool(number % 2)) for number in range(25)]
        records = [line + "\n[已连接] meme-player / fake\n" for line in lines]
        self.path.write_text("".join(records), encoding="utf-8")
        output = []
        console.follow_file(self.path, lambda: False, emit=output.append)
        self.assertEqual(output, lines[-20:])
        self.assertIn("你好，Meme", output[0])

    def test_tail_zero_shows_only_new_turns(self):
        old, new = dialogue(1), dialogue(1, reply=True, text="喵，我在听呢！")
        self.path.write_text(old + "\n", encoding="utf-8")
        output = self.follow_changes([lambda _: self.append((new + "\n").encode())], tail=0)
        self.assertEqual(output, [new])

    def test_waits_for_split_utf8_and_newline_without_printing_partial_records(self):
        line = dialogue(1)
        encoded = line.encode("utf-8")
        boundary = encoded.index("你".encode("utf-8")) + 1
        self.path.write_bytes(encoded[:boundary])
        def complete_utf8(output):
            self.assertEqual(output, [])
            self.append(encoded[boundary:])
        def complete_line(output):
            self.assertEqual(output, [])
            self.append(b"\n")
        output = self.follow_changes([complete_utf8, complete_line])
        self.assertEqual(output, [line])

    def test_unfinished_final_line_is_not_printed_when_owner_exits(self):
        line = dialogue(1)
        self.path.write_text(line, encoding="utf-8")
        output = []
        console.follow_file(self.path, lambda: False, emit=output.append)
        self.assertEqual(output, [])

    def test_waits_for_new_journal_creation(self):
        line = dialogue(1)
        output = self.follow_changes([lambda _: self.path.write_text(line + "\n", encoding="utf-8")])
        self.assertEqual(output, [line])

    def test_absent_journal_returns_immediately_after_owner_exits(self):
        output = []
        console.follow_file(self.path, lambda: False, emit=output.append,
                            sleep=lambda _: self.fail("Stopped server must not keep polling"))
        self.assertEqual(output, [])
        self.assertFalse(self.path.exists())

    def test_follows_truncated_journal_without_repeating_old_lines(self):
        old, new = dialogue(1, text="旧" * 1000), dialogue(2)
        self.path.write_text(old + "\n", encoding="utf-8")
        output = self.follow_changes([lambda _: self.path.write_text(new + "\n", encoding="utf-8")])
        self.assertEqual(output, [old, new])

    def test_follows_replacement_journal_from_start(self):
        old, new = dialogue(1), dialogue(2, reply=True)
        self.path.write_text(old + "\n", encoding="utf-8")
        def rotate(_):
            replacement = self.root / "next.log"
            replacement.write_text(new + "\n", encoding="utf-8")
            replacement.replace(self.path)
        self.assertEqual(self.follow_changes([rotate]), [old, new])

    def test_drains_complete_final_reply_before_returning_after_owner_exit(self):
        request, reply = dialogue(1), dialogue(1, reply=True)
        self.path.write_text(request + "\n", encoding="utf-8")
        output, alive = [], True
        def sleep(_):
            nonlocal alive
            self.append((reply + "\n").encode("utf-8"))
            alive = False
        console.follow_file(self.path, lambda: alive, emit=output.append, sleep=sleep)
        self.assertEqual(output, [request, reply])

    def test_discards_unrelated_invalid_and_oversized_lines_then_recovers(self):
        self.path.write_bytes(b'{"type":"hello","token":"private-fixture"}\n\xff\xfe\n')
        valid = dialogue(2, reply=True)
        def oversized(_):
            self.append(dialogue(1, text="x" * (FRAME_LIMIT + 1)).encode())
        def terminate_and_reply(_):
            self.append(("\n" + valid + "\n").encode("utf-8"))
        self.assertEqual(self.follow_changes([oversized, terminate_and_reply]), [valid])

    def test_ctrl_c_closes_viewer_file_and_preserves_journal(self):
        original = (dialogue(1) + "\n").encode("utf-8")
        self.path.write_bytes(original)
        stream = self.path.open("rb")
        def interrupt(_):
            raise KeyboardInterrupt
        with patch.object(Path, "open", return_value=stream), self.assertRaises(KeyboardInterrupt):
            console.follow_file(self.path, lambda: True, emit=lambda _: None, sleep=interrupt)
        self.assertTrue(stream.closed)
        self.assertEqual(self.path.read_bytes(), original)

    def test_rejects_invalid_tail_values(self):
        for tail in (-1, True, 1.5, "20"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                console.follow_file(self.path, lambda: False, tail=tail)

    def test_runtime_journal_is_selected_before_it_exists(self):
        self.lock.write_text("123\n" + json.dumps({"pid": 123, "dialogue_log": str(self.path)}) + "\n")
        self.assertEqual(console.resolve_log_source(self.lock, self.path), (self.path, 123))
        self.assertFalse(self.path.exists())

    def test_rejects_mismatched_runtime_pid_or_journal_path(self):
        for metadata in ({"pid": 124, "dialogue_log": str(self.path)},
                         {"pid": 123, "dialogue_log": str(self.root / "other.log")},
                         {"pid": 123, "dialogue_log": None}):
            with self.subTest(metadata=metadata):
                self.lock.write_text("123\n" + json.dumps(metadata) + "\n")
                with self.assertRaises(RuntimeError):
                    console.resolve_log_source(self.lock, self.path)

    def test_legacy_runtime_can_follow_regular_stdout_only(self):
        stdout = Path("/proc/123/fd/1")
        for suffix in ("", json.dumps({"pid": 123, "version": "V2.0.0"}) + "\n"):
            with self.subTest(suffix=suffix):
                self.lock.write_text("123\n" + suffix)
                with patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=stat.S_IFREG)) as probe:
                    self.assertEqual(console.resolve_log_source(self.lock, self.path), (stdout, 123))
                probe.assert_called_once_with()
                for mode in (stat.S_IFCHR, stat.S_IFIFO, stat.S_IFSOCK):
                    with self.subTest(mode=mode), patch.object(Path, "stat", return_value=SimpleNamespace(st_mode=mode)):
                        with self.assertRaisesRegex(RuntimeError, "原终端"):
                            console.resolve_log_source(self.lock, self.path)

    def test_invalid_or_unavailable_runtime_fails_without_opening_stdout(self):
        for record in ("", "not-a-pid\n", "0\n", "123\nnot-json\n", "123\n[]\n"):
            with self.subTest(record=record):
                self.lock.write_text(record)
                with patch.object(Path, "stat") as probe, self.assertRaises(RuntimeError):
                    console.resolve_log_source(self.lock, self.path)
                probe.assert_not_called()
        self.lock.write_text("123\n")
        with patch.object(Path, "stat", side_effect=FileNotFoundError), self.assertRaises(RuntimeError):
            console.resolve_log_source(self.lock, self.path)


if __name__ == "__main__":
    unittest.main()
