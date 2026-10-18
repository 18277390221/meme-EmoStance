"""Journal only complete user/reply lines, without affecting model results."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_information_qwen.dialogue_log import append_dialogue, is_dialogue_line
from model_information_qwen.server import log_message


def make_line(direction="树莓派 → 服务器", **payload):
    detail = {"device": "meme-player", "request_id": "test-id", **payload}
    return f"[2026-10-18T12:34:56+08:00] [{direction}] {json.dumps(detail, ensure_ascii=False)}"


class DialogueLogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "runs" / "conversation.log"

    def test_accepts_chinese_user_reply_and_business_error_lines(self):
        for line in (
            make_line(text="你睡了吗 😺"),
            make_line(text="你好", end=True),
            make_line(text="你好", status="rejected"),
            make_line("服务器 → 树莓派", reply="还没呢，陪你聊会儿。", meme_id="meme-0001"),
            make_line("服务器 → 树莓派", code="handler_error", message="请重试"),
        ):
            with self.subTest(line=line):
                self.assertTrue(is_dialogue_line(line))
                self.assertTrue(is_dialogue_line(line + "\n"))
                self.assertTrue(is_dialogue_line(line + "\r\n"))

    def test_rejects_authentication_transport_diagnostics_and_injected_lines(self):
        valid = make_line(text="你好")
        for line in (
            '{"type":"hello","token":"do-not-print"}',
            '[已连接] meme-player / abcd1234',
            '[输入规范化] {"original":"你好"}',
            make_line("推理失败", error="RuntimeError"),
            make_line(text="你好", token="do-not-print"),
            make_line(text="你好", type="hello"),
            make_line("服务器 → 树莓派", type="pong"),
            make_line("服务器 → 树莓派", message="缺少业务错误编号"),
            make_line("服务器 → 树莓派", reply="缺少素材编号"),
            make_line(text="你好", request_id=None),
            make_line(text="你好", request_id=""),
            make_line(text="你好", device=None),
            valid.replace("2026-10-18T12:34:56+08:00", "invalid-time"),
            "debug-prefix " + valid,
            valid + "\n" + valid,
            valid + "\n\n",
            valid.replace('"text": "你好"', '"text": "你好\n第二行"'),
            None,
        ):
            with self.subTest(line=line):
                self.assertFalse(is_dialogue_line(line))

    def test_append_preserves_chinese_and_json_escaped_newlines_in_single_record(self):
        text = "你好\n第二行\t😺"
        line = make_line(text=text)
        self.assertTrue(append_dialogue(line, self.path))
        self.assertTrue(append_dialogue(line + "\n", self.path))
        lines = self.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, [line, line])
        self.assertEqual(json.loads(lines[0].split("] ", 2)[2])["text"], text)
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)

    def test_existing_file_permissions_become_private(self):
        self.path.parent.mkdir()
        self.path.write_text("", encoding="utf-8")
        self.path.chmod(0o644)
        self.assertTrue(append_dialogue(make_line(text="你好"), self.path))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_concurrent_writer_threads_keep_each_unicode_record_complete(self):
        lines = [make_line(text=f"第 {index} 句话，猫咪 😺", request_id=str(index)) for index in range(100)]
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda line: append_dialogue(line, self.path), lines))
        self.assertTrue(all(results))
        self.assertCountEqual(self.path.read_text(encoding="utf-8").splitlines(), lines)

    def test_partial_write_completes_record(self):
        original = os.write

        def short_write(fd, data):
            return original(fd, data[:7])

        line = make_line(text="中文在部分写入后仍完整 😺")
        with patch("model_information_qwen.dialogue_log.os.write", side_effect=short_write):
            self.assertTrue(append_dialogue(line, self.path))
        self.assertEqual(self.path.read_text(encoding="utf-8"), line + "\n")

    def test_diagnostics_are_not_written_and_journal_failures_do_not_raise(self):
        self.assertFalse(append_dialogue("[已连接] meme-player", self.path))
        self.assertFalse(self.path.exists())
        with patch("model_information_qwen.dialogue_log.os.open", side_effect=PermissionError("private-path")):
            self.assertFalse(append_dialogue(make_line(text="你好"), self.path))
        with patch("model_information_qwen.dialogue_log.os.write", side_effect=OSError("private-device")):
            self.assertFalse(append_dialogue(make_line(text="你好"), self.path))

    def test_symlink_destination_does_not_change_other_file(self):
        self.path.parent.mkdir()
        other = self.path.parent / "other.txt"
        other.write_text("untouched", encoding="utf-8")
        other.chmod(0o644)
        self.path.symlink_to(other)
        self.assertFalse(append_dialogue(make_line(text="你好"), self.path))
        self.assertEqual(other.read_text(encoding="utf-8"), "untouched")
        self.assertEqual(other.stat().st_mode & 0o777, 0o644)

    def test_server_logs_same_business_line_to_console_and_private_journal(self):
        logs = []
        with patch("model_information_qwen.server.log", side_effect=logs.append), patch(
            "model_information_qwen.dialogue_log.DEFAULT_LOG_PATH", self.path
        ):
            log_message("树莓派 → 服务器", "meme-player", "one", {"text": "你好\n第二行"})
            log_message("服务器 → 树莓派", "meme-player", "one", {"reply": "喵，你好", "meme_id": "meme-0001"})
            log_message("服务器 → 树莓派", "meme-player", "two", {"code": "busy", "message": "稍后重试"})
            log_message("推理失败", "meme-player", "three", {"error": "RuntimeError"})
        self.assertEqual(len(logs), 4)
        self.assertEqual(self.path.read_text(encoding="utf-8").splitlines(), logs[:3])

    def test_server_continues_and_reports_only_fixed_warning_when_journal_unwritable(self):
        logs = []
        with patch("model_information_qwen.server.log", side_effect=logs.append), patch(
            "model_information_qwen.dialogue_log.DEFAULT_LOG_PATH", self.path
        ), patch("model_information_qwen.dialogue_log.os.open", side_effect=OSError("sensitive-error-detail")):
            result = log_message("服务器 → 树莓派", "meme-player", "one", {"reply": "正常回复", "meme_id": "meme-0001"})
        self.assertIsNone(result)
        self.assertEqual(len(logs), 2)
        self.assertTrue(is_dialogue_line(logs[0]))
        self.assertEqual(logs[1], "[对话日志] 写入失败，当前对话继续处理")
        self.assertNotIn("sensitive-error-detail", "\n".join(logs))


if __name__ == "__main__":
    unittest.main()
