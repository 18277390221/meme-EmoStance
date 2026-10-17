"""Single-pass dialogue, lightweight role names and optional ASR repair."""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

from communication.common import log
from .generation import LocalChatGenerator
from .normalization import ASRNormalizer
from .protocol import validate_reply, validate_text
from .turns import TurnResult


DEFAULT_CONFIG = Path(__file__).with_name("chat_config.json")


def load_chat_settings(path=None):
    path = Path(path) if path is not None else DEFAULT_CONFIG
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("chat_config.json 必须是 JSON 对象")
    values.setdefault("role_prompt_enabled", True)
    for key in ("max_prompt_tokens", "max_reply_tokens", "asr_max_new_tokens"):
        if isinstance(values.get(key), bool) or not isinstance(values.get(key), int) or values[key] < 1:
            raise ValueError(f"chat_config.json 的 {key} 必须是正整数")
    for key in ("asr_history_messages", "no_repeat_ngram_size"):
        if isinstance(values.get(key), bool) or not isinstance(values.get(key), int) or values[key] < 0:
            raise ValueError(f"chat_config.json 的 {key} 必须是非负整数")
    for key in ("asr_enabled", "do_sample", "role_prompt_enabled"):
        if not isinstance(values.get(key), bool):
            raise ValueError(f"chat_config.json 的 {key} 必须是布尔值")
    for key in ("temperature", "top_p"):
        if isinstance(values.get(key), bool) or not isinstance(values.get(key), (int, float)) or not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f"chat_config.json 的 {key} 必须是有限正数")
    if values["top_p"] > 1:
        raise ValueError("chat_config.json 的 top_p 不能大于 1")
    for key in ("role_prompt", "asr_prompt"):
        if key == "role_prompt" and not values["role_prompt_enabled"]:
            values[key] = ""
            continue
        filename = values.get(key)
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"chat_config.json 缺少 {key}")
        prompt_file = Path(filename).expanduser()
        prompt_file = prompt_file if prompt_file.is_absolute() else path.resolve().parent / prompt_file
        text = prompt_file.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"提示词文件为空：{prompt_file}")
        values[key] = text
    return values


class EmotionalChatEngine:
    """Share loaded weights; planner/retrieval only see real dialogue turns."""
    def __init__(self, raw_engine, settings=None, *, generator=None):
        self.raw_engine = raw_engine
        self.config = raw_engine.config
        self.planner = raw_engine.planner
        self.index = raw_engine.index
        self.settings = load_chat_settings() if settings is None else dict(settings)
        if not self.settings.get("role_prompt_enabled", True):
            self.settings["role_prompt"] = ""
        self.generator = generator if generator is not None else LocalChatGenerator(
            raw_engine.generator, max_prompt_tokens=self.settings["max_prompt_tokens"])
        self.normalizer = ASRNormalizer(self.generator, self.settings["asr_prompt"],
            max_new_tokens=self.settings["asr_max_new_tokens"],
            history_messages=self.settings["asr_history_messages"], enabled=self.settings["asr_enabled"])
        # We implement the stages ourselves; never call raw_engine.reply while
        # holding its non-reentrant lock.
        self._lock = getattr(raw_engine, "_lock", threading.Lock())

    @staticmethod
    def _messages(messages):
        if not isinstance(messages, list) or not messages or len(messages) % 2 != 1:
            raise ValueError("对话必须按 user/assistant 交替，并以当前 user 发言结束")
        copied = []
        for index, message in enumerate(messages):
            if message.get("role") != ("user" if index % 2 == 0 else "assistant"):
                raise ValueError("只允许真实 user/assistant 对话，不得在历史中加入 system 指令")
            copied.append({"role": message["role"], "content": validate_text(message.get("content"))})
        return copied

    def _nearest_reply(self, history, reply):
        realized = self.planner.predict(history + [{"role": "B", "text": reply}], "A")
        nearest = self.index.nearest(realized["source_realized_vector"])
        return realized, nearest

    def reply_turn(self, messages):
        messages = self._messages(messages)
        with self._lock:
            original = messages[-1]["content"]
            prepared = self.normalizer.normalize(messages)
            log("[输入规范化] " + json.dumps({"original": original, "normalized": prepared.text,
                "status": prepared.status, "reason": prepared.reason}, ensure_ascii=False))
            messages[-1]["content"] = prepared.text
            history = [{"role": "A" if m["role"] == "user" else "B", "text": m["content"]} for m in messages]
            if prepared.status == "uncertain":
                # Ask the editor's question instead of treating a guessed
                # interpretation as fact. It still uses the same meme index.
                reply = prepared.clarification
            else:
                plan = self.planner.predict(history, "B")
                # Generate the final response once, with the optional role
                # prompt and the trained prefix. No candidate pool, ranking
                # or second style-generation pass is used.
                reply = self.generator.complete(self.settings["role_prompt"], messages,
                    max_new_tokens=self.settings["max_reply_tokens"], stance_vector=plan["target_vector"],
                    do_sample=self.settings["do_sample"], temperature=self.settings["temperature"],
                    top_p=self.settings["top_p"], no_repeat_ngram_size=self.settings["no_repeat_ngram_size"])
            # Retrieve against the emotion vector of the actual returned text,
            # including an essential clarification supplied by ASR repair.
            _, nearest = self._nearest_reply(history, reply)
            result = validate_reply({"reply": reply, "meme_id": nearest["meme_id"]})
            if len(result["reply"]) > 2000:
                raise RuntimeError("生成回复超过树莓派播放器 2000 字符限制")
            return TurnResult(prepared.text, result["reply"], result["meme_id"])

    def reply(self, messages):
        """Convenience API; the session manager calls reply_turn to commit text."""
        return self.reply_turn(messages).as_reply()
