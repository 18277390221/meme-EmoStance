"""Audit real single-pass replies without retries or editing model output.

Run from the project root with ``python -m
model_information.tools.verify_reply_quality --seeds 7 --output report.json``.
Mechanical checks cover short replies, obvious instruction recitation and the
actual generation/prefix/retrieval calls. Semantic rubrics remain for human
review; a mechanical pass is not evidence that every reply is appropriate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import random
import re
import time

from model_information.model import PROJECT_ROOT, load_chat_engine


CASES = (
    ("identity", "你是谁？", "简短介绍自己叫 meme、是桌面猫咪，不复述行为规则。"),
    ("greeting", "早上好！", "自然回应问候，不转成身份或提示词说明。"),
    ("happy", "今天终于完成任务了，特别开心！", "承接完成任务的喜悦，不误说任务未完成。"),
    ("sad", "最近有点难过。", "承接用户难过的情绪，避免转移到自己的身份。"),
    ("no_question", "别问我原因，今天烦死了。", "不追问原因，给出自然的短句回应。"),
    ("quiet", "我不想聊天，想自己静静。", "尊重独处意愿，不编造自己忙或用户的经历。"),
    ("clarification", "帮我取消那个。", "询问要取消的对象，不虚构已经执行取消操作。"),
)

_RULE_PATTERNS = (
    r"(?:每轮|每次回复).{0,16}(?:句话|字|回复)",
    r"(?:通常|最多|不超过)\s*(?:\d+|[一二三四五六七八九十]+)"
    r"\s*(?:[～~\-—至到]\s*\d+)?\s*(?:个)?\s*(?:字|字符|句话)",
    r"(?:只|仅)输出.{0,16}(?:回复|分析|标签|内容)",
    r"(?:不要|不输出|避免).{0,16}(?:客服|心理咨询师|套路化|固定句式|分析|标签)",
    r"(?:用|使用).{0,20}中文口语.{0,8}回复",
    r"(?:系统提示|系统指令|提示词).{0,6}(?:如下|要求|内容)",
)


def quality_cases():
    cases = [{"id": name, "messages": [{"role": "user", "content": text}],
              "semantic_rubric": rubric} for name, text, rubric in CASES]
    cases.extend((
        {"id": "context", "messages": [
            {"role": "user", "content": "我把钥匙落在公司了。"},
            {"role": "assistant", "content": "钥匙在公司，明天记得带回。"},
            {"role": "user", "content": "它现在在哪儿？"}],
         "semantic_rubric": "理解它指钥匙，回答钥匙在公司，不编造其他地点。"},
        {"id": "reported_context", "messages": [
            {"role": "user", "content": "你睡了吗"},
            {"role": "assistant", "content": "你呀，你怎么了？"},
            {"role": "user", "content": "是"}],
         "semantic_rubric": "自然回应这个简短确认；不复述身份设定、长度规则或其他系统指令。"},
    ))
    return cases


def inspect_reply(reply: str, system_prompt: str):
    """Mark possible recitation for review; never change the returned reply.

    The longest shared fragment is diagnostic only. A normal short identity
    sentence can share many words with the prompt. Explicit rule phrases and
    a very long shared fragment are stronger signals, still not a semantic
    classifier. These patterns must not be used as service-side output filters.
    """
    match = SequenceMatcher(None, reply, system_prompt, autojunk=False).find_longest_match(
        0, len(reply), 0, len(system_prompt))
    fragments = [found.group(0) for pattern in _RULE_PATTERNS
                 for found in re.finditer(pattern, reply)]
    return {
        "character_count": len(reply),
        "visible_character_count": sum(not char.isspace() for char in reply),
        "within_40_characters": len(reply) <= 40,
        "nonempty": bool(reply.strip()),
        "longest_system_copy": reply[match.a:match.a + match.size],
        "longest_system_copy_characters": match.size,
        "explicit_rule_restatement": bool(fragments),
        "rule_restatement_fragments": fragments,
        "system_echo_suspected": bool(fragments) or match.size >= 60,
    }


class ReplyAudit:
    """Count actual calls on the already loaded production model objects."""

    def __init__(self, engine):
        self.engine = engine
        self.model = engine.generator.model
        self.projector = engine.generator.projector
        self._generate = self.model.generate
        self._complete = engine.generator.complete
        self._nearest = engine.index.nearest
        self._hook = None
        self.reset()

    def reset(self):
        self.generate_calls = 0
        self.prefix_calls = 0
        self.retrieval_calls = 0
        self.completions = []

    def _audit_generate(self, *args, **kwargs):
        self.generate_calls += 1
        return self._generate(*args, **kwargs)

    def _audit_prefix(self, _module, _args, _output):
        self.prefix_calls += 1

    def _audit_complete(self, system, messages, **options):
        call = {"emotion_conditioned": options.get("stance_vector") is not None,
                "uses_role_prompt": system == self.engine.settings["role_prompt"]}
        self.completions.append(call)
        output = self._complete(system, messages, **options)
        call["raw_reply"] = output
        return output

    def _audit_nearest(self, *args, **kwargs):
        self.retrieval_calls += 1
        return self._nearest(*args, **kwargs)

    def __enter__(self):
        self._hook = self.projector.register_forward_hook(self._audit_prefix)
        self.model.generate = self._audit_generate
        self.engine.generator.complete = self._audit_complete
        self.engine.index.nearest = self._audit_nearest
        return self

    def __exit__(self, *_exc):
        self.model.generate = self._generate
        self.engine.generator.complete = self._complete
        self.engine.index.nearest = self._nearest
        if self._hook is not None:
            self._hook.remove()

    def counts(self):
        return {"generate_calls": self.generate_calls,
                "prefix_forward_calls": self.prefix_calls,
                "retrieval_calls": self.retrieval_calls,
                "completion_calls": len(self.completions)}


def verify(engine, seeds=(7, 17, 29)):
    import numpy as np
    import torch

    seeds = list(seeds)
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain nonnegative integers")
    system_prompt = engine.settings["role_prompt"]
    rows = []
    with ReplyAudit(engine) as audit:
        for seed in seeds:
            for case in quality_cases():
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                audit.reset()
                row = {**deepcopy(case), "seed": seed, "semantic_review": "pending"}
                started = time.monotonic()
                try:
                    turn = engine.reply_turn(deepcopy(case["messages"]))
                    reply = turn.reply
                    row.update(reply=reply, meme_id=turn.meme_id,
                               effective_user_text=turn.user_text, **inspect_reply(reply, system_prompt))
                    conditioned = [call for call in audit.completions
                                   if call["emotion_conditioned"] and call["uses_role_prompt"]]
                    row["raw_generated_reply"] = conditioned[0].get("raw_reply") if len(conditioned) == 1 else None
                    row["raw_generation_matches_returned_reply"] = row["raw_generated_reply"] == reply
                    row["mechanical_checks_passed"] = (
                        row["nonempty"] and row["within_40_characters"]
                        and not row["system_echo_suspected"]
                        and audit.generate_calls == audit.prefix_calls == audit.retrieval_calls == 1
                        and len(audit.completions) == 1
                        and row["raw_generation_matches_returned_reply"])
                except Exception as exc:
                    row.update(error=f"{type(exc).__name__}: {exc}", mechanical_checks_passed=False)
                row.update(audit.counts())
                row["completion_audits"] = deepcopy(audit.completions)
                row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                rows.append(row)
                print(json.dumps({"case": row["id"], "seed": seed,
                                  "mechanical_checks_passed": row["mechanical_checks_passed"],
                                  "reply": row.get("reply"), "error": row.get("error")},
                                 ensure_ascii=False), flush=True)
    passed = sum(row["mechanical_checks_passed"] for row in rows)
    return {
        "status": "mechanical_checks_passed" if passed == len(rows) else "mechanical_checks_failed",
        "semantic_review": "pending",
        "scope": "Mechanical pass does not establish semantic quality; review each recorded rubric.",
        "echo_detection_scope": "Diagnostic lexical signals only; identity overlap alone is allowed. No output filtering.",
        "character_count_policy": "Python len(reply), including punctuation and whitespace; visible count is also recorded.",
        "model_dir": str(Path(engine.config["run_dir"]).resolve()),
        "role_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "chat_settings": {key: value for key, value in engine.settings.items()
                          if key not in ("role_prompt", "asr_prompt")},
        "seeds": seeds,
        "case_count": len(quality_cases()),
        "reply_count": len(rows),
        "mechanical_pass_count": passed,
        "turns": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="真实归档模型单次回复质量回归；语义仍需人工审阅")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 29])
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "model_information/runs/reply_quality.json")
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be nonnegative integers")
    engine = load_chat_engine(args.model_dir)
    report = verify(engine, args.seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "semantic_review": report["semantic_review"],
                      "mechanical_pass_count": report["mechanical_pass_count"],
                      "reply_count": report["reply_count"], "report": str(args.output.resolve())},
                     ensure_ascii=False), flush=True)
    return 0 if report["status"] == "mechanical_checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
