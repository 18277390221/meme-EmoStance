"""Replay user turns through the deployed chat engine and retain its own replies.

This is a local, read-only model check; it sends no messages to the player.
The resulting transcript needs semantic review and is not a quality certificate.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import time

from model_information_qwen.model import load_chat_engine
from model_information_qwen.server import DEFAULT_MAX_TURNS
from model_information_qwen.sessions import BACKGROUND_ROLL_INTERVAL, DialogueManager


def expected_rollout_layout(turns, max_turns=None):
    """Derive technical rollover expectations from actual successful replies.

    The replay clock is fixed and no manual reset is requested, so only two
    adjacent identical replies start a new context. A finite history window
    slides after successful commits while keeping the same conversation.
    Reply semantics never enter this calculation.
    """
    if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
        raise ValueError("max_turns must be a positive integer or None")
    committed, recent, group, slides = [], [], 0, 0
    lengths, groups, reset_reasons, revisions, slide_counts = [], [], [], [], []
    for index, turn in enumerate(turns):
        reasons = []
        if len(recent) == 2 and recent[0] == recent[1]:
            reasons.append("repeated_reply")
        if reasons:
            committed = []
            recent = []
            slides = 0
            group += 1
        lengths.append(2 * len(committed) + 1)
        groups.append(group)
        reset_reasons.append(reasons if index else ["initial"])
        committed.append(turn["reply"])
        if max_turns is not None and len(committed) > max_turns:
            removed = len(committed) - max_turns
            del committed[:removed]
            slides += removed
        recent = [*recent[-1:], turn["reply"]]
        slide_counts.append(slides)
        revisions.append(slides // BACKGROUND_ROLL_INTERVAL)
    return {"input_messages": lengths, "conversation_groups": groups,
            "reset_reasons": reset_reasons, "window_slides": slide_counts,
            "background_revisions": revisions}


def validate_rollout_layout(turns, max_turns=None):
    """Verify observed history/IDs rather than assuming a fixed modulo cycle."""
    expected = expected_rollout_layout(turns, max_turns)
    assert [turn["input_messages"] for turn in turns] == expected["input_messages"], "Unexpected retained history lengths"
    assert [turn["background_revision"] for turn in turns] == expected["background_revisions"], "Unexpected background revisions"
    group_ids = {}
    for turn, group in zip(turns, expected["conversation_groups"]):
        identifier = turn["conversation_id"]
        if group in group_ids:
            assert identifier == group_ids[group], "Unexpected conversation reset"
        else:
            assert identifier not in group_ids.values(), "Expected rollover did not create a new conversation"
            group_ids[group] = identifier
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output already exists; preserve previous results")
    import numpy as np
    import torch
    torch.set_num_threads(8)
    engine = load_chat_engine()
    raw = engine.raw_engine.generator.model
    original_generate = raw.generate
    original_reply_turn = engine.reply_turn
    calls, input_lengths = [], []

    def counted_generate(*positional, **keywords):
        assert keywords.get("num_return_sequences") == 1
        assert keywords.get("num_beams") == 1
        calls.append(1)
        return original_generate(*positional, **keywords)

    raw.generate = counted_generate

    def audited_reply_turn(messages):
        input_lengths.append(len(messages))
        return original_reply_turn(messages)

    engine.reply_turn = audited_reply_turn
    users = ["我昨天晚上熬夜了，现在有点难受。", "对", "是的，我熬夜了。",
             "小猫，橙子和香蕉你喜欢哪个？", "你好", "你好，我们重新聊。",
             "你叫什么名字？", "你平时喜欢做什么？",
             "新一轮，你还记得我刚才问了什么吗？", "那你现在想聊什么？"]
    report = {"status": "running", "scope": "autoregressive_dialogue_regression_only",
              "model_dir": engine.config["run_dir"], "seeds": args.seeds,
              "system_prompt": engine.settings["role_prompt"],
              "dialogue_policy": engine.dialogue_policy_state,
              "max_turns": DEFAULT_MAX_TURNS, "background_roll_interval": BACKGROUND_ROLL_INTERVAL, "rollouts": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    async def exercise():
        manager = DialogueManager(engine, max_turns=DEFAULT_MAX_TURNS, clock=lambda: 0., cleanup_interval=None)
        manager.start()
        try:
            for seed in args.seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                session, turns = f"rollout-{seed}", []
                rollout = {"seed": seed, "turns": turns}
                report["rollouts"].append(rollout)
                for text in users:
                    before = len(calls)
                    started = time.monotonic()
                    future = manager.accept(session, text)
                    result = await future
                    assert len(calls) == before+1, "Each turn must call generate exactly once"
                    entry = {"user": text, **result, "conversation_id": future.conversation_id,
                             "background_revision": future.background_revision,
                             "generate_calls": 1, "input_messages": input_lengths[-1],
                             "elapsed_seconds": round(time.monotonic()-started, 4)}
                    turns.append(entry)
                    print(json.dumps({"seed": seed, **entry}, ensure_ascii=False), flush=True)
                    save()
                rollout["expected_layout"] = validate_rollout_layout(turns, DEFAULT_MAX_TURNS)
        finally:
            await manager.close()

    try:
        asyncio.run(exercise())
        report["status"] = "complete"
        report["actual_generate_calls"] = len(calls)
        save()
    finally:
        raw.generate = original_generate
        engine.reply_turn = original_reply_turn


if __name__ == "__main__":
    main()
