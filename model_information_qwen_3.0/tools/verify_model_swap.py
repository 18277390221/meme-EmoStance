"""Verify V3 checkpoint identity and the unchanged server flow on localhost.

This checks transport, context, retrieval and actual generation calls. It
does not score reply quality or change which loss-selected checkpoint is used.
No Raspberry Pi request, playback or software restart is performed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import secrets

from communication.common import FRAME_LIMIT, Peer, read_message
from .. import dialogue_log
from ..model import load_chat_engine, inspect_model_version
from ..protocol import validate_reply
from ..server import InferenceServer, DEFAULT_MAX_TURNS
from ..sessions import BACKGROUND_ROLL_INTERVAL
from .verify_integration import AuditedEngine, verify, verify_v2


async def verify_default_turn_limit(engine):
    audited = AuditedEngine(engine)
    token = secrets.token_urlsafe(32)
    server = InferenceServer(audited, token, model_version=engine.model_version)
    raw = engine.raw_engine.generator.model
    original = raw.generate
    calls, turns = [], []
    peer = None

    def generate(*args, **kwargs):
        assert kwargs.get("num_return_sequences") == 1 and kwargs.get("num_beams") == 1
        calls.append(1)
        return original(*args, **kwargs)

    raw.generate = generate
    try:
        await server.start("127.0.0.1", 0)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.addresses[0][1], limit=FRAME_LIMIT)
        peer = Peer(writer)
        await peer.send(dict(type="hello", name="qwen-v3-default-limit", protocol_version=2,
                             session_id="qwen-v3-default-limit", token=token))
        hello = await read_message(reader, 5)
        assert hello["type"] == "hello_ok" and hello["model_ready"] is True
        assert hello["model_version"] == engine.model_version
        assert hello["max_turns"] == DEFAULT_MAX_TURNS
        assert hello["context_window_turns"] == DEFAULT_MAX_TURNS
        assert hello["background_roll_interval"] == BACKGROUND_ROLL_INTERVAL
        assert "background_revision" in hello["capabilities"]
        users = ["你好。", "我今天完成了项目。", "你记得我完成了什么吗？",
                 "我现在想吃香蕉。", "对。", "谢谢你。", "我们继续聊。",
                 "今天我看到一只橘猫。", "它在晒太阳。", "你喜欢晴天吗？",
                 "我准备去公园散步。", "公园里有很多树。", "树叶开始变黄了。",
                 "秋天的风很舒服。", "你记得我准备去哪里吗？", "我们继续聊秋天吧。"]
        active_turns, active_replies, resets = 0, [], []
        for index, text in enumerate(users):
            request_id = f"v3-limit-{index}"
            repeated = len(active_replies) >= 2 and active_replies[-1] == active_replies[-2]
            should_reset = repeated
            if should_reset:
                resets.append(dict(turn=index+1, repeated_reply=True))
                active_turns, active_replies = 0, []
            retained = active_turns if DEFAULT_MAX_TURNS is None else min(active_turns, DEFAULT_MAX_TURNS)
            expected_input_length = retained*2+1
            before = len(calls)
            await peer.send(dict(type="request", id=request_id, text=text, end=True))
            accepted = await read_message(reader, 5)
            assert accepted == dict(type="accepted", id=request_id, status="processing")
            response = await read_message(reader, 300)
            assert response["type"] == "result" and response["id"] == request_id
            validate_reply({"reply":response["reply"], "meme_id":response["meme_id"]})
            assert len(calls) == before+1
            assert len(audited.inputs[-1]) == expected_input_length
            expected_revision = (max(0, active_turns + 1 - DEFAULT_MAX_TURNS)//BACKGROUND_ROLL_INTERVAL
                                 if DEFAULT_MAX_TURNS is not None else 0)
            assert response["background_revision"] == expected_revision
            assert len(server.manager.history("qwen-v3-default-limit")) <= 2 * (DEFAULT_MAX_TURNS or len(users))
            if turns:
                changed = turns[-1]["conversation_id"] != response["conversation_id"]
                assert changed == should_reset
            await peer.send(dict(type="request_status", id=request_id))
            assert await read_message(reader, 5) == response
            assert len(calls) == before+1
            await peer.send(dict(type="playback_done", id=request_id))
            assert (await read_message(reader, 5))["type"] == "control_ack"
            turns.append(dict(user=text, reply=response["reply"], meme_id=response["meme_id"],
                              conversation_id=response["conversation_id"],
                              background_revision=response["background_revision"], generate_calls=1))
            active_turns += 1
            active_replies.append(response["reply"])
        lengths = [len(messages) for messages in audited.inputs]
        assert lengths[0] == 1, lengths
        if DEFAULT_MAX_TURNS is None:
            assert all(reset["repeated_reply"] for reset in resets)
        else:
            assert max(lengths) <= DEFAULT_MAX_TURNS*2+1
        assert len(calls) == len(users) == len(audited.retrievals)
        return dict(status="passed", model_version=engine.model_version, max_turns=DEFAULT_MAX_TURNS,
                    model_input_lengths=lengths, actual_generate_calls=len(calls),
                    request_status_without_regeneration="passed", resets=resets, turns=turns,
                    nearest_vector_checks=audited.retrievals)
    finally:
        raw.generate = original
        try:
            if peer is not None:
                await peer.close()
            await server.close()
        finally:
            audited.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"runs/model_swap_verification.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve earlier verification evidence")
    import torch
    torch.set_num_threads(8)
    info = inspect_model_version()
    engine = load_chat_engine()
    state = engine.raw_engine.generator.checkpoint_state
    assert info["version"] == engine.model_version == "3.0.0"
    assert info["base_model_version"] == "qwen3_8b_v3_loss_ce"
    assert state["global_step"] == 150000 and state["model_version"] == "3.0.0"
    assert state["selection_criterion"] == "dev_response_ce"
    current = engine.raw_engine.generator.projector.state_dict()
    assert current.keys() == state["projector"].keys()
    assert all(torch.equal(value.detach().cpu(), state["projector"][key]) for key,value in current.items())
    assert engine.settings["max_prompt_tokens"] == 4096 and engine.settings["max_reply_tokens"] == 128
    assert engine.settings["asr_enabled"] is False
    report = dict(status="running", scope="checkpoint_identity_and_server_pipeline_only",
                  model=info, actual_loaded_step=state["global_step"],
                  loaded_prefix_parameters_equal_checkpoint=True, chat_quality_evaluated=False,
                  production_endpoint_modified=False, pi_state_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    journal = dialogue_log.DEFAULT_LOG_PATH
    try:
        dialogue_log.DEFAULT_LOG_PATH = args.output.parent/"model_swap_conversation.log"
        report["protocol_v1"] = asyncio.run(verify(engine))
        report["protocol_v2"] = asyncio.run(verify_v2(engine))
        report["default_context_limit"] = asyncio.run(verify_default_turn_limit(engine))
        report["status"] = "passed"
        report["actual_generate_calls"] = 13
    finally:
        dialogue_log.DEFAULT_LOG_PATH = journal
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":report["status"], "version":engine.model_version,
                      "actual_loaded_step":state["global_step"], "actual_generate_calls":13,
                      "default_context_input_lengths":report["default_context_limit"]["model_input_lengths"],
                      "report":str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
