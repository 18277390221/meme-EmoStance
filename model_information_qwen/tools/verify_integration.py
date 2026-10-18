"""Exercise the real archived model over loopback TCP, with no shared secrets.

The controlled monotonic clock makes the user-arrival timeout deterministic
even on machines where generating a reply itself takes more than ten seconds.
By default each turn uses lightweight role-name rules and one emotion-
conditioned reply call, including an essential clarification. The optional
independent repair mode is audited separately when enabled.
No actual Raspberry Pi connection is implied by this local integration check.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import secrets
import time

from model_information_qwen.model import PROJECT_ROOT, load_chat_engine
from model_information_qwen.pi_client import InferenceClient
from model_information_qwen.protocol import validate_reply
from model_information_qwen.role_names import canonicalize_role_name
from model_information_qwen.server import InferenceServer
from model_information_qwen.sessions import DialogueManager
from communication.common import FRAME_LIMIT, Peer, read_message


async def verify_v2(engine):
    """Check the real Qwen model with the installed player's v2 lifecycle."""
    audited = AuditedEngine(engine)
    clock = ControlledClock()
    manager = DialogueManager(audited, clock=clock, cleanup_interval=None)
    token = secrets.token_urlsafe(32)
    version = Path(engine.config["run_dir"]).name
    server = InferenceServer(audited, token, manager=manager, model_version=version)
    peers, replies, actual_calls = [], [], []
    raw_model = engine.raw_engine.generator.model
    original_generate = raw_model.generate

    def counted_generate(*args, **kwargs):
        assert kwargs.get("num_return_sequences") == 1 and kwargs.get("num_beams") == 1
        actual_calls.append(1)
        return original_generate(*args, **kwargs)

    raw_model.generate = counted_generate

    async def connect():
        reader, writer = await asyncio.open_connection("127.0.0.1", server.addresses[0][1], limit=FRAME_LIMIT)
        peer = Peer(writer)
        peers.append(peer)
        await peer.send({"type":"hello", "name":"qwen-integration-v2", "protocol_version":2,
                         "session_id":"qwen-v2-integration", "token":token})
        hello = await read_message(reader, 5)
        assert hello["type"] == "hello_ok" and hello["model_ready"] is True
        assert hello["model_version"] == version and hello["protocol_version"] == 2
        assert hello["server_instance_id"] == server.server_instance_id
        assert "request_status" in hello["capabilities"] and hello["max_inflight"] == 1
        return reader, peer

    async def control(reader, peer, kind, request_id=None):
        message = {"type":kind}
        if request_id is not None:
            message["id"] = request_id
        await peer.send(message)
        response = await read_message(reader, 5)
        assert response["type"] == "control_ack", response
        return response

    try:
        await server.start("127.0.0.1", 0)
        reader, peer = await connect()
        for index, (speech_at, request_at, playback_at, text) in enumerate((
            (0., 20., 50., "今天终于完成任务了。"),
            (55., 65., 80., "你还记得我刚才完成了什么吗？"),
            (90., 95., 100., "你好，我们重新聊。"),
        )):
            request_id = f"qwen-v2-{index}"
            clock.now = speech_at
            await control(reader, peer, "speech_start")
            clock.now = request_at
            before = len(actual_calls)
            await peer.send({"type":"request", "id":request_id, "text":text, "end":True})
            accepted = await read_message(reader, 5)
            assert accepted == {"type":"accepted", "id":request_id, "status":"processing"}, accepted
            response = await read_message(reader, 300)
            assert response["type"] == "result", response
            result = validate_reply({"reply":response["reply"], "meme_id":response["meme_id"]})
            assert len(actual_calls) == before + 1
            assert "<think>" not in result["reply"] and "</think>" not in result["reply"]
            await peer.send({"type":"request_status", "id":request_id})
            assert await read_message(reader, 5) == response
            assert len(actual_calls) == before + 1, "Cache lookup must not generate again"
            if index == 1:
                await peer.close()
                for _ in range(100):
                    if not any(c.session_id == "qwen-v2-integration" for c in server.clients.values()):
                        break
                    await asyncio.sleep(.01)
                reader, peer = await connect()
                await peer.send({"type":"request_status", "id":request_id})
                assert await read_message(reader, 5) == response
                assert len(actual_calls) == before + 1
            clock.now = playback_at
            await control(reader, peer, "playback_done", request_id)
            replies.append({"speech_start":speech_at, "request_at":request_at,
                "playback_done":playback_at, "request_id":request_id, "result":result,
                "conversation_id":response["conversation_id"], "generate_calls":1})
        lengths = [len(messages) for messages in audited.inputs]
        assert lengths == [1,3,1], lengths
        assert replies[0]["conversation_id"] == replies[1]["conversation_id"]
        assert replies[1]["conversation_id"] != replies[2]["conversation_id"]
        assert len(actual_calls) == 3 and len(audited.retrievals) == 3
        return {"status":"passed", "model_version":version, "model_ready":True,
            "model_input_lengths":lengths, "actual_generate_calls":len(actual_calls),
            "request_status_cache":"passed", "reconnect_without_regeneration":"passed",
            "speech_and_playback_pause_idle":"passed", "idle_reset_seconds":10,
            "turns":replies, "nearest_vector_checks":audited.retrievals}
    finally:
        raw_model.generate = original_generate
        try:
            for peer in peers:
                await peer.close()
            await server.close()
        finally:
            audited.restore()


class ControlledClock:
    now = 0.0

    def __call__(self):
        return self.now


class AuditedEngine:
    """Capture model inputs and independently check the real nearest lookup."""
    def __init__(self, engine):
        self.engine = engine
        self.config = engine.config
        self.inputs = []
        self.effective_users = []
        self.retrievals = []
        self.generations = []
        self.normalizations = []
        self.turn_audits = []
        self._nearest = engine.index.nearest
        self._complete = engine.generator.complete
        self._normalize = engine.normalizer.normalize
        engine.index.nearest = self._audit_nearest
        engine.generator.complete = self._audit_complete
        engine.normalizer.normalize = self._audit_normalize

    def restore(self):
        self.engine.index.nearest = self._nearest
        self.engine.generator.complete = self._complete
        self.engine.normalizer.normalize = self._normalize

    def _audit_complete(self, system_prompt, messages, **options):
        conditioned = options.get("stance_vector") is not None
        if system_prompt == self.engine.settings["asr_prompt"] and not conditioned:
            stage = "asr_repair"
        elif system_prompt == self.engine.settings["role_prompt"] and conditioned:
            stage = "emotion_reply"
        else:
            raise AssertionError("Unexpected generation stage or missing emotion prefix")
        call = {"stage": stage, "emotion_conditioned": conditioned,
                "max_new_tokens": options["max_new_tokens"]}
        self.generations.append(call)
        started = time.monotonic()
        try:
            output = self._complete(system_prompt, messages, **options)
            call["status"] = "completed"
            # Only the final response is compared with the returned reply;
            # the repair model's raw JSON is not a real dialogue turn.
            if stage == "emotion_reply":
                call["reply"] = output
            return output
        except Exception:
            call["status"] = "failed"
            raise
        finally:
            call["elapsed_seconds"] = round(time.monotonic() - started, 3)

    def _audit_normalize(self, messages):
        normalized = self._normalize(messages)
        self.normalizations.append(normalized)
        return normalized

    def _audit_nearest(self, vector):
        import numpy as np
        result = self._nearest(vector)
        index = self.engine.index
        query = np.asarray(vector, dtype=np.float32)
        if index.metric == "cosine":
            distances = 1 - np.clip(index.normalized @ (query / np.linalg.norm(query)), -1, 1)
        else:
            distances = np.linalg.norm(index.vectors - query, axis=1)
        minimum = float(distances.min())
        tied = np.flatnonzero(distances == minimum)
        best = min(tied, key=lambda i: (
            index.anchors[i]["meme_id"], index.anchors[i]["reply"], index.anchors[i]["sample_id"]
        ))
        assert result["meme_id"] == index.anchors[best]["meme_id"]
        assert result["sample_id"] == index.anchors[best]["sample_id"]
        assert abs(result["distance"] - minimum) < 1e-6
        self.retrievals.append({"metric": index.metric, "vector_dim": len(query),
            "distance": minimum, "meme_id": result["meme_id"],
            "sample_id": result["sample_id"]})
        return result

    def reply_turn(self, messages):
        self.inputs.append([dict(message) for message in messages])
        generation_start = len(self.generations)
        normalization_start = len(self.normalizations)
        retrieval_start = len(self.retrievals)
        turn = self.engine.reply_turn(messages)
        assert len(self.normalizations) == normalization_start + 1
        normalized = self.normalizations[-1]
        calls = self.generations[generation_start:]
        repair_calls = [call for call in calls if call["stage"] == "asr_repair"]
        reply_calls = [call for call in calls if call["stage"] == "emotion_reply"]
        repair_enabled = self.engine.normalizer.enabled
        assert len(repair_calls) == int(repair_enabled)
        assert turn.user_text == normalized.text
        if not repair_enabled:
            assert turn.user_text == canonicalize_role_name(messages[-1]["content"], messages[:-1])
            assert normalized.status != "uncertain"
        if repair_enabled and normalized.status == "uncertain":
            assert not reply_calls
            assert turn.reply == normalized.clarification
        else:
            assert len(reply_calls) == 1
            assert turn.reply == reply_calls[0]["reply"]
        assert len(self.retrievals) == retrieval_start + 1
        self.turn_audits.append({"normalization_status": normalized.status,
            "asr_model_calls": len(repair_calls), "reply_model_calls": len(reply_calls),
            "generation_calls": [dict(call) for call in calls]})
        self.effective_users.append(turn.user_text)
        return turn

    def reply(self, messages):
        return self.reply_turn(messages).as_reply()


async def verify(engine):
    audited = AuditedEngine(engine)
    clock = ControlledClock()
    manager = DialogueManager(audited, clock=clock, cleanup_interval=None)
    token = secrets.token_urlsafe(32)
    server = InferenceServer(audited, token, manager=manager)
    client = None
    turns = []
    try:
        await server.start("127.0.0.1", 0)
        client = InferenceClient("127.0.0.1", server.addresses[0][1], token=token)
        await client.start()
        for received_at, text in (
            (0.0, "咪咪猫，我今天不开心，鸭力很大。"),
            (4.0, "谢谢你听我说，我慢慢好一点了。"),
            (14.0, "我今天有点难过，不太想聊天。"),
        ):
            clock.now = received_at
            started = time.monotonic()
            result = validate_reply(await client.request(text, timeout=300))
            turns.append({"received_at": received_at, "user": text,
                "result": result, "elapsed_seconds": round(time.monotonic() - started, 3),
                "model_calls": audited.turn_audits[-1]})
        assert [len(messages) for messages in audited.inputs] == [1, 3, 1]
        assert [m["role"] for m in audited.inputs[1]] == ["user", "assistant", "user"]
        assert audited.inputs[1][0]["content"] == audited.effective_users[0]
        assert audited.inputs[1][1]["content"] == turns[0]["result"]["reply"]
        assert audited.inputs[2] == [{"role": "user", "content": turns[2]["user"]}]
        assert len(audited.retrievals) == len(turns)
        assert [row["meme_id"] for row in audited.retrievals] == [row["result"]["meme_id"] for row in turns]
        assert len(manager.history(client.session_id)) == 2
        if not engine.normalizer.enabled:
            assert audited.effective_users[0] == "meme，我今天不开心，鸭力很大。"
        return {"status": "passed", "transport": "localhost TCP",
            "reply_strategy": "single_emotion_conditioned_generation",
            "independent_repair_enabled": bool(engine.normalizer.enabled),
            "model_dir": str(Path(engine.config["run_dir"]).resolve()),
            "controlled_arrival_seconds": [0, 4, 14],
            "model_input_lengths": [len(m) for m in audited.inputs],
            "effective_user_texts": audited.effective_users,
            "history_roles": [[m["role"] for m in messages] for messages in audited.inputs],
            "turns": turns, "nearest_vector_checks": audited.retrievals}
    finally:
        try:
            if client is not None:
                await client.close()
            await server.close()
        finally:
            audited.restore()


def main():
    parser = argparse.ArgumentParser(description="本机 TCP 与真实归档模型联调")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path,
        default=PROJECT_ROOT / "model_information_qwen/runs/integration_chat_smoke.json")
    args = parser.parse_args()
    engine = load_chat_engine(args.model_dir, config_path=args.config)
    from model_information_qwen import dialogue_log
    journal = dialogue_log.DEFAULT_LOG_PATH
    try:
        dialogue_log.DEFAULT_LOG_PATH = args.output.parent / "integration_conversation.log"
        result = asyncio.run(verify(engine))
        result["protocol_v2"] = asyncio.run(verify_v2(engine))
    finally:
        dialogue_log.DEFAULT_LOG_PATH = journal
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "model_input_lengths": result["model_input_lengths"],
        "replies": [turn["result"] for turn in result["turns"]],
        "report": str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
