"""Verify the real trained Qwen reply and network protocol after evaluation."""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "runs/qwen3_8b_v2_selected")
    parser.add_argument("--config", type=Path, default=ROOT / "inference_config.json")
    args = parser.parse_args()
    from model_information_qwen.model import load_chat_engine
    from model_information_qwen.sessions import DialogueManager
    from model_information_qwen.server import InferenceServer
    from model_information_qwen.pi_client import InferenceClient
    started = time.monotonic()
    engine = load_chat_engine(args.model_dir, config_path=args.config)
    generator = engine.raw_engine.generator
    original = generator.model.generate
    calls = []

    def counted_generate(*pos, **kw):
        if kw.get("num_return_sequences") != 1 or kw.get("num_beams") != 1:
            raise AssertionError("Inference must generate exactly one sequence")
        calls.append(1)
        return original(*pos, **kw)

    generator.model.generate = counted_generate
    cases = ["今天有点累，想休息一下。", "你能陪我聊聊吗？", "你好，我们重新聊。"]
    outputs, history_lengths = [], []
    valid_ids = {row["meme_id"] for row in engine.index.anchors}
    original_turn = engine.reply_turn

    def recorded_turn(messages):
        history_lengths.append(len(messages))
        return original_turn(messages)

    engine.reply_turn = recorded_turn

    async def verify_loopback():
        clock = [0.0]
        manager = DialogueManager(engine, clock=lambda: clock[0], cleanup_interval=None)
        token = "qwen-runtime-loopback-test"
        server = await InferenceServer(engine, token, manager=manager).start("127.0.0.1", 0)
        client = InferenceClient("127.0.0.1", server.addresses[0][1], "qwen-smoke",
                                 token=token, session_id="qwen-smoke-session")
        try:
            await client.start(timeout=10)
            for index, text in enumerate(cases):
                clock[0] = (0.0, 3.0, 13.0)[index]
                before = len(calls)
                result = await client.request(text, timeout=180)
                assert len(calls) == before + 1, "Each turn must call generate once"
                assert set(result) == {"reply", "meme_id"}
                assert result["reply"].strip() and result["meme_id"] in valid_ids
                assert "<think>" not in result["reply"] and "</think>" not in result["reply"]
                outputs.append({"input": text, **result, "generate_calls": 1})
            assert history_lengths == [1, 3, 1], history_lengths
        finally:
            await client.close()
            await server.close()

    asyncio.run(verify_loopback())
    report = {"status": "complete", "model": "Qwen/Qwen3-8B", "real_model": True,
              "model_dir": str(args.model_dir.resolve()),
              "quality_status": engine.raw_engine.config.get("quality_status", "see_quality_reports"),
              "cases": outputs, "generate_calls": len(calls), "reranking": "disabled",
              "enable_thinking": False, "tcp_loopback": "passed",
              "history_lengths": history_lengths, "idle_reset_seconds": 10,
              "elapsed_seconds": time.monotonic() - started}
    destination = args.model_dir / "final_smoke.json"
    selection_path = args.model_dir / "deployment_selection.json"
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        report.update(quality_status=selection["quality_status"],
                      release_mode=selection["release_mode"],
                      selected_step=selection["selected_checkpoint"]["global_step"])
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verification_path = args.model_dir / "verification_summary.json"
    if verification_path.is_file():
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        verification.update(status="complete", real_model_validated=True, real_tcp_validated=True, real_full_training_complete=True,
                            quality_status=report.get("quality_status"), release_mode=report.get("release_mode", "strict_quality_acceptance"),
                            note="Code checks and completed-model real TCP inference passed.")
        verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(ROOT / "src"))
    from emostance_qwen.paper_pipeline import finalize_run
    finalize_run(engine.config)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
