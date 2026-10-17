#!/usr/bin/env python3
"""Verify the completed real model, ten-second sessions, and local HTTP API.

Run from any directory with the project's existing Python environment:
    python model/tools/verify_paper_runtime.py [--config model/paper_config.json]

This loads the completed PaperMemeEngine once. All prompts are fixed artificial
inputs; no development/test examples or training/optimizer steps are used.
"""

from __future__ import annotations

import argparse
import gc
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback
from typing import Any
from urllib.request import ProxyHandler, Request, build_opener

MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT / "src"))


class FakeClock:
    """Keep synthetic arrival times independent of actual generation duration."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class GenerationCounter:
    def __init__(self, original: Any) -> None:
        self.original = original
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        return self.original(*args, **kwargs)


class RecordingEngine:
    """Record actual service-to-engine histories without replacing inference."""

    def __init__(self, engine: Any, counter: GenerationCounter) -> None:
        self.engine = engine
        self.counter = counter
        self.calls: list[dict[str, Any]] = []

    def reply(self, messages: list[dict[str, str]]) -> dict[str, str]:
        record = {"messages": [dict(message) for message in messages]}
        self.calls.append(record)
        before = self.counter.calls
        result = self.engine.reply(messages)
        record["generated_candidates"] = self.counter.calls - before
        record["output"] = dict(result)
        return result


def require(condition: Any, message: str) -> None:
    # Explicit checks also remain active if the interpreter is run with -O.
    if not condition:
        raise AssertionError(message)


def check_reply(result: Any, legal_meme_ids: set[str], location: str) -> None:
    require(isinstance(result, dict), f"{location}: response is not an object")
    require(set(result) == {"reply", "meme_id"}, f"{location}: expected exactly reply and meme_id fields, got {sorted(result)}")
    require(isinstance(result["reply"], str) and bool(result["reply"].strip()), f"{location}: generated reply is empty")
    require(isinstance(result["meme_id"], str) and result["meme_id"] in legal_meme_ids, f"{location}: illegal meme_id {result['meme_id']!r}")


def verify(config_path: str | None) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from emostance_meme.paper_pipeline import load_config, write_json

    try:
        config = load_config(config_path)
    except Exception as exc:
        print(f"Runtime verification failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1

    import numpy as np
    import torch
    from emostance_meme.http_server import make_handler
    from emostance_meme.paper_engine import PaperMemeEngine
    from emostance_meme.service import ChatService

    started = time.monotonic()
    output_path = Path(config["run_dir"]) / "final_smoke.json"
    report: dict[str, Any] = {"format_version": 1, "status": "running", "preview": False,
                              "purpose": "Completed-model engineering verification, not a conversational-quality assessment",
                              "seed": 42, "training_or_optimizer_steps": 0, "dev_or_test_samples_read": 0,
                              "expected_candidates_per_request": 4, "results": []}
    engine = recorder = counter = service = http_service = server = thread = None
    exit_code = 1
    try:
        require(bool(config.get("do_sample", True)), "Four-candidate smoke requires the saved sampling configuration")
        require(int(config.get("num_candidates", 4)) == 4, "Four-candidate smoke requires num_candidates=4")
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.set_num_threads(8)
        engine = PaperMemeEngine(config)
        require(all(not parameter.requires_grad for parameter in engine.generator.model.parameters()), "Mistral base model has trainable parameters")
        report["base_model_frozen"] = True
        report["run_dir"] = config["run_dir"]
        checkpoint = engine.generator.checkpoint_state
        require(checkpoint is not None, "Completed generator loaded without a checkpoint")
        report["generator_checkpoint_global_step"] = int(checkpoint["global_step"])
        # Release checkpoint optimizer tensors held on CPU; inference uses only
        # the already loaded projector weights, and no optimizer is created.
        engine.generator.checkpoint_state = None
        del checkpoint
        legal_meme_ids = set(engine.planner.space.meme_ids)
        report["legal_meme_count"] = len(legal_meme_ids)
        counter = GenerationCounter(engine.generator.generate)
        engine.generator.generate = counter
        recorder = RecordingEngine(engine, counter)
        clock = FakeClock()
        service = ChatService(recorder, idle_timeout=10.0, clock=clock)
        prompts = ("今天终于完成任务了，特别开心！", "最近有点难过，想找人聊聊。", "你好，你在做什么？")
        replies = []
        for index, (arrival, prompt) in enumerate(zip((0.0, 4.0, 14.0), prompts)):
            clock.now = arrival
            call_started = time.monotonic()
            result = service.reply(prompt, session_id="final-smoke")
            check_reply(result, legal_meme_ids, f"service request {index + 1}")
            record = recorder.calls[index]
            require(record["generated_candidates"] == 4, f"service request {index + 1}: expected 4 real candidates, got {record['generated_candidates']}")
            replies.append(result)
            report["results"].append({"arrival_seconds": arrival, "input": prompt, **result,
                                       "history_length": len(record["messages"]), "actual_messages": record["messages"],
                                       "generated_candidates": record["generated_candidates"],
                                       "elapsed_seconds": time.monotonic() - call_started})
            print(json.dumps({"event": "service_smoke", **report["results"][-1]}, ensure_ascii=False), flush=True)
        lengths = [len(call["messages"]) for call in recorder.calls]
        report["actual_history_lengths"] = lengths
        require(lengths == [1, 3, 1], f"Expected histories [1, 3, 1] at t=0/4/14, got {lengths}")
        require(recorder.calls[1]["messages"] == [
            {"role": "user", "content": prompts[0]},
            {"role": "assistant", "content": replies[0]["reply"]},
            {"role": "user", "content": prompts[1]}], "Second request does not contain the exact previous user/assistant exchange")
        require(recorder.calls[2]["messages"] == [{"role": "user", "content": prompts[2]}], "Old context remains after exactly ten idle seconds")
        require(service.sessions.history("final-smoke") == [
            {"role": "user", "content": prompts[2]},
            {"role": "assistant", "content": replies[2]["reply"]}], "Reset conversation did not commit only the current user/assistant pair")
        report["ten_second_reset_verified"] = True
        service.close()
        service = None

        http_clock = FakeClock()
        http_service = ChatService(recorder, idle_timeout=10.0, clock=http_clock)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(http_service))
        thread = threading.Thread(target=server.serve_forever, name="final-smoke-http", daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(ProxyHandler({}))

        def request(path: str, body: dict[str, str] | None = None) -> tuple[int, Any]:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
            headers = {"Content-Type": "application/json"} if data is not None else {}
            with opener.open(Request(base_url + path, data=data, headers=headers), timeout=120) as response:
                return response.status, json.loads(response.read().decode("utf-8"))

        status, health = request("/health")
        require(status == 200 and health == {"status": "ok"}, f"Unexpected /health response: {status}, {health}")
        before_http_calls = len(recorder.calls)
        status, chat = request("/chat", {"session_id": "http-final-smoke", "text": "你好，很高兴见到你。"})
        require(status == 200, f"Unexpected /chat HTTP status {status}")
        check_reply(chat, legal_meme_ids, "HTTP /chat")
        require(len(recorder.calls) == before_http_calls + 1, "HTTP /chat did not call the same real model engine exactly once")
        require(recorder.calls[-1]["generated_candidates"] == 4, "HTTP /chat did not generate four real candidates")
        require(len(http_service.sessions.history("http-final-smoke")) == 2, "HTTP /chat failed to commit a user/assistant pair")
        status, reset = request("/reset", {"session_id": "http-final-smoke"})
        require(status == 200 and reset == {"status": "reset"}, f"Unexpected /reset response: {status}, {reset}")
        require(http_service.sessions.history("http-final-smoke") == [], "HTTP /reset left conversation history behind")
        report["http"] = {"bind_host": "127.0.0.1", "ephemeral_port": server.server_port, "health": health,
                           "chat": chat, "chat_exact_fields_verified": True,
                           "generated_candidates": recorder.calls[-1]["generated_candidates"],
                           "reset": reset, "history_after_reset": []}
        report["total_real_generation_candidates"] = counter.calls
        require(counter.calls == 16, f"Expected 16 generated candidates across four real requests, got {counter.calls}")
        report["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        print(f"Runtime verification failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=10)
            if thread.is_alive():
                report["status"] = "failed"
                report["cleanup_error"] = "HTTP server thread did not stop"
                exit_code = 1
        for active_service in (service, http_service):
            if active_service is not None:
                active_service.close()
        if engine is not None and counter is not None:
            engine.generator.generate = counter.original
        # Break proxy references before dropping the single model instance.
        if recorder is not None:
            recorder.engine = None
            recorder.counter = None
        if counter is not None:
            counter.original = None
        engine = recorder = counter = service = http_service = server = thread = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(output_path, report)
        print(json.dumps({"status": report["status"], "report": str(output_path),
                          "actual_history_lengths": report.get("actual_history_lengths"),
                          "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Optional paper_config.json path")
    raise SystemExit(verify(parser.parse_args().config))
