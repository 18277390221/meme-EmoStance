"""Arrival-timed dialogue queues for the server-to-Raspberry-Pi adapter.

All public methods belong to one asyncio event loop. ``accept`` registers a
user's arrival synchronously before any executor or model-lock waiting. Jobs
capture their conversation object, so timeout/reset can detach old history
without canceling already accepted questions or letting late replies revive it.

The engine exposes synchronous ``reply(messages)`` or optional
``reply_turn(messages) -> TurnResult``. The latter may prepare the current user
text and returns the effective text to commit with its final reply. Calls run
serially outside the event loop. This module has no model, network or credential
dependency and never reads configuration files.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable

from .protocol import validate_reply, validate_text
from .turns import TurnResult


class BusyError(RuntimeError):
    """The per-session or global accepted-request limit has been reached."""


@dataclass(eq=False)
class _Conversation:
    history: list[dict[str, str]] = field(default_factory=list)
    last_user_at: float | None = None


@dataclass(eq=False)
class _Session:
    current: _Conversation = field(default_factory=_Conversation)
    queue: deque[_Job] = field(default_factory=deque)
    pending: int = 0
    worker: asyncio.Task[None] | None = None


@dataclass(eq=False)
class _Job:
    session_id: str
    session: _Session
    conversation: _Conversation
    text: str
    future: asyncio.Future[dict[str, str]]
    released: bool = False


class DialogueManager:
    """Independent FIFO dialogues with a last-user-arrival idle deadline.

    Call synchronous ``start()`` in the running server loop, then synchronous
    ``accept(session_id, text)`` as soon as a valid network request is read.
    Await the returned future with ``asyncio.shield`` if disconnecting the
    caller must not cancel its result future. Client identity/isolation comes
    from the supplied session ID, not a human-readable device name.

    Limits count both queued and executing accepted requests. Rejected
    requests do not modify history, activity timestamps or queue position.
    Canceling a result future does not retract an already accepted question;
    ``close()`` cancels all unfinished jobs. A borrowed executor remains owned
    by its caller; the manager shuts down only an executor it creates itself.

    A conversation contains at most ``max_turns`` successfully committed
    user/reply pairs. Its next inference starts a new conversation, including
    requests that were already queued before the preceding reply completed.
    """

    def __init__(
        self,
        engine: Any,
        idle_timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        cleanup_interval: float | None = .25,
        max_pending_per_session: int = 8,
        max_pending_total: int = 128,
        executor: Executor | None = None,
        max_turns: int = 5,
    ) -> None:
        if not callable(getattr(engine, "reply_turn", None)) and not callable(getattr(engine, "reply", None)):
            raise TypeError("engine must provide synchronous reply_turn(messages) or reply(messages)")
        if isinstance(idle_timeout, bool) or not isinstance(idle_timeout, (int, float)) or not math.isfinite(idle_timeout) or idle_timeout <= 0:
            raise ValueError("idle_timeout must be finite and positive")
        if cleanup_interval is not None and (
            isinstance(cleanup_interval, bool) or not isinstance(cleanup_interval, (int, float))
            or not math.isfinite(cleanup_interval) or cleanup_interval <= 0
        ):
            raise ValueError("cleanup_interval must be finite and positive or None")
        for name, value in (("max_pending_per_session", max_pending_per_session),
                            ("max_pending_total", max_pending_total), ("max_turns", max_turns)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.engine = engine
        self.idle_timeout = float(idle_timeout)
        self.clock = clock
        self.cleanup_interval = cleanup_interval
        self.max_pending_per_session = max_pending_per_session
        self.max_pending_total = max_pending_total
        self.max_turns = max_turns
        self._executor = executor
        self._owns_executor = executor is None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._model_lock: asyncio.Lock | None = None
        self._cleaner: asyncio.Task[None] | None = None
        self._sessions: dict[str, _Session] = {}
        self._jobs: dict[asyncio.Future[dict[str, str]], _Job] = {}
        self._workers: set[asyncio.Task[None]] = set()
        self._closed = False

    def start(self) -> DialogueManager:
        """Start maintenance synchronously in the running server event loop."""
        if self._closed:
            raise RuntimeError("DialogueManager is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None:
            if loop is not self._loop:
                raise RuntimeError("DialogueManager belongs to another event loop")
            return self
        self._loop = loop
        self._model_lock = asyncio.Lock()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-information")
        if self.cleanup_interval is not None:
            self._cleaner = loop.create_task(self._cleanup(), name="model-information-idle-cleanup")
        return self

    def _check_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Call DialogueManager.start() in the running event loop first")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("DialogueManager methods must run in their owning event loop")
        return self._loop

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a nonempty string")

    def _now(self) -> float:
        value = self.clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("clock must return finite monotonic seconds")
        return float(value)

    def _expire(self, session: _Session, now: float) -> bool:
        last = session.current.last_user_at
        if last is not None and now - last >= self.idle_timeout:
            session.current = _Conversation()
            return True
        return False

    def _forget_empty(self, session_id: str, session: _Session) -> None:
        if session.pending == 0 and session.current.last_user_at is None and self._sessions.get(session_id) is session:
            del self._sessions[session_id]

    @staticmethod
    def _observe_future(future: asyncio.Future[Any]) -> None:
        # Retrieving an exception suppresses "never retrieved" when a peer has
        # disconnected. Awaiting the future later still raises that exception.
        if not future.cancelled():
            future.exception()

    def accept(self, session_id: str, text: str) -> asyncio.Future[dict[str, str]]:
        """Capture arrival and enqueue immediately, without waiting for the GPU."""
        loop = self._check_loop()
        if self._closed:
            raise RuntimeError("DialogueManager is closed")
        self._validate_session_id(session_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a nonempty string")
        session = self._sessions.get(session_id)
        if (session is not None and session.pending >= self.max_pending_per_session) or self.pending_count >= self.max_pending_total:
            raise BusyError("The request queue is full; retry after a pending reply completes")
        received_at = self._now()
        if session is None:
            session = _Session()
            self._sessions[session_id] = session
        self._expire(session, received_at)
        conversation = session.current
        conversation.last_user_at = received_at
        future: asyncio.Future[dict[str, str]] = loop.create_future()
        future.add_done_callback(self._observe_future)
        job = _Job(session_id, session, conversation, text, future)
        session.queue.append(job)
        session.pending += 1
        self._jobs[future] = job
        if session.worker is None or session.worker.done():
            session.worker = loop.create_task(self._work(session_id, session), name="model-information-dialogue-worker")
            self._workers.add(session.worker)
            session.worker.add_done_callback(self._workers.discard)
        return future

    @staticmethod
    def _validate_reply(value: Any) -> dict[str, str]:
        # Validate the exact wire contract before committing either turn.
        return validate_reply(value)

    def _infer_turn(self, messages: list[dict[str, str]], original_text: str) -> TurnResult:
        """Prepare and generate once, wholly inside the existing model worker.

        Legacy engines keep the original user text independently of mutations
        to their input snapshot. A turn-aware engine owns STT normalization and
        its fallback policy; only a complete, validated TurnResult can commit.
        """
        reply_turn = getattr(self.engine, "reply_turn", None)
        if callable(reply_turn):
            turn = reply_turn(messages)
            if not isinstance(turn, TurnResult):
                raise TypeError("engine.reply_turn must return TurnResult")
            user_text = validate_text(turn.user_text)
            result = self._validate_reply(turn.as_reply())
        else:
            user_text = original_text
            result = self._validate_reply(self.engine.reply(messages))
        return TurnResult(user_text=user_text, reply=result["reply"], meme_id=result["meme_id"])

    def _release(self, job: _Job) -> None:
        if job.released:
            return
        job.released = True
        self._jobs.pop(job.future, None)
        job.session.pending -= 1
        self._forget_empty(job.session_id, job.session)

    def _rollover_if_full(self, job: _Job) -> None:
        # Only validated, successful pairs enter history. Admission/pending
        # counts therefore cannot prematurely consume the conversation limit.
        previous = job.conversation
        if len(previous.history) < 2 * self.max_turns:
            return
        following = _Conversation(last_user_at=previous.last_user_at)
        job.conversation = following
        # Queued requests captured the previous object at arrival. Move only
        # that same conversation's jobs, preserving later idle/manual resets.
        for queued in job.session.queue:
            if queued.conversation is previous:
                queued.conversation = following
        if job.session.current is previous:
            job.session.current = following

    async def _work(self, session_id: str, session: _Session) -> None:
        try:
            while session.queue and not self._closed:
                job = session.queue.popleft()
                try:
                    self._rollover_if_full(job)
                    # Copy only after preceding jobs commit, retaining FIFO
                    # history even if this captured conversation was detached.
                    messages = [dict(message) for message in job.conversation.history]
                    messages.append({"role": "user", "content": job.text})
                    assert self._model_lock is not None and self._loop is not None and self._executor is not None
                    async with self._model_lock:
                        turn = await self._loop.run_in_executor(self._executor, self._infer_turn, messages, job.text)
                    result = turn.as_reply()
                    self._expire(session, self._now())
                    job.conversation.history.extend((
                        {"role": "user", "content": turn.user_text},
                        {"role": "assistant", "content": result["reply"]},
                    ))
                    if not job.future.done():
                        job.future.set_result(result)
                except asyncio.CancelledError:
                    if not job.future.done():
                        job.future.cancel()
                    raise
                except Exception as exc:
                    if not job.future.done():
                        # A long-lived worker must not share its live traceback
                        # frame with consumers that clear exception tracebacks.
                        job.future.set_exception(exc.with_traceback(None))
                finally:
                    self._release(job)
        finally:
            session.worker = None
            self._forget_empty(session_id, session)

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Return the current committed dialogue; never expose detached jobs."""
        self._check_loop()
        self._validate_session_id(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            return []
        self._expire(session, self._now())
        history = [dict(message) for message in session.current.history]
        self._forget_empty(session_id, session)
        return history

    def reset(self, session_id: str) -> bool:
        """Detach one current dialogue while accepted jobs keep captured context."""
        self._check_loop()
        self._validate_session_id(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.current = _Conversation()
        self._forget_empty(session_id, session)
        return True

    @property
    def pending_count(self) -> int:
        return len(self._jobs)

    async def _cleanup(self) -> None:
        assert self.cleanup_interval is not None
        while True:
            await asyncio.sleep(self.cleanup_interval)
            now = self._now()
            for session_id, session in list(self._sessions.items()):
                self._expire(session, now)
                self._forget_empty(session_id, session)

    async def close(self) -> None:
        """Cancel pending jobs without blocking on an already running model call."""
        if self._loop is not None:
            self._check_loop()
        if self._closed:
            return
        self._closed = True
        for future in list(self._jobs):
            if not future.done():
                future.cancel()
        tasks = list(self._workers)
        if self._cleaner is not None:
            tasks.append(self._cleaner)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job in list(self._jobs.values()):
            self._release(job)
        self._sessions.clear()
        if self._owns_executor and self._executor is not None:
            # Running calls cannot be interrupted; their worker coroutine was
            # canceled, so their eventual result cannot commit any history.
            self._executor.shutdown(wait=False, cancel_futures=True)
