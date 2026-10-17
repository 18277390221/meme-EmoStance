"""Thread-safe conversation history with a user-arrival-based idle deadline.

A request is assigned a conversation and a FIFO ticket when ``respond`` is
called, before waiting for another request's generation. Ten seconds means ten
seconds since the *last user's arrival*, not the last assistant completion.

An idle or explicit reset detaches the old conversation. Already accepted
requests keep its history and finish normally, while their results cannot
repopulate the newly reset conversation. Failed requests never commit a user
message without its matching assistant message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Mapping


Message = dict[str, str]
Reply = dict[str, Any]
Generator = Callable[[list[Message]], Mapping[str, Any]]


@dataclass
class _Conversation:
    history: list[Message] = field(default_factory=list)
    last_user_at: float | None = None


@dataclass
class _Session:
    current: _Conversation = field(default_factory=_Conversation)
    next_ticket: int = 0
    serving_ticket: int = 0
    pending: int = 0


class SessionManager:
    """Manage independent sessions and serialize requests within each session.

    The injected ``clock`` must have monotonic semantics and return seconds.
    ``cleanup_interval`` optionally starts a daemon that expires idle sessions
    without waiting for another request. Lazy checks always run on arrivals,
    completions, and history reads. ``close`` stops only that cleanup thread;
    it does not cancel accepted requests or own the generation engine.
    """

    def __init__(
        self,
        idle_timeout: float = 10.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        cleanup_interval: float | None = None,
    ) -> None:
        if not math.isfinite(idle_timeout) or idle_timeout <= 0:
            raise ValueError("idle_timeout must be a finite positive number")
        if cleanup_interval is not None and (
            not math.isfinite(cleanup_interval) or cleanup_interval <= 0
        ):
            raise ValueError("cleanup_interval must be a finite positive number")
        self.idle_timeout = float(idle_timeout)
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._sessions: dict[str, _Session] = {}
        self._stop = threading.Event()
        self._cleaner: threading.Thread | None = None
        if cleanup_interval is not None:
            self._cleaner = threading.Thread(
                target=self._cleanup_loop,
                args=(float(cleanup_interval),),
                name="emostance-session-cleanup",
                daemon=True,
            )
            self._cleaner.start()

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")

    def _expire(self, session: _Session, now: float) -> bool:
        last_user_at = session.current.last_user_at
        if last_user_at is not None and now - last_user_at >= self.idle_timeout:
            session.current = _Conversation()
            return True
        return False

    @staticmethod
    def _validate_reply(value: Mapping[str, Any]) -> Reply:
        if not isinstance(value, Mapping):
            raise TypeError("engine.reply must return a mapping")
        for key in ("reply", "meme_id"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"engine.reply must return a non-empty {key}")
        result: Reply = {"reply": value["reply"], "meme_id": value["meme_id"]}
        if "distance" in value:
            distance = value["distance"]
            if isinstance(distance, bool) or not isinstance(distance, (int, float)):
                raise ValueError("engine.reply distance must be numeric")
            if not math.isfinite(distance):
                raise ValueError("engine.reply distance must be finite")
            result["distance"] = float(distance)
        return result

    def respond(self, session_id: str, text: str, generate: Generator) -> Reply:
        """Generate and atomically commit one complete user/assistant pair.

        Requests from one session run in arrival order. Different sessions may
        call ``generate`` concurrently; an engine that cannot do this must
        provide its own model lock. No shared lock is held during generation.
        """
        self._validate_session_id(session_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        with self._condition:
            # Capture arrival before waiting for the previous model call.
            received_at = self._clock()
            session = self._sessions.setdefault(session_id, _Session())
            self._expire(session, received_at)
            conversation = session.current
            conversation.last_user_at = received_at
            ticket = session.next_ticket
            session.next_ticket += 1
            session.pending += 1
            while ticket != session.serving_ticket:
                self._condition.wait()
            # Work on a copy so engines cannot accidentally change history.
            messages = [dict(message) for message in conversation.history]
            messages.append({"role": "user", "content": text})

        try:
            result = self._validate_reply(generate(messages))
            with self._condition:
                self._expire(session, self._clock())
                # The captured conversation may now be detached. Keeping it
                # lets already queued old requests preserve their context,
                # without exposing any of it in the current session history.
                conversation.history.extend(
                    (
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": result["reply"]},
                    )
                )
            return result
        finally:
            with self._condition:
                session.pending -= 1
                session.serving_ticket += 1
                if session.pending == 0 and session.current.last_user_at is None:
                    if self._sessions.get(session_id) is session:
                        del self._sessions[session_id]
                self._condition.notify_all()

    def history(self, session_id: str) -> list[Message]:
        """Return a copy of committed messages in the current conversation."""
        self._validate_session_id(session_id)
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            self._expire(session, self._clock())
            history = [dict(message) for message in session.current.history]
            if session.pending == 0 and session.current.last_user_at is None:
                del self._sessions[session_id]
            return history

    def reset(self, session_id: str) -> bool:
        """Reset one session; accepted requests retain their captured context."""
        self._validate_session_id(session_id)
        with self._condition:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            if session.pending:
                session.current = _Conversation()
            else:
                del self._sessions[session_id]
            return True

    def expire_idle(self) -> int:
        """Expire idle conversations and release unused session records.

        The return value is the number of conversations expired by this call.
        In-flight and queued jobs are never canceled by cleanup.
        """
        with self._condition:
            now = self._clock()
            expired = 0
            for session_id, session in list(self._sessions.items()):
                expired += int(self._expire(session, now))
                if session.pending == 0 and session.current.last_user_at is None:
                    del self._sessions[session_id]
            return expired

    @property
    def active_session_count(self) -> int:
        """Number of retained records, including unfinished detached jobs."""
        with self._condition:
            return len(self._sessions)

    def _cleanup_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.expire_idle()

    def close(self) -> None:
        self._stop.set()
        if self._cleaner is not None and self._cleaner is not threading.current_thread():
            self._cleaner.join()

    def __enter__(self) -> SessionManager:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
