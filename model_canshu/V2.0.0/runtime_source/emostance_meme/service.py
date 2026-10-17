"""Dependency-free business interface for CLI and HTTP adapters.

``engine.reply(messages)`` receives alternating user/assistant messages ending
with the new user input, each encoded as ``{"role": ..., "content": ...}``.
The engine must return ``{"reply": str, "meme_id": str}`` and may also return
a finite numeric ``distance``. ``ChatService.reply`` returns that public result;
an HTTP adapter can add ``session_id`` to its response envelope.

Pass ``cleanup_interval`` (for example 0.25 seconds) to expire idle histories in
the background. Expiry is measured from the latest received *user* message.
Model generation does not extend the 10 second deadline. Already queued work
finishes in order with its original conversation, and cannot restore a reset
session's old history.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Protocol

from .session import Message, Reply, SessionManager


class ReplyEngine(Protocol):
    def reply(self, messages: list[Message]) -> Mapping[str, Any]:
        """Generate a reply and select its corresponding meme."""
        ...


class ChatService:
    def __init__(
        self,
        engine: ReplyEngine,
        *,
        idle_timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        cleanup_interval: float | None = None,
    ) -> None:
        self.engine = engine
        self.sessions = SessionManager(
            idle_timeout=idle_timeout,
            clock=clock,
            cleanup_interval=cleanup_interval,
        )

    def reply(self, text: str, session_id: str = "default") -> Reply:
        """Handle one received user sentence and return reply plus meme_id."""
        return self.sessions.respond(session_id, text, self.engine.reply)

    def reset(self, session_id: str = "default") -> bool:
        return self.sessions.reset(session_id)

    def close(self) -> None:
        self.sessions.close()

    def __enter__(self) -> ChatService:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
