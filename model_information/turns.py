"""Internal completed-turn value; user preparation metadata stays off the wire."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TurnResult:
    """The effective user text and final reply for one accepted request.

    The session manager validates all fields before committing history.
    ``user_text`` can differ from the received STT text; it is internal and is
    never added to the public reply payload by ``as_reply``.
    """

    user_text: str
    reply: str
    meme_id: str

    def as_reply(self) -> dict[str, str]:
        return {"reply": self.reply, "meme_id": self.meme_id}
