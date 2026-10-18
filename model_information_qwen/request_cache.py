"""Bounded in-memory v2 request receipts, scoped to one server instance.

All calls run on the server event loop. A receipt is registered before any
network await; final data is saved before delivery, so reconnecting clients
can recover a result without committing a second dialogue turn.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import time
from typing import Callable


@dataclass(eq=False)
class RequestReceipt:
    session_id: str
    request_id: str
    text: str
    completed: asyncio.Future
    response: dict | None = None
    completed_at: float | None = None
    playback_until: float | None = None


class RequestCache:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic,
                 ttl: float = 600., max_completed: int = 1024,
                 playback_lease: float = 120.):
        for name, value in (("ttl", ttl), ("playback_lease", playback_lease)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(max_completed, bool) or not isinstance(max_completed, int) or max_completed < 1:
            raise ValueError("max_completed must be a positive integer")
        self.clock = clock
        self.ttl = float(ttl)
        self.max_completed = max_completed
        self.playback_lease = float(playback_lease)
        self.receipts: dict[tuple[str, str], RequestReceipt] = {}
        self.active: dict[str, RequestReceipt] = {}

    def get(self, session_id: str, request_id: str) -> RequestReceipt | None:
        return self.receipts.get((session_id, request_id))

    def register(self, session_id: str, request_id: str, text: str) -> RequestReceipt:
        key = (session_id, request_id)
        if key in self.receipts or session_id in self.active:
            raise ValueError("request or active session already registered")
        receipt = RequestReceipt(session_id, request_id, text, asyncio.get_running_loop().create_future())
        self.receipts[key] = receipt
        self.active[session_id] = receipt
        return receipt

    def finish(self, receipt: RequestReceipt, response: dict) -> None:
        receipt.response = dict(response)
        receipt.completed_at = self.clock()
        if response["type"] == "result":
            receipt.playback_until = receipt.completed_at + self.playback_lease
        else:
            self.release(receipt)
        if not receipt.completed.done():
            receipt.completed.set_result(receipt.response)

    def release(self, receipt: RequestReceipt) -> bool:
        if self.active.get(receipt.session_id) is not receipt:
            return False
        self.active.pop(receipt.session_id)
        receipt.playback_until = None
        return True

    def expire(self) -> list[RequestReceipt]:
        """Release expired playback leases and prune only finished receipts."""
        now = self.clock()
        released = []
        for receipt in list(self.active.values()):
            if receipt.playback_until is not None and now >= receipt.playback_until:
                self.release(receipt)
                released.append(receipt)
        completed = []
        protected_completed = 0
        for key, receipt in list(self.receipts.items()):
            if receipt.completed_at is None:
                continue
            if self.active.get(receipt.session_id) is receipt:
                protected_completed += 1
                continue
            if now - receipt.completed_at >= self.ttl:
                self.receipts.pop(key)
            else:
                completed.append((receipt.completed_at, key))
        completed.sort(key=lambda item: item[0])
        for _, key in completed[:max(0, len(completed) + protected_completed - self.max_completed)]:
            self.receipts.pop(key)
        return released

    def close(self) -> None:
        for receipt in self.receipts.values():
            if not receipt.completed.done():
                receipt.completed.cancel()
        self.receipts.clear()
        self.active.clear()
