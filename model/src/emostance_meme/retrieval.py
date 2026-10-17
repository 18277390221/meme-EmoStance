"""Local nearest-response search in a learned emotion space (no ML dependency)."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path
from typing import Any, Iterable


def _vector(values: Iterable[float], dimension: int | None = None) -> list[float]:
    try:
        result = list(values)
    except TypeError as exc:
        raise ValueError("Emotion vector must be an iterable of finite numbers.") from exc
    if not result or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in result):
        raise ValueError("Emotion vector must contain finite real numbers and cannot be empty.")
    if dimension is not None and len(result) != dimension:
        raise ValueError(f"Emotion vector dimension mismatch: expected {dimension}, got {len(result)}.")
    if not any(v != 0 for v in result):
        raise ValueError("Emotion vector cannot be all zeros.")
    return [float(v) for v in result]


class ResponseIndex:
    """Map the nearest labeled training reply to its meme id.

    Pass only training records.  The returned ``reply`` is the retrieved anchor;
    the conversational engine should keep its separately generated reply text.
    Ties are resolved by meme id, reply, dialogue id, then turn id, independent
    of file ordering.  Cosine distances range from 0 to 2.
    """

    def __init__(self, records: Iterable[dict[str, Any]], metric: str = "cosine"):
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'.")
        self.metric = metric
        self.records: list[dict[str, Any]] = []
        self.dimension: int | None = None
        for raw in records:
            if not isinstance(raw, dict):
                raise ValueError("Each response index record must be an object.")
            if not isinstance(raw.get("reply"), str) or not raw["reply"].strip():
                raise ValueError("Each response index record needs nonempty reply text.")
            if not isinstance(raw.get("meme_id"), str) or not raw["meme_id"].strip():
                raise ValueError("Each response index record needs a nonempty meme_id.")
            # Refuse explicitly marked evaluation records even if a caller
            # accidentally mixes them into the training retrieval bank.
            split = raw.get("source_split", raw.get("split"))
            if split is not None and split != "train":
                raise ValueError(f"Response index accepts training records only; got split {split!r}.")
            if "vector" not in raw:
                raise ValueError("Each response index record needs an emotion vector.")
            vector = _vector(raw["vector"], self.dimension)
            self.dimension = len(vector)
            record = dict(raw)
            record["vector"] = vector
            self.records.append(record)
        if not self.records:
            raise ValueError("Cannot build a response index without labeled training replies.")

    def _distance(self, left: list[float], right: list[float]) -> float:
        if self.metric == "euclidean":
            return math.dist(left, right)
        # Scaling first avoids overflow/underflow for finite very large/small
        # input values while preserving cosine direction exactly.
        a_scale, b_scale = max(abs(v) for v in left), max(abs(v) for v in right)
        a, b = [v / a_scale for v in left], [v / b_scale for v in right]
        cosine = math.fsum(x * y for x, y in zip(a, b)) / (math.sqrt(math.fsum(x * x for x in a)) * math.sqrt(math.fsum(y * y for y in b)))
        return 1.0 - min(1.0, max(-1.0, cosine))

    def search(self, vector: Iterable[float], k: int = 1) -> list[dict[str, Any]]:
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        query = _vector(vector, self.dimension)
        candidates = [{**record, "vector": list(record["vector"]), "distance": self._distance(query, record["vector"])} for record in self.records]
        candidates.sort(key=lambda record: (record["distance"], record["meme_id"], record["reply"], str(record.get("dialogue_id", "")), str(record.get("turn_id", ""))))
        return candidates[:k]

    def nearest(self, vector: Iterable[float]) -> dict[str, Any]:
        return self.search(vector, k=1)[0]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": 1, "metric": self.metric, "dimension": self.dimension, "records": self.records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, metric: str | None = None) -> "ResponseIndex":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(value, list):
            return cls(value, metric=metric or "cosine")
        if not isinstance(value, dict) or "records" not in value:
            raise ValueError("Invalid response index: expected records array.")
        if value.get("version", 1) != 1:
            raise ValueError("Unsupported response index format version.")
        index = cls(value["records"], metric=metric or value.get("metric", "cosine"))
        if "dimension" in value and value["dimension"] != index.dimension:
            raise ValueError("Stored response index dimension does not match its records.")
        return index
