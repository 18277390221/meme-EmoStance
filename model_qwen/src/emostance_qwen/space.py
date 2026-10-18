"""Train-only, name-free weak meme supervision and prototype reconstruction.

The caller must pass training turns only. Meme identifiers are opaque keys:
neither their names nor annotation reasons enter the representation. Features
can come from the existing local language model; this module never downloads
an encoder. All fitted artifacts needed at inference are saved as JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize without overflowing on large, finite input values."""
    scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    scaled = np.divide(matrix, scale, out=np.zeros_like(matrix), where=scale > 0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    return np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0)


def _distribution(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("meme_distribution must be a nonempty mapping")
    result: dict[str, float] = {}
    for meme_id, value in values.items():
        if not isinstance(meme_id, str) or not meme_id.strip():
            raise ValueError("meme identifiers must be nonempty strings")
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("meme distribution weights must be finite numbers") from exc
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("meme distribution weights must be finite and nonnegative")
        result[meme_id] = weight
    # Scaling first also handles very large finite vote counts.
    maximum = max(result.values())
    if maximum <= 0:
        raise ValueError("meme distribution must contain positive mass")
    scaled = {key: value / maximum for key, value in result.items()}
    total = sum(scaled.values())
    return {key: value / total for key, value in scaled.items()}


def _cluster(affinity: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic spectral clustering with an in-repo k-means solver."""
    size = len(affinity)
    if count == 1:
        return np.zeros(size, dtype=np.int64)
    if count == size:
        return np.arange(size, dtype=np.int64)
    degrees = affinity.sum(axis=1)
    inverse = 1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    normalized = inverse[:, None] * affinity * inverse[None, :]
    _, eigenvectors = np.linalg.eigh(normalized)
    points = _unit_rows(eigenvectors[:, -count:])
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(size))]
    while len(chosen) < count:
        distances = np.min(
            ((points[:, None, :] - points[chosen][None, :, :]) ** 2).sum(axis=2),
            axis=1,
        )
        distances[chosen] = 0.0
        if distances.sum() <= 1e-12:
            chosen.append(next(index for index in range(size) if index not in chosen))
        else:
            chosen.append(int(rng.choice(size, p=distances / distances.sum())))
    centers = points[chosen].copy()
    previous: np.ndarray | None = None
    for _ in range(100):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        # Equal features may collapse assignments. Keep every region nonempty
        # by moving a farthest point out of a region with multiple members.
        counts = np.bincount(labels, minlength=count)
        for empty in np.flatnonzero(counts == 0):
            eligible = np.flatnonzero(counts[labels] > 1)
            donor = int(eligible[np.argmax(distances[eligible, labels[eligible]])])
            counts[labels[donor]] -= 1
            labels[donor] = empty
            counts[empty] += 1
        if previous is not None and np.array_equal(labels, previous):
            break
        centers = np.stack([points[labels == index].mean(axis=0) for index in range(count)])
        previous = labels.copy()
    # Stable cluster numbering makes persisted artifacts easier to compare.
    groups = sorted(range(count), key=lambda index: int(np.flatnonzero(labels == index)[0]))
    remap = np.empty(count, dtype=np.int64)
    remap[groups] = np.arange(count)
    return remap[labels]


@dataclass
class EmotionSpace:
    meme_ids: list[str]
    centroids: np.ndarray
    membership: np.ndarray
    prototypes: np.ndarray

    def __post_init__(self) -> None:
        self.meme_ids = list(self.meme_ids)
        self.centroids = np.asarray(self.centroids, dtype=np.float64)
        self.membership = np.asarray(self.membership, dtype=np.float64)
        self.prototypes = np.asarray(self.prototypes, dtype=np.float64)
        if not self.meme_ids or len(set(self.meme_ids)) != len(self.meme_ids):
            raise ValueError("EmotionSpace requires unique, nonempty meme identifiers")
        if any(not isinstance(value, str) or not value.strip() for value in self.meme_ids):
            raise ValueError("EmotionSpace meme identifiers must be nonempty strings")
        size = len(self.meme_ids)
        if self.centroids.ndim != 2 or self.centroids.shape[0] != size or self.centroids.shape[1] < 1:
            raise ValueError("centroids must have shape (number of memes, positive vector dimension)")
        if self.membership.ndim != 2 or self.membership.shape[0] != size or self.membership.shape[1] < 1:
            raise ValueError("membership must have shape (number of memes, positive region count)")
        if self.prototypes.shape != (self.membership.shape[1], self.centroids.shape[1]):
            raise ValueError("prototypes shape must match region count and vector dimension")
        for matrix in (self.centroids, self.membership, self.prototypes):
            if not np.isfinite(matrix).all():
                raise ValueError("EmotionSpace contains non-finite values")
        if (self.membership < 0).any() or not np.allclose(self.membership.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("every meme membership row must be a probability distribution")

    @property
    def vector_dim(self) -> int:
        return int(self.centroids.shape[1])

    @property
    def num_clusters(self) -> int:
        return int(self.membership.shape[1])

    @classmethod
    def fit(
        cls,
        turn_records: Sequence[Mapping[str, Any]],
        features_np: np.ndarray,
        *,
        vector_dim: int = 32,
        num_clusters: int = 8,
        seed: int = 42,
        shrinkage: float = 1.0,
        temperature: float = 0.2,
    ) -> EmotionSpace:
        records = list(turn_records)
        features = np.asarray(features_np, dtype=np.float64)
        if features.ndim != 2 or not records or features.shape[0] != len(records) or features.shape[1] < 1:
            raise ValueError("features must be a nonempty (number of training turns, feature dimension) matrix")
        if not np.isfinite(features).all():
            raise ValueError("training features must be finite")
        if not isinstance(vector_dim, int) or vector_dim < 1:
            raise ValueError("vector_dim must be a positive integer")
        if not isinstance(num_clusters, int) or num_clusters < 1:
            raise ValueError("num_clusters must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and nonnegative")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        distributions = [_distribution(record.get("meme_distribution", {})) for record in records]
        try:
            weights = np.asarray([float(record.get("weight", 1.0)) for record in records], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("training confidence weights must be finite and positive") from exc
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("training confidence weights must be finite and positive")
        meme_ids = sorted({key for distribution in distributions for key, value in distribution.items() if value > 0})
        lookup = {key: index for index, key in enumerate(meme_ids)}
        votes = np.zeros((len(records), len(meme_ids)), dtype=np.float64)
        for index, distribution in enumerate(distributions):
            for meme_id, value in distribution.items():
                if value > 0:
                    votes[index, lookup[meme_id]] = value

        features = _unit_rows(features)
        if features.shape[1] > vector_dim:
            # Uncentered weighted PCA/SVD keeps the global direction and remains
            # meaningful for a single sample (centered PCA would be all zero).
            scaled_weights = weights / weights.max()
            _, singular, axes = np.linalg.svd(features * np.sqrt(scaled_weights[:, None]), full_matrices=False)
            rank = max(1, int(np.count_nonzero(singular > max(float(singular[0]), 1.0) * 1e-10)))
            dimensions = min(vector_dim, rank, axes.shape[0])
            axes = axes[:dimensions].copy()
            for axis in axes:
                if axis[np.argmax(np.abs(axis))] < 0:
                    axis *= -1
            features = _unit_rows(features @ axes.T)

        # Use scaled weights for numerically stable means. Shrinkage retains
        # its original confidence-mass units through a matching rescaling.
        weight_scale = weights.max()
        stable_weights = weights / weight_scale
        weighted_votes = stable_weights[:, None] * votes
        effective = weighted_votes.sum(axis=0)
        means = (weighted_votes.T @ features) / effective[:, None]
        global_mean = (stable_weights @ features) / stable_weights.sum()
        scaled_shrinkage = shrinkage / weight_scale
        alpha = effective / (effective + scaled_shrinkage)
        centroids = _unit_rows(alpha[:, None] * means + (1.0 - alpha[:, None]) * global_mean)

        confusion = votes.T @ weighted_votes
        diagonal = np.sqrt(np.maximum(np.diag(confusion), 0.0))
        denominator = diagonal[:, None] * diagonal[None, :]
        confusion = np.divide(confusion, denominator, out=np.zeros_like(confusion), where=denominator > 0)
        context_similarity = np.clip((centroids @ centroids.T + 1.0) / 2.0, 0.0, 1.0)
        affinity = 0.65 * context_similarity + 0.35 * confusion
        np.fill_diagonal(affinity, 1.0)
        regions = min(num_clusters, len(meme_ids))
        labels = _cluster(affinity, regions, seed)
        support = np.stack([affinity[:, labels == index].mean(axis=1) for index in range(regions)], axis=1)
        # Subtract the maximum before dividing, including for tiny temperatures.
        with np.errstate(under="ignore", over="ignore"):
            logits = (support - support.max(axis=1, keepdims=True)) / temperature
            membership = np.exp(logits)
        membership /= membership.sum(axis=1, keepdims=True)
        targets = votes @ membership
        vectors = votes @ centroids
        prototype_weights = stable_weights[:, None] * targets
        mass = prototype_weights.sum(axis=0)
        prototypes = np.divide(
            prototype_weights.T @ vectors,
            mass[:, None],
            out=np.zeros((regions, centroids.shape[1]), dtype=np.float64),
            where=mass[:, None] > 0,
        )
        # An extremely small temperature may underflow every membership of a
        # weak region to zero. Preserve finite reconstruction with a train-only
        # global prototype rather than inventing held-out supervision.
        prototypes[mass <= 0] = (stable_weights @ vectors) / stable_weights.sum()
        return cls(meme_ids, centroids, membership, prototypes)

    def targets(self, distribution: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        normalized = _distribution(distribution)
        lookup = {key: index for index, key in enumerate(self.meme_ids)}
        unknown = sorted(set(normalized) - lookup.keys())
        if unknown:
            raise ValueError("Unknown meme identifiers in fitted training space: " + ", ".join(unknown))
        votes = np.zeros(len(self.meme_ids), dtype=np.float64)
        for meme_id, value in normalized.items():
            votes[lookup[meme_id]] = value
        return votes @ self.membership, votes @ self.centroids

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "meme_ids": self.meme_ids,
            "centroids": self.centroids.tolist(),
            "membership": self.membership.tolist(),
            "prototypes": self.prototypes.tolist(),
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmotionSpace:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported EmotionSpace JSON format")
        return cls(payload["meme_ids"], payload["centroids"], payload["membership"], payload["prototypes"])
