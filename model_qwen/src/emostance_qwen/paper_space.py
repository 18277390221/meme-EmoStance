"""Train-only name-free graph space, adapted from EmoStance (MIT).

See model/THIRD_PARTY_LICENSES/EmoStance.txt. Chinese characters replace the
English-only token regex; feature hashing, shrinkage and graph equations follow
the reference implementation. No meme description or annotation reason is used.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .space import EmotionSpace


def unit_rows(a):
    a = np.asarray(a, dtype=np.float64)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, norms, out=np.zeros_like(a), where=norms > 0)


class ChineseHashedTfidf:
    """Stable token/adjacent-bigram TF-IDF with a persisted training IDF."""
    token_re = re.compile(r"[\u3400-\u9fff]|[a-z0-9']+", re.I)

    def __init__(self, dim=256):
        self.dim = int(dim)
        self.idf = None

    def features(self, text):
        tokens = self.token_re.findall(text.lower())
        # Keep an otherwise punctuation-only turn represented without using labels.
        if not tokens:
            tokens = [c for c in text if not c.isspace()] or ["<empty>"]
        return [f"tok:{t}" for t in tokens] + [f"bigram:{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def hash_feature(self, feature):
        value = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return value % self.dim, 1.0 if ((value >> 63) & 1) == 0 else -1.0

    def fit_transform(self, texts):
        document_frequency = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for i in {self.hash_feature(f)[0] for f in self.features(text)}:
                document_frequency[i] += 1
        self.idf = np.log((len(texts) + 1) / (document_frequency + 1)) + 1
        return self.transform(texts)

    def transform(self, texts):
        if self.idf is None:
            raise ValueError("Fit IDF on training utterances first")
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for feature in self.features(text):
                i, sign = self.hash_feature(feature)
                matrix[row, i] += sign
        return unit_rows(matrix * self.idf)


def normalize_similarity(matrix):
    n = len(matrix)
    off = matrix[~np.eye(n, dtype=bool)]
    if not off.size or np.ptp(off) < 1e-12:
        result = np.zeros_like(matrix)
    else:
        result = np.clip((matrix - off.min()) / np.ptp(off), 0, 1)
    np.fill_diagonal(result, 1)
    return result


def build_space(turns, train_samples, run_dir, config):
    """Fit graph and target prototypes exclusively on the training partition."""
    import igraph as ig
    import leidenalg

    if not turns or any(r.get("split", r.get("source_split")) != "train" for r in turns):
        raise ValueError("Graph fitting requires exclusively training utterances")
    if not train_samples or any(r.get("split", r.get("source_split")) != "train" for r in train_samples):
        raise ValueError("Prototypes require exclusively training samples")
    run_dir = Path(run_dir)
    out = run_dir / "graph"
    out.mkdir(parents=True, exist_ok=True)
    meme_ids = sorted({m for r in turns for m in r["meme_distribution"]})
    lookup = {m: i for i, m in enumerate(meme_ids)}
    q = np.zeros((len(turns), len(meme_ids)), dtype=np.float64)
    observed = Counter()
    for i, row in enumerate(turns):
        for meme, probability in row["meme_distribution"].items():
            q[i, lookup[meme]] = probability
        # Observed frequency in the reference is raw annotator vote frequency.
        observed.update(a["meme_id"] for a in row["annotations"])
    assert np.allclose(q.sum(1), 1)
    encoder = ChineseHashedTfidf(config.get("vector_dim", 256))
    features = encoder.fit_transform([r["text"] for r in turns])
    weights = q * np.asarray([r["weight"] for r in turns])[:, None]
    effective = weights.sum(0)
    raw = (weights.T @ features) / effective[:, None]
    counts = np.asarray([observed[m] for m in meme_ids], dtype=float)
    alpha = counts / (counts + config.get("centroid_shrinkage_tau", 50.0))
    centroids = alpha[:, None] * raw + (1-alpha[:, None]) * features.mean(0)
    ctx = unit_rows(centroids) @ unit_rows(centroids).T
    confusion = q.T @ q
    denom = np.sqrt(np.outer(confusion.diagonal(), confusion.diagonal()))
    confusion = np.divide(confusion, denom, out=np.zeros_like(confusion), where=denom > 0)
    affinity = .65 * normalize_similarity(ctx) + .35 * normalize_similarity(confusion)
    np.fill_diagonal(affinity, 1)
    k = min(config.get("knn_k", 8), len(meme_ids)-1)
    edges = {}
    for i in range(len(meme_ids)):
        neighbors = sorted((j for j in range(len(meme_ids)) if j != i), key=lambda j: (-affinity[i,j], meme_ids[j]))[:k]
        for j in neighbors:
            edges[tuple(sorted((i,j)))] = float(affinity[i,j])
    graph = ig.Graph(n=len(meme_ids), edges=list(edges), directed=False)
    graph.es["weight"] = list(edges.values())
    partition = leidenalg.find_partition(graph, leidenalg.RBConfigurationVertexPartition,
        weights="weight", resolution_parameter=config.get("leiden_resolution", 1.6), seed=config.get("seed",42))
    groups = sorted([sorted(group) for group in partition], key=lambda g: g[0])
    labels = np.empty(len(meme_ids), dtype=int)
    for ci, group in enumerate(groups):
        labels[group] = ci
    membership = np.zeros((len(meme_ids),len(groups)), dtype=np.float64)
    for i in range(len(meme_ids)):
        support = np.asarray([max((affinity[i,j] for j in group if j != i), default=0.) for group in groups])
        own = labels[i]
        threshold = max(config.get("boundary_absolute_threshold", .70), support[own] * config.get("boundary_relative_threshold",.85))
        cross = sorted((c for c in range(len(groups)) if c != own and support[c] >= threshold), key=lambda c: (-support[c],c))
        selected = [own] + cross[:config.get("boundary_max_clusters",4)-1]
        if len(selected) == 1:
            membership[i,own] = 1
        else:
            # Reference sharpen_memberships: normalized support^(1/T).
            values = np.maximum(support[selected],0) ** (1/config.get("soft_membership_temperature",.7))
            membership[i,selected] = values/values.sum()
    vectors, target_q = [], []
    for row in train_samples:
        vote = np.zeros(len(meme_ids), dtype=np.float64)
        for meme, probability in row["target_meme_distribution"].items():
            vote[lookup[meme]] = probability
        target_q.append(vote @ membership)
        vectors.append(vote @ centroids)
    target_q, vectors = np.asarray(target_q), np.asarray(vectors)
    # Reference cluster_prototypes uses target soft mass, without additional confidence.
    mass = target_q.sum(0)
    prototypes = (target_q.T @ vectors) / np.maximum(mass[:,None],1e-12)
    prototypes[mass <= 1e-12] = vectors.mean(0)
    space = EmotionSpace(meme_ids,centroids,membership,prototypes)
    space.save(run_dir/"space.json")
    np.savez_compressed(out/"graph_arrays.npz",meme_ids=np.asarray(meme_ids),idf=encoder.idf,
        centroids=centroids,membership=membership,affinity=affinity,labels=labels,
        counts=counts,effective_counts=effective,shrinkage_alpha=alpha)
    report = dict(training_utterances=len(turns),training_samples=len(train_samples),
        meme_count=len(meme_ids),vector_dim=space.vector_dim,num_clusters=space.num_clusters,
        cluster_sizes=[len(g) for g in groups],soft_membership_memes=int((np.count_nonzero(membership,axis=1)>1).sum()),
        zero_feature_utterances=int((np.linalg.norm(features,axis=1)==0).sum()),
        algorithm="leidenalg.RBConfigurationVertexPartition",resolution=config.get("leiden_resolution",1.6),
        fit_splits=["train"],feature_template="Chinese character/Latin word unigrams + adjacent bigrams; signed hashed TF-IDF",
        meme_descriptions_used=False,annotation_reasons_used=False)
    (out/"manifest.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return space,report
