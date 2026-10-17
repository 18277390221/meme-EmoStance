#!/usr/bin/env python3
"""Split whole dialogues before constructing EmoStance adjacent-turn examples.

This local meme adaptation keeps all four weak annotators on each real turn,
uses confidence-weighted soft labels, and binds whitespace-normalized duplicate
and whole-turn proper-prefix dialogues to a single split. It never deletes or
rewrites its canonical input. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "dev", "test")
ANNOTATORS = frozenset(f"paper{i}" for i in range(1, 5))
SCHEMA_VERSION = "emostance-meme-adjacent-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    """Match the previous prefix audit without changing stored utterances."""
    text = " ".join(text.split())
    text = re.sub(r"(?<=[\u3400-\u9fff\uf900-\ufaff])\s+|\s+(?=[\u3400-\u9fff\uf900-\ufaff])", "", text)
    return re.sub(r"(?<=\d)\s+(?=\d)", "", text)


def dialogue_sequence(dialogue: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((turn["role"], normalized_text(turn["text"])) for turn in dialogue["messages"])


def load_dialogues(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path}:{line_no}"
            if not isinstance(row, dict) or not isinstance(row.get("dialogue_id"), str) or not row["dialogue_id"]:
                raise ValueError(f"{where}: dialogue_id must be a nonempty string")
            if row["dialogue_id"] in seen:
                raise ValueError(f"{where}: duplicate dialogue_id {row['dialogue_id']!r}")
            seen.add(row["dialogue_id"])
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{where}: a dialogue needs at least two messages")
            for turn_id, turn in enumerate(messages):
                turn_where = f"{where}, turn {turn_id}"
                if not isinstance(turn, dict) or turn.get("turn_id") != turn_id or isinstance(turn.get("turn_id"), bool):
                    raise ValueError(f"{turn_where}: turn_id must be contiguous and zero-based")
                if turn.get("role") != ("A" if turn_id % 2 == 0 else "B"):
                    raise ValueError(f"{turn_where}: expected alternating A/B roles starting with A")
                if not isinstance(turn.get("text"), str) or not turn["text"].strip():
                    raise ValueError(f"{turn_where}: original utterance text is required")
                annotations = turn.get("annotations")
                if not isinstance(annotations, list) or len(annotations) != 4:
                    raise ValueError(f"{turn_where}: exactly four annotations are required")
                if not all(isinstance(annotation, dict) for annotation in annotations):
                    raise ValueError(f"{turn_where}: annotations must be objects")
                if {annotation.get("annotator") for annotation in annotations} != ANNOTATORS:
                    raise ValueError(f"{turn_where}: annotations must be paper1, paper2, paper3 and paper4")
                for annotation in annotations:
                    meme_id, confidence = annotation.get("meme_id"), annotation.get("confidence")
                    if not isinstance(meme_id, str) or not meme_id.strip():
                        raise ValueError(f"{turn_where}: meme_id must be a nonempty string")
                    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 1 <= confidence <= 5:
                        raise ValueError(f"{turn_where}: confidence must be finite and between 1 and 5")
            rows.append(row)
    if not rows:
        raise ValueError(f"No dialogues in {path}")
    return sorted(rows, key=lambda row: row["dialogue_id"])


def soft_labels(annotations: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    """p(m) = sum confidence for m / sum confidence; weight = mean c / 5."""
    votes: dict[str, float] = defaultdict(float)
    for annotation in annotations:
        votes[annotation["meme_id"]] += annotation["confidence"]
    total = math.fsum(votes.values())
    return {meme: votes[meme] / total for meme in sorted(votes)}, total / (5 * len(annotations))


def related_groups(rows: list[dict[str, Any]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    """Connected components of equal/strict-prefix normalized turn sequences."""
    parents = {row["dialogue_id"]: row["dialogue_id"] for row in rows}

    def find(dialogue_id: str) -> str:
        while parents[dialogue_id] != dialogue_id:
            parents[dialogue_id] = parents[parents[dialogue_id]]
            dialogue_id = parents[dialogue_id]
        return dialogue_id

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[max(left, right)] = min(left, right)

    indexed: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
    for row in rows:
        indexed[dialogue_sequence(row)].append(row["dialogue_id"])
    relations = []
    for sequence, ids in sorted(indexed.items()):
        for duplicate in ids[1:]:
            union(ids[0], duplicate)
            relations.append({"kind": "normalized_duplicate", "first_dialogue_id": ids[0], "second_dialogue_id": duplicate})
        for length in range(1, len(sequence)):
            for short_id in indexed.get(sequence[:length], []):
                for long_id in ids:
                    union(short_id, long_id)
                    relations.append({"kind": "normalized_proper_prefix", "first_dialogue_id": short_id, "second_dialogue_id": long_id})
    components: dict[str, list[str]] = defaultdict(list)
    for dialogue_id in parents:
        components[find(dialogue_id)].append(dialogue_id)
    return sorted((sorted(ids) for ids in components.values()), key=lambda ids: ids[0]), relations


def split_groups(groups: list[list[str]], *, seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[dict[str, str], dict[str, str], dict[str, int]]:
    if len(ratios) != 3 or any(isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0 for ratio in ratios) or not math.isclose(sum(ratios), 1):
        raise ValueError("Three finite, nonnegative split ratios must sum to 1")
    total = sum(map(len, groups))
    raw_targets = [total * ratio for ratio in ratios]
    targets = [math.floor(value) for value in raw_targets]
    # Largest remainder gives reproducible integer targets summing to N.
    remainder_order = sorted(range(3), key=lambda i: (-(raw_targets[i] - targets[i]), i))
    for i in remainder_order[:total - sum(targets)]:
        targets[i] += 1
    shuffled = sorted(groups, key=lambda ids: ids[0])
    random.Random(seed).shuffle(shuffled)
    # Larger components go first, leaving singleton components to fill targets.
    shuffled.sort(key=len, reverse=True)
    remaining = targets[:]
    assignments, group_ids = {}, {}
    for ids in shuffled:
        index = max(range(3), key=lambda i: (remaining[i], -i))
        remaining[index] -= len(ids)
        group_id = "group-" + hashlib.sha256("\0".join(ids).encode()).hexdigest()[:16]
        for dialogue_id in ids:
            assignments[dialogue_id] = SPLITS[index]
            group_ids[dialogue_id] = group_id
    return assignments, group_ids, dict(zip(SPLITS, targets))


def make_records(row: dict[str, Any], split: str, group_id: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    original_split = row.get("original_source_split", row.get("source_split"))
    common = {"dialogue_id": row["dialogue_id"], "original_dialogue_id": row.get("original_dialogue_id", row["dialogue_id"]),
              "original_source_split": original_split, "source_split": split, "split": split, "group_id": group_id}
    dialogue = {**row, **common}
    turns = []
    for turn in row["messages"]:
        distribution, weight = soft_labels(turn["annotations"])
        turns.append({**common, "turn_id": turn["turn_id"], "role": turn["role"], "text": turn["text"],
                      "meme_distribution": distribution, "weight": weight, "annotations": turn["annotations"]})
    samples = []
    for source, target in zip(turns, turns[1:]):
        samples.append({**common, "sample_id": f"{row['dialogue_id']}:{source['turn_id']}->{target['turn_id']}",
                        "source_turn_id": source["turn_id"], "target_turn_id": target["turn_id"],
                        "history": [{"role": turn["role"], "text": turn["text"]} for turn in turns[:target["turn_id"]]],
                        "source_text": source["text"], "response": target["text"],
                        "source_role": source["role"], "next_role": target["role"],
                        "source_meme_distribution": source["meme_distribution"], "target_meme_distribution": target["meme_distribution"],
                        "source_weight": source["weight"], "target_weight": target["weight"],
                        "source_annotations": source["annotations"], "target_annotations": target["annotations"]})
    return dialogue, samples, turns


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_splits(input_path: Path, output_root: Path, *, seed: int = 42) -> dict[str, Any]:
    input_path, output_root = input_path.resolve(), output_root.resolve()
    if input_path == output_root or output_root in input_path.parents:
        raise ValueError("Output directory must not contain the canonical input")
    source_hash = sha256_file(input_path)
    provenance_path = input_path.with_suffix(".manifest.json")
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("output_sha256") != source_hash:
            raise ValueError("Canonical source hash disagrees with its export manifest")
    rows = load_dialogues(input_path)
    groups, relations = related_groups(rows)
    assignments, group_ids, targets = split_groups(groups, seed=seed)
    dialogues: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    samples: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    turns: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in rows:
        split = assignments[row["dialogue_id"]]
        dialogue, dialogue_samples, dialogue_turns = make_records(row, split, group_ids[row["dialogue_id"]])
        dialogues[split].append(dialogue)
        samples[split].extend(dialogue_samples)
        turns[split].extend(dialogue_turns)
    # Independently inspect the produced records before publishing any output.
    id_sets = {split: {row["dialogue_id"] for row in dialogues[split]} for split in SPLITS}
    cross_dialogue = sorted((id_sets["train"] & id_sets["dev"]) | (id_sets["train"] & id_sets["test"]) | (id_sets["dev"] & id_sets["test"]))
    crossing_relations = [relation for relation in relations if assignments[relation["first_dialogue_id"]] != assignments[relation["second_dialogue_id"]]]
    if cross_dialogue or crossing_relations or sum(map(len, id_sets.values())) != len(rows):
        raise AssertionError("Dialogue grouping leakage or dialogue loss")
    for split in SPLITS:
        if len(samples[split]) != sum(len(row["messages"]) - 1 for row in dialogues[split]):
            raise AssertionError("Incorrect adjacent-pair count")
        if len(turns[split]) != sum(len(row["messages"]) for row in dialogues[split]):
            raise AssertionError("Incorrect real-turn count")
        if {sample["dialogue_id"] for sample in samples[split]} != id_sets[split]:
            raise AssertionError("Sample lost or assigned to a different split")
    meme_sets = {split: {meme for turn in turns[split] for meme in turn["meme_distribution"]} for split in SPLITS}
    coverage = {}
    for split in SPLITS:
        meme_annotation_counts = Counter(annotation["meme_id"] for turn in turns[split] for annotation in turn["annotations"])
        coverage[split] = {"meme_ids": sorted(meme_sets[split]), "meme_id_count": len(meme_sets[split]),
                           "annotation_counts": dict(sorted(meme_annotation_counts.items())),
                           "unseen_in_train": sorted(meme_sets[split] - meme_sets["train"])}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": str(input_path), "sha256": source_hash, "dialogues": len(rows), "turns": sum(len(row["messages"]) for row in rows)},
        "seed": seed, "requested_ratios": {"train": .8, "dev": .1, "test": .1}, "target_dialogue_counts": targets,
        "rules": {
            "split_unit": "whole dialogue and its four annotators; related normalized duplicate/proper-prefix dialogues remain in one connected group",
            "normalization": "collapse whitespace; remove spaces adjacent to Chinese characters and between consecutive digits; preserve punctuation, case and turn roles",
            "slicing": "all consecutive source/target turns, A->B and B->A, with original full history through source; N turns yield N-1 examples",
            "meme_distribution": "sum confidence per meme_id divided by total confidence across paper1..paper4",
            "turn_weight": "sum confidence across paper1..paper4 / (4 * 5)",
            "original_source_split": "provenance only; source_split and split are the newly assigned split",
        },
        "splits": {}, "meme_coverage": coverage,
        "grouping": {"group_count": len(groups), "nonsingleton_group_count": sum(len(group) > 1 for group in groups),
                     "related_dialogue_count": sum(len(group) for group in groups if len(group) > 1),
                     "relations": relations,
                     "bound_groups": [{"group_id": group_ids[group[0]], "dialogue_ids": group, "split": assignments[group[0]]} for group in groups if len(group) > 1]},
        "leakage_audit": {"cross_split_dialogue_ids": cross_dialogue, "cross_split_normalized_duplicate_or_prefix_relations": crossing_relations,
                          "all_dialogues_preserved": sum(map(len, id_sets.values())) == len(rows),
                          "all_four_annotators_preserved": all(len(turn["annotations"]) == 4 for split in SPLITS for turn in turns[split]),
                          "passed": True},
        "output_files": {},
    }
    if provenance is not None:
        manifest["source"]["export_manifest"] = {"path": str(provenance_path), "sha256": sha256_file(provenance_path)}
        manifest["source"]["upstream_directory"] = provenance.get("source_directory")
        manifest["source"]["upstream_input_sha256"] = provenance.get("input_sha256", {})
    for split in SPLITS:
        manifest["splits"][split] = {"dialogues": len(dialogues[split]), "dialogue_ratio": len(dialogues[split]) / len(rows),
                                    "real_turns": len(turns[split]), "annotations": 4 * len(turns[split]), "adjacent_samples": len(samples[split]),
                                    "sample_source_roles": dict(sorted(Counter(row["source_role"] for row in samples[split]).items())),
                                    "sample_target_roles": dict(sorted(Counter(row["next_role"] for row in samples[split]).items())),
                                    "original_source_splits": dict(sorted(Counter(str(row["original_source_split"]) for row in dialogues[split]).items()))}
        for relative, records in [(f"dialogues/{split}.jsonl", dialogues[split]), (f"{split}.jsonl", samples[split]), (f"annotation_rows/{split}.jsonl", turns[split])]:
            path = output_root / relative
            write_jsonl(path, records)
            manifest["output_files"][relative] = {"sha256": sha256_file(path), "records": len(records), "bytes": path.stat().st_size}
    if sha256_file(input_path) != source_hash:
        raise RuntimeError("Canonical input changed during preparation")
    manifest_path = output_root / "metadata/split_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "dataset/v1.0.0/cleaned_dialogues.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data_spilt/v1.0.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = build_splits(args.input, args.output, seed=args.seed)
    print(json.dumps({"source": manifest["source"], "splits": manifest["splits"], "grouping": manifest["grouping"],
                      "leakage_audit": manifest["leakage_audit"], "manifest": str(args.output / "metadata/split_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
