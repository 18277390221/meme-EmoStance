#!/usr/bin/env python3
"""Join cleaned dialogue text with four annotations without changing inputs.

The source is deliberately an explicit set of five current JSONL files, plus
the meme catalogue. Legacy turn-level annotations and backups are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emostance_meme.data import normalize_text  # noqa: E402


ANNOTATORS = tuple(f"paper{i}" for i in range(1, 5))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "dataset/v.1.0.0/data"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset/v1.0.0/cleaned_dialogues.jsonl"
SOURCE_FIELDS = ("source", "source_model", "source_prompt_set")
ANNOTATION_FIELDS = ("model", "prompt_set", "prompt_variant", "prompt_sha256")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_dialogues(rows: list[dict[str, Any]], location: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        identifier = row.get("dialogue_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{location}: missing dialogue_id")
        if identifier in result:
            raise ValueError(f"{location}: duplicate dialogue_id {identifier}")
        result[identifier] = row
    return result


def merge_dialogues(source: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    """Validate and merge all turns; any missing label aborts the entire export."""
    source = Path(source).resolve()
    text_path = source / "dialogues_pool.jsonl"
    catalogue_path = source / "meme_catalog.jsonl"
    input_paths = [text_path, catalogue_path]
    raw = read_jsonl(text_path)
    dialogue_index = index_dialogues(raw, text_path)
    if not raw:
        raise ValueError("Source contains no dialogues")
    catalogue = read_jsonl(catalogue_path)
    catalogue_ids = [row.get("meme_id") for row in catalogue]
    if any(not isinstance(value, str) or not re.fullmatch(r"meme-\d{4}", value) for value in catalogue_ids):
        raise ValueError(f"{catalogue_path}: invalid meme_id")
    valid_memes = set(catalogue_ids)
    if not valid_memes or len(valid_memes) != len(catalogue_ids):
        raise ValueError(f"{catalogue_path}: catalogue must be nonempty and have unique meme ids")

    annotation_index = {}
    for annotator in ANNOTATORS:
        path = source / f"annotations_dialogue_pool_{annotator}.jsonl"
        rows = index_dialogues(read_jsonl(path), path)
        missing = set(dialogue_index) - set(rows)
        extra = set(rows) - set(dialogue_index)
        if missing or extra:
            raise ValueError(f"{annotator}: incomplete dialogue coverage; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
        annotation_index[annotator] = rows
        input_paths.append(path)

    result = []
    for row in raw:
        identifier = row["dialogue_id"]
        original_id = row.get("original_dialogue_id", identifier)
        if not isinstance(original_id, str) or not original_id:
            raise ValueError(f"{identifier}: invalid original_dialogue_id")
        split = row.get("source_split")
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"{identifier}: invalid original source_split {split!r}")
        texts = row.get("turns")
        if not isinstance(texts, list) or len(texts) < 2:
            raise ValueError(f"{identifier}: expected at least two dialogue turns")
        per_annotator = {}
        for annotator in ANNOTATORS:
            annotation = annotation_index[annotator][identifier]
            if annotation.get("annotator") != annotator:
                raise ValueError(f"{identifier}: annotator name mismatches {annotator} file")
            if annotation.get("source_split") != split:
                raise ValueError(f"{identifier}/{annotator}: original source_split mismatch")
            if annotation.get("original_dialogue_id", original_id) != original_id:
                raise ValueError(f"{identifier}/{annotator}: original_dialogue_id mismatch")
            turns = annotation.get("turns")
            if not isinstance(turns, list):
                raise ValueError(f"{identifier}/{annotator}: missing annotation turns")
            turn_map = {}
            for turn in turns:
                if not isinstance(turn, dict):
                    raise ValueError(f"{identifier}/{annotator}: annotation turn must be an object")
                turn_id = turn.get("turn_id")
                if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id in turn_map:
                    raise ValueError(f"{identifier}/{annotator}: invalid or duplicate turn_id {turn_id!r}")
                turn_map[turn_id] = turn
            if set(turn_map) != set(range(len(texts))):
                raise ValueError(f"{identifier}/{annotator}: missing or extra turn annotations")
            per_annotator[annotator] = turn_map

        messages = []
        for turn_id, raw_text in enumerate(texts):
            if not isinstance(raw_text, str):
                raise ValueError(f"{identifier}/{turn_id}: dialogue text must be a string")
            text = normalize_text(raw_text)
            if not text or "\ufffd" in text:
                raise ValueError(f"{identifier}/{turn_id}: empty or corrupted dialogue text")
            labels = []
            for annotator in ANNOTATORS:
                turn = per_annotator[annotator][turn_id]
                meme_id, confidence, reason = (turn.get(key) for key in ("meme_id", "confidence", "reason"))
                if meme_id not in valid_memes:
                    raise ValueError(f"{identifier}/{turn_id}/{annotator}: unknown meme_id {meme_id!r}")
                if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
                    raise ValueError(f"{identifier}/{turn_id}/{annotator}: confidence must be an integer from 1 to 5")
                if not isinstance(reason, str) or not reason.strip() or "\ufffd" in reason:
                    raise ValueError(f"{identifier}/{turn_id}/{annotator}: missing or corrupted annotation reason")
                label = {"annotator": annotator, "meme_id": meme_id, "confidence": confidence, "reason": reason}
                label.update({key: turn[key] for key in SOURCE_FIELDS if key in turn})
                parent = annotation_index[annotator][identifier]
                label["annotation_provenance"] = {key: parent[key] for key in ANNOTATION_FIELDS if key in parent}
                labels.append(label)
            messages.append({"turn_id": turn_id, "role": "A" if turn_id % 2 == 0 else "B", "text": text, "annotations": labels})
        result.append({"dialogue_id": identifier, "original_dialogue_id": original_id,
                       "source_split": split, "messages": messages})
    return result, input_paths


def _temporary_bytes(destination: Path, content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                                     dir=destination.parent, delete=False) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
        return Path(stream.name)


def export_dataset(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Export atomically; an existing differing output or manifest is refused."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the original source directory")
    rows, inputs = merge_dialogues(source)
    source_hashes = {path.name: sha256_file(path) for path in inputs}
    payload = b"".join((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    manifest_path = output.with_suffix(".manifest.json")
    split_counts = {}
    for split in ("train", "dev", "test"):
        records = [row for row in rows if row["source_split"] == split]
        split_counts[split] = {"dialogues": len(records), "turns": sum(len(row["messages"]) for row in records)}
    turns = sum(len(row["messages"]) for row in rows)
    manifest = {
        "schema_version": 1,
        "source_directory": str(source),
        "output_file": str(output),
        "dialogues": len(rows), "turns": turns, "annotations": turns * len(ANNOTATORS),
        "annotators": list(ANNOTATORS), "confidence_range": [1, 5],
        "source_split_counts": split_counts,
        "distinct_meme_ids": len({label["meme_id"] for row in rows for message in row["messages"] for label in message["annotations"]}),
        "text_normalization": "existing normalize_text: join Chinese character spacing and digit spacing, preserve Latin word spaces",
        "role_assignment": "alternating A/B by zero-based turn_id; original source contains ordered text turns",
        "input_sha256": source_hashes,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "output_bytes": len(payload),
    }
    manifest_payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    destinations = [(output, payload), (manifest_path, manifest_payload)]
    for destination, content in destinations:
        if destination.exists() and sha256_file(destination) != hashlib.sha256(content).hexdigest():
            raise FileExistsError(f"Refusing to overwrite differing existing file: {destination}; choose a new --output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = []
    try:
        for destination, content in destinations:
            if not destination.exists():
                temporary.append((_temporary_bytes(destination, content), destination))
        if any(sha256_file(path) != source_hashes[path.name] for path in inputs):
            raise RuntimeError("Source changed during export; no output was installed")
        for temp_path, destination in temporary:
            os.replace(temp_path, destination)
    finally:
        for temp_path, _ in temporary:
            temp_path.unlink(missing_ok=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = export_dataset(args.source, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
