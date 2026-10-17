"""Prepare local dialogue annotations without inventing missing utterances.

All operations use the Python standard library.  Dialogue ids are the split unit;
annotators never become separate train/evaluation examples for the same turn.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class DataError(ValueError):
    """An invalid input, optionally accompanied by a preparation report."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report or {}


_ROLE = {"A": "user", "B": "assistant", "a": "user", "b": "assistant",
         "user": "user", "assistant": "assistant"}
_SPLIT = {"train": "train", "dev": "dev", "val": "dev", "valid": "dev",
          "validation": "dev", "test": "test"}
_CONTEXT_MARKER = re.compile(r"(?m)^\s*(A|B|user|assistant)\s*[:：]\s*")
_CHINESE = "\u3400-\u9fff\uf900-\ufaff"


def normalize_text(value: Any, *, detokenize_chinese: bool = True) -> str | None:
    """Strip empty text and optionally join the character-spaced LCCC Chinese."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataError("Utterance text/content must be a string or null.")
    value = " ".join(value.split())
    if detokenize_chinese:
        value = re.sub(rf"(?<=[{_CHINESE}])\s+|\s+(?=[{_CHINESE}])", "", value)
        value = re.sub(r"(?<=\d)\s+(?=\d)", "", value)
        value = re.sub(r"\s+([，。！？、；：～])", r"\1", value)
        value = re.sub(r"([，。！？、；：～])\s+", r"\1", value)
    return value or None


def _paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    found: dict[str, Path] = {}
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            raise DataError(f"Input path does not exist: {path}")
        candidates = path.rglob("*") if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}:
                if candidate.name == "report.json" or any(p.startswith(".") for p in candidate.parts):
                    continue
                found[str(candidate.resolve())] = candidate.resolve()
    return [found[key] for key in sorted(found)]


def _objects(value: Any, location: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if not all(isinstance(row, dict) for row in value):
            raise DataError(f"{location}: each dialogue/turn must be a JSON object with dialogue_id.")
        return value
    if isinstance(value, dict):
        if "dialogue_id" in value:
            return [value]
        for key in ("dialogues", "records", "data"):
            if key in value:
                return _objects(value[key], location)
    raise DataError(f"{location}: expected a dialogue object, an array of objects, or a dialogues/records/data container.")


def _read(path: Path) -> Iterable[tuple[dict[str, Any], str]]:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            if path.suffix.lower() == ".jsonl":
                for line_no, line in enumerate(stream, 1):
                    if line.strip():
                        location = f"{path}:{line_no}"
                        try:
                            rows = _objects(json.loads(line), location)
                        except json.JSONDecodeError as exc:
                            raise DataError(f"{location}: invalid JSON: {exc.msg}") from exc
                        for row in rows:
                            yield row, location
            else:
                for index, row in enumerate(_objects(json.load(stream), str(path))):
                    yield row, f"{path}#record={index + 1}"
    except json.JSONDecodeError as exc:
        raise DataError(f"{path}:{exc.lineno}: invalid JSON: {exc.msg}") from exc


def _turn_id(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataError(f"{location}: turn_id must be a nonnegative integer.")
    return value


def _role(value: Any, turn_id: int, location: str) -> tuple[str, bool]:
    if value is None:
        return ("user" if turn_id % 2 == 0 else "assistant"), False
    if not isinstance(value, str) or value not in _ROLE:
        raise DataError(f"{location}: role must be A/B or user/assistant, got {value!r}.")
    return _ROLE[value], True


def _context_turns(context: Any, target_id: int, location: str) -> list[dict[str, Any]]:
    if context is None or context == "":
        return []
    if isinstance(context, list):
        if not all(isinstance(turn, dict) for turn in context):
            raise DataError(f"{location}: context list must contain role/text objects.")
        result = [dict(turn) for turn in context]
    elif isinstance(context, str):
        markers = list(_CONTEXT_MARKER.finditer(context))
        if not markers:
            if target_id != 1:
                raise DataError(f"{location}: unlabeled context text is only supported before turn_id=1; use A:/B: lines.")
            result = [{"role": "A", "text": context}]
        else:
            if context[:markers[0].start()].strip():
                raise DataError(f"{location}: content before the first A:/B: context marker is ambiguous.")
            result = []
            for i, marker in enumerate(markers):
                end = markers[i + 1].start() if i + 1 < len(markers) else len(context)
                result.append({"role": marker.group(1), "text": context[marker.end():end]})
    else:
        raise DataError(f"{location}: context must be a string or a list of messages.")
    start_id = target_id - len(result)
    if start_id < 0:
        raise DataError(f"{location}: context contains more turns than the target turn_id allows.")
    for offset, turn in enumerate(result):
        turn.setdefault("turn_id", start_id + offset)
    return result


def prepare_dataset(
    annotation_paths: str | Path | Iterable[str | Path],
    text_paths: str | Path | Iterable[str | Path] = (),
    *,
    max_confidence: float = 5,
    seed: int = 42,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    strict: bool = True,
    detokenize_chinese: bool = True,
) -> dict[str, Any]:
    """Join annotations/text, aggregate labels, and split complete dialogues.

    Paths may be files or recursively searched directories.  Annotation-only
    turns require matching original text from ``text_paths``.  Missing text is
    fatal by default; ``strict=False`` skips the entire incomplete dialogue and
    records its missing turn ids and source locations in ``report``.  Invalid
    roles, conflicting text, and invalid confidence always raise ``DataError``.

    Unannotated assistant turns remain generation examples with an empty label
    distribution and emotion weight 0.  Their text is never given a fake meme.
    Explicit train/dev/test sources keep their split.  Unspecified sources use
    a reproducible hash of seed and dialogue_id, so all annotators stay together.
    """
    if isinstance(max_confidence, bool) or not isinstance(max_confidence, (int, float)) or not math.isfinite(max_confidence) or max_confidence <= 0:
        raise DataError("max_confidence must be a finite positive number.")
    if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0 for r in (dev_ratio, test_ratio)) or dev_ratio + test_ratio >= 1:
        raise DataError("dev_ratio and test_ratio must be nonnegative and sum to less than 1.")
    inputs = _paths(annotation_paths)
    text_inputs = _paths(text_paths)
    if not inputs and not text_inputs:
        raise DataError("No JSON/JSONL input files found. Dataset directories containing only .gitkeep have no training data.")
    dialogues: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {"annotation_files": [str(p) for p in inputs], "text_files": [str(p) for p in text_inputs],
                              "input_rows": 0, "duplicate_annotations": 0, "missing_text": [],
                              "skipped_dialogues": [], "unannotated_turns": [], "max_confidence": max_confidence,
                              "seed": seed, "dev_ratio": dev_ratio, "test_ratio": test_ratio}

    def add_turn(dialogue: dict[str, Any], raw: dict[str, Any], parent: dict[str, Any], location: str, default_id: int | None = None) -> None:
        turn_id = _turn_id(raw.get("turn_id", default_id), location)
        role, explicit = _role(raw.get("role"), turn_id, location)
        text = normalize_text(raw.get("text"), detokenize_chinese=detokenize_chinese)
        content = normalize_text(raw.get("content"), detokenize_chinese=detokenize_chinese)
        if text and content and text != content:
            raise DataError(f"{location}: text and content disagree for {dialogue['id']} turn {turn_id}.")
        text = text or content
        existing = dialogue["turns"].setdefault(turn_id, {"role": role, "explicit_role": explicit, "text": None, "labels": {}, "locations": []})
        existing["locations"].append(location)
        if explicit and existing["explicit_role"] and role != existing["role"]:
            raise DataError(f"{location}: conflicting roles for {dialogue['id']} turn {turn_id}.")
        if explicit:
            existing.update(role=role, explicit_role=True)
        if text:
            if existing["text"] is not None and existing["text"] != text:
                raise DataError(f"{location}: conflicting original text for {dialogue['id']} turn {turn_id}.")
            existing["text"] = text
        meme = raw.get("meme_id")
        confidence = raw.get("confidence")
        if meme is None and confidence is None:
            return
        if not isinstance(meme, str) or not meme.strip():
            raise DataError(f"{location}: an annotated turn requires a nonempty meme_id.")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or confidence <= 0 or confidence > max_confidence:
            raise DataError(f"{location}: confidence for {dialogue['id']} turn {turn_id} must be > 0 and <= {max_confidence}; got {confidence!r}.")
        annotator = raw.get("annotator", parent.get("annotator", "__unspecified__"))
        if not isinstance(annotator, str) or not annotator:
            raise DataError(f"{location}: annotator must be a nonempty string when supplied.")
        label = {"meme_id": meme.strip(), "confidence": float(confidence)}
        previous = existing["labels"].get(annotator)
        if previous:
            if previous != label:
                raise DataError(f"{location}: conflicting duplicate annotation by {annotator!r} for {dialogue['id']} turn {turn_id}; choose one revision before preparing data.")
            report["duplicate_annotations"] += 1
        else:
            existing["labels"][annotator] = label

    # Read both sources identically: the original text file may already carry
    # labels, and a prepared dialogue may need no separate text file at all.
    for path in dict.fromkeys(inputs + text_inputs):
        for row, location in _read(path):
            report["input_rows"] += 1
            dialogue_id = row.get("dialogue_id")
            if not isinstance(dialogue_id, str) or not dialogue_id.strip():
                raise DataError(f"{location}: dialogue_id must be a nonempty string; positional/raw LCCC arrays must first be given matching dialogue ids.")
            dialogue_id = dialogue_id.strip()
            dialogue = dialogues.setdefault(dialogue_id, {"id": dialogue_id, "turns": {}, "source_splits": set(), "locations": []})
            dialogue["locations"].append(location)
            source = row.get("source_split", row.get("split"))
            if source is not None:
                if not isinstance(source, str):
                    raise DataError(f"{location}: source_split/split must be a string when supplied.")
                normalized = _SPLIT.get(source.lower())
                if normalized:
                    dialogue["source_splits"].add(normalized)
            if len(dialogue["source_splits"]) > 1:
                raise DataError(f"{location}: dialogue {dialogue_id} has conflicting explicit source splits {sorted(dialogue['source_splits'])}; refusing possible evaluation leakage.")
            if "messages" in row or "turns" in row:
                groups = [row[key] for key in ("messages", "turns") if key in row]
                for turns in groups:
                    if not isinstance(turns, list) or not all(isinstance(t, dict) for t in turns):
                        raise DataError(f"{location}: messages/turns must be an array of turn objects.")
                    for index, turn in enumerate(turns):
                        add_turn(dialogue, turn, row, f"{location}/turn={index}", index)
            else:
                target_id = _turn_id(row.get("turn_id"), location)
                for turn in _context_turns(row.get("context"), target_id, location):
                    add_turn(dialogue, turn, {}, f"{location}/context")
                add_turn(dialogue, row, row, location)

    complete: dict[str, dict[str, Any]] = {}
    for dialogue_id in sorted(dialogues):
        dialogue = dialogues[dialogue_id]
        turns = dialogue["turns"]
        if not turns:
            report["skipped_dialogues"].append({"dialogue_id": dialogue_id, "reason": "empty_dialogue", "locations": dialogue["locations"]})
            continue
        missing = []
        for turn_id in range(max(turns) + 1):
            turn = turns.get(turn_id)
            if turn is None or not turn["text"]:
                missing.append({"dialogue_id": dialogue_id, "turn_id": turn_id,
                                "locations": turn["locations"] if turn else dialogue["locations"]})
            if turn and turn["role"] != ("user" if turn_id % 2 == 0 else "assistant"):
                raise DataError(f"Dialogue {dialogue_id} turn {turn_id}: malformed alternation; require A/user, B/assistant, A/user, ... starting at turn_id=0.", report)
        report["missing_text"].extend(missing)
        if missing:
            report["skipped_dialogues"].append({"dialogue_id": dialogue_id, "reason": "missing_original_text", "turn_ids": [r["turn_id"] for r in missing]})
        else:
            complete[dialogue_id] = dialogue
    if strict and report["missing_text"]:
        details = "; ".join(f"{r['dialogue_id']} turn {r['turn_id']} ({r['locations'][0]})" for r in report["missing_text"][:8])
        raise DataError(f"Missing original text for {len(report['missing_text'])} turn(s): {details}. Supply matching text_paths using dialogue_id + turn_id. Annotation reasons are not dialogue text.", report)

    splits: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "dev", "test")}
    records: dict[str, list[dict[str, Any]]] = {key: [] for key in splits}
    split_dialogues: dict[str, list[str]] = {key: [] for key in splits}
    for dialogue_id, dialogue in complete.items():
        if dialogue["source_splits"]:
            split = next(iter(dialogue["source_splits"]))
        else:
            score = int.from_bytes(hashlib.sha256(f"{seed}:{dialogue_id}".encode()).digest(), "big") / 2 ** 256
            split = "test" if score < test_ratio else "dev" if score < test_ratio + dev_ratio else "train"
        split_dialogues[split].append(dialogue_id)
        history = []
        for turn_id, turn in sorted(dialogue["turns"].items()):
            labels = list(turn["labels"].values())
            weights: dict[str, float] = defaultdict(float)
            for label in labels:
                # Normalizing each confidence first also supports very large
                # finite confidence scales without overflowing their sum.
                weights[label["meme_id"]] += label["confidence"] / max_confidence
            total = sum(weights.values())
            distribution = {meme: weight / total for meme, weight in sorted(weights.items())} if total else {}
            meme = min(distribution, key=lambda m: (-distribution[m], m)) if distribution else None
            weight = sum(label["confidence"] / max_confidence for label in labels) / len(labels) if labels else 0.0
            common = {"dialogue_id": dialogue_id, "turn_id": turn_id, "reply": turn["text"],
                      "meme_id": meme, "meme_distribution": distribution, "weight": weight}
            if labels:
                records[split].append({**common, "role": turn["role"]})
            else:
                report["unannotated_turns"].append({"dialogue_id": dialogue_id, "turn_id": turn_id, "split": split})
            if turn["role"] == "assistant":
                splits[split].append({**common, "messages": [dict(message) for message in history]})
            history.append({"role": turn["role"], "content": turn["text"]})
    report["split_dialogues"] = split_dialogues
    report["counts"] = {key: {"dialogues": len(split_dialogues[key]), "samples": len(splits[key]), "labeled_turns": len(records[key])} for key in splits}
    return {"splits": splits, "turn_records": records, "report": report}


def save_prepared(dataset: dict[str, Any], output_dir: str | Path) -> None:
    """Write generation JSONL, turn supervision JSONL, and a preparation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        for directory, rows in ((output_dir, dataset["splits"][name]), (output_dir / "turn_records", dataset["turn_records"][name])):
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "report.json").write_text(json.dumps(dataset["report"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

