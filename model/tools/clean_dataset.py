"""Audit, back up, clean and consistently renumber the four-annotator dataset.

No model calls. Dry-run is the default; --apply replaces only the active data/
directory after validation and preserves the original directory in backups/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from zoneinfo import ZoneInfo

ANNOTATORS = tuple(f"paper{i}" for i in range(1, 5))


def load_rows(path):
    result = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                # Without a trustworthy identifier, deleting the correct
                # dialogue cannot be guaranteed. Stop instead of guessing.
                raise ValueError(f"{path}:{number}: 无法解析 JSON，需先定位原对话") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: 数据行不是对象")
            result.append(row)
    return result


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_objects(text, dialogue_format):
    """Recover top-level answer objects, tolerating fences and explanation.

    Move past a complete matching object so nested turn dictionaries are not
    mistaken for additional answers. Multiple answers remain visible for
    conflict checks; only whitespace/int representation is normalized later.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    decoder = json.JSONDecoder()
    result = []
    position = 0
    while position < len(text):
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        matches = isinstance(value, dict) and (
            isinstance(value.get("turns"), list) if dialogue_format
            else "meme_id" in value and "confidence" in value
        )
        if matches:
            result.append(value)
            position = start + length
        else:
            position = start + 1
    return result


def candidate_signature(value, dialogue_format, turn_id=None):
    turns = value.get("turns", []) if dialogue_format else [{**value, "turn_id": turn_id}]
    result = {}
    for turn in turns:
        raw_id = turn.get("turn_id")
        if isinstance(raw_id, bool):
            raise ValueError("Invalid raw response turn_id")
        index = int(raw_id)
        if index in result:
            raise ValueError("Repeated raw response turn_id")
        result[index] = (turn.get("meme_id"), turn.get("confidence"), str(turn.get("reason", "")).strip())
    return result


def audit(directory):
    directory = Path(directory)
    raw = load_rows(directory / "dialogues_pool.jsonl")
    dialogues = {row["dialogue_id"]: row for row in raw}
    if len(dialogues) != len(raw):
        raise ValueError("原始对话 ID 重复，无法安全重编号")
    catalog = load_rows(directory / "meme_catalog.jsonl")
    meme_ids = {row["meme_id"] for row in catalog}
    if len(meme_ids) != len(catalog):
        raise ValueError("表情目录 ID 重复")
    new = {}
    legacy = {}
    indexed = {}
    issues = defaultdict(list)
    reason_only = set()

    def issue(identifier, code, **detail):
        issues[identifier].append({"code": code, **detail})

    for identifier, row in dialogues.items():
        if row.get("source_split") not in {"train", "dev", "test"}:
            raise ValueError(f"无效 source_split：{identifier}")
        if not isinstance(row.get("turns"), list) or len(row["turns"]) < 2:
            issue(identifier, "invalid_dialogue_text")
            continue
        for index, text in enumerate(row["turns"]):
            if not isinstance(text, str) or not text.strip() or "\ufffd" in text or "\x00" in text:
                issue(identifier, "invalid_dialogue_text", turn_id=index)

    def inspect_label(identifier, label, annotator, filename, line):
        detail = {"annotator": annotator, "file": filename, "line": line, "turn_id": label.get("turn_id")}
        if label.get("meme_id") not in meme_ids:
            issue(identifier, "invalid_meme_id", **detail)
        confidence = label.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 1 <= confidence <= 5:
            issue(identifier, "invalid_confidence", **detail)
        reason = label.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            issue(identifier, "empty_annotation_reason", **detail)
        elif "\ufffd" in reason or "\x00" in reason:
            issue(identifier, "garbled_annotation_reason", **detail)
        elif reason.strip().lower() in {"无", "none", "null", "n/a"} or any(word in reason for word in ("占位", "解析失败", "标注失败", "接口错误")):
            issue(identifier, "failed_annotation_reason", **detail)

    def inspect_response(identifier, row, annotator, filename, line, dialogue_format):
        if "raw_response_text" not in row:
            return
        objects = response_objects(row["raw_response_text"], dialogue_format)
        if not objects:
            issue(identifier, "unrecoverable_raw_response", annotator=annotator, file=filename, line=line)
            return
        canonical = row["turns"] if dialogue_format else [row]
        standard = {turn["turn_id"]: (turn["meme_id"], turn["confidence"], str(turn.get("reason", "")).strip()) for turn in canonical}
        requested = set(row.get("requested_turn_ids", standard)) if dialogue_format else {row["turn_id"]}
        for value in objects:
            try:
                signature = candidate_signature(value, dialogue_format, row.get("turn_id"))
            except (ValueError, TypeError, AttributeError):
                issue(identifier, "invalid_raw_response", annotator=annotator, file=filename, line=line)
                continue
            if set(signature) != requested:
                issue(identifier, "raw_response_turn_mismatch", annotator=annotator, file=filename, line=line)
                continue
            for index, annotation in signature.items():
                if index not in standard or annotation[:2] != standard[index][:2]:
                    issue(identifier, "raw_label_or_confidence_conflict", annotator=annotator, file=filename, line=line, turn_id=index)
                elif annotation[2] != standard[index][2]:
                    reason_only.add(identifier)

    for annotator in ANNOTATORS:
        filename = f"annotations_dialogue_pool_{annotator}.jsonl"
        new[annotator] = load_rows(directory / filename)
        indexed[annotator] = {}
        for line, row in enumerate(new[annotator], 1):
            identifier = row["dialogue_id"]
            if identifier not in dialogues:
                raise ValueError(f"标注引用不存在的对话：{identifier}")
            if identifier in indexed[annotator]:
                issue(identifier, "duplicate_dialogue_annotation", annotator=annotator)
            indexed[annotator][identifier] = row
            if row.get("annotator") != annotator or row.get("source_split") != dialogues[identifier]["source_split"]:
                issue(identifier, "annotator_or_split_mismatch", annotator=annotator)
            if not isinstance(row.get("turns"), list):
                issue(identifier, "invalid_annotation_turns", annotator=annotator)
                continue
            if [turn.get("turn_id") for turn in row["turns"]] != list(range(len(dialogues[identifier]["turns"]))):
                issue(identifier, "incomplete_or_duplicate_turns", annotator=annotator)
            for turn in row["turns"]:
                inspect_label(identifier, turn, annotator, filename, line)
            inspect_response(identifier, row, annotator, filename, line, True)
        filename = f"annotations_pool_{annotator}.jsonl"
        # Modern exports contain complete dialogue-level annotations only.
        # Audit legacy turn rows when supplied, without requiring or inventing
        # an empty legacy file for a newer export.
        legacy_path = directory / filename
        legacy[annotator] = load_rows(legacy_path) if legacy_path.is_file() else []
        seen = set()
        for line, row in enumerate(legacy[annotator], 1):
            identifier = row["dialogue_id"]
            if identifier not in dialogues:
                raise ValueError(f"旧标注引用不存在的对话：{identifier}")
            index = row.get("turn_id")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(dialogues[identifier]["turns"]):
                issue(identifier, "invalid_legacy_turn_id", annotator=annotator)
                continue
            key = (identifier, index)
            if key in seen:
                issue(identifier, "duplicate_legacy_turn", annotator=annotator, turn_id=index)
            seen.add(key)
            if row.get("annotator") != annotator or row.get("role") != ("A" if index % 2 == 0 else "B"):
                issue(identifier, "legacy_annotator_or_role_mismatch", annotator=annotator, turn_id=index)
            text = dialogues[identifier]["turns"][index]
            context = "\n".join(f"{'A' if i % 2 == 0 else 'B'}: {text}" for i, text in enumerate(dialogues[identifier]["turns"][:index]))
            if row.get("text") != text or row.get("context") != context:
                issue(identifier, "legacy_text_or_context_mismatch", annotator=annotator, turn_id=index)
            inspect_label(identifier, row, annotator, filename, line)
            parent = indexed[annotator].get(identifier)
            inherited = {t["turn_id"]: t for t in parent["turns"]}.get(index) if parent else None
            if inherited is None or any(row.get(field) != inherited.get(field) for field in ("meme_id", "confidence", "reason")):
                issue(identifier, "legacy_label_mismatch", annotator=annotator, turn_id=index)
            inspect_response(identifier, row, annotator, filename, line, False)

    coverage = {identifier: sorted(annotator for annotator in ANNOTATORS if identifier in indexed[annotator]) for identifier in dialogues}
    kept = [row for row in raw if len(coverage[row["dialogue_id"]]) == 4 and row["dialogue_id"] not in issues]
    return {"raw": raw, "dialogues": dialogues, "catalog": catalog, "new": new, "legacy": legacy, "indexed": indexed,
            "coverage": coverage, "issues": dict(issues), "kept": kept, "reason_only": sorted(reason_only)}


def statistics(data):
    splits = defaultdict(lambda: {"dialogues": 0, "turns": 0})
    confidence = Counter()
    memes = Counter()
    for row in data["raw"]:
        splits[row["source_split"]]["dialogues"] += 1
        splits[row["source_split"]]["turns"] += len(row["turns"])
    for rows in data["new"].values():
        for row in rows:
            for turn in row["turns"]:
                confidence[turn["confidence"]] += 1
                memes[turn["meme_id"]] += 1
    return {"dialogues": len(data["raw"]), "turns": sum(len(row["turns"]) for row in data["raw"]),
            "source_split": dict(sorted(splits.items())), "catalog_meme_ids": len(data["catalog"]),
            "used_meme_ids": len(memes), "new_dialogue_rows_per_annotator": {key: len(value) for key, value in data["new"].items()},
            "new_turn_annotations_per_annotator": {key: sum(len(row["turns"]) for row in value) for key, value in data["new"].items()},
            "new_turn_annotations_total": sum(confidence.values()),
            "legacy_rows_per_annotator": {key: len(value) for key, value in data["legacy"].items()},
            "confidence_distribution_new_only": dict(sorted(confidence.items()))}


def prefix_pairs(rows, *, normalize=False):
    """Find only whole-turn, proper prefixes; equal-length records are kept."""
    def clean(text):
        text = " ".join(text.split())
        text = re.sub(r"(?<=[\u3400-\u9fff\uf900-\ufaff])\s+|\s+(?=[\u3400-\u9fff\uf900-\ufaff])", "", text)
        return re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    grouped = defaultdict(list)
    for row in rows:
        sequence = tuple(clean(text) for text in row["turns"]) if normalize else tuple(row["turns"])
        grouped[sequence].append(row)
    matches = []
    for sequence, long_rows in grouped.items():
        for length in range(1, len(sequence)):
            for short in grouped.get(sequence[:length], []):
                for long in long_rows:
                    matches.append({"short_dialogue_id": short["dialogue_id"], "short_original_id": short.get("original_dialogue_id", short["dialogue_id"]),
                        "short_turns": length, "long_dialogue_id": long["dialogue_id"], "long_original_id": long.get("original_dialogue_id", long["dialogue_id"]),
                        "long_turns": len(sequence), "short_source_split": short["source_split"], "long_source_split": long["source_split"]})
    return matches


def prefix_audit(rows):
    result = {}
    for mode in ("exact", "normalized"):
        normalize = mode == "normalized"
        def clean(text):
            text = " ".join(text.split())
            text = re.sub(r"(?<=[\u3400-\u9fff\uf900-\ufaff])\s+|\s+(?=[\u3400-\u9fff\uf900-\ufaff])", "", text)
            return re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        grouped = defaultdict(list)
        for row in rows:
            sequence = tuple(clean(text) for text in row["turns"]) if normalize else tuple(row["turns"])
            grouped[sequence].append(row)
        duplicates = [{"dialogue_ids": [row["dialogue_id"] for row in values],
                       "original_dialogue_ids": [row.get("original_dialogue_id", row["dialogue_id"]) for row in values],
                       "turns": len(sequence)} for sequence, values in grouped.items() if len(values) > 1]
        pairs = prefix_pairs(rows, normalize=normalize)
        result[mode] = {"dialogues": len(rows), "unique_sequences": len(grouped), "duplicate_groups": duplicates,
                        "duplicate_extra_dialogues": sum(len(group["dialogue_ids"]) - 1 for group in duplicates),
                        "strict_prefix_pair_count": len(pairs), "strict_prefix_pairs": pairs}
    return result


def run(root, apply=False, remove_exact_prefixes=False):
    root = Path(root).resolve()
    source = root / "data"
    data = audit(source)
    before = statistics(data)
    pairs = prefix_pairs(data["kept"]) if remove_exact_prefixes else []
    shorter_ids = {pair["short_dialogue_id"] for pair in pairs}
    matched_longer_ids = defaultdict(set)
    for pair in pairs:
        matched_longer_ids[pair["short_dialogue_id"]].add(pair["long_dialogue_id"])
    data["kept"] = [row for row in data["kept"] if row["dialogue_id"] not in shorter_ids]
    keep_ids = {row["dialogue_id"] for row in data["kept"]}
    category = Counter()
    deletions = []
    for row in data["raw"]:
        identifier = row["dialogue_id"]
        if identifier in keep_ids:
            continue
        reasons = []
        if len(data["coverage"][identifier]) < 4:
            reasons.append("no_annotations" if not data["coverage"][identifier] else "missing_annotators")
        if identifier in data["issues"]:
            reasons.append("invalid_annotation")
        if identifier in shorter_ids:
            reasons.append("exact_dialogue_prefix")
        category.update(reasons)
        deletions.append({"old_dialogue_id": identifier, "source_split": row["source_split"], "turns": len(row["turns"]),
                          "original_dialogue_id": row.get("original_dialogue_id", identifier),
                          "reasons": reasons, "present_annotators": data["coverage"][identifier], "issues": data["issues"].get(identifier, []),
                          "matching_longer_dialogue_ids": sorted(matched_longer_ids[identifier])})
    report = {"status": "dry_run", "rules": ["Keep only dialogues fully annotated by paper1-paper4.",
        "Delete entire dialogue for malformed labels, visible replacement characters, explicit failure placeholders, or conflicting raw meme/confidence values.",
        "Do not reject cross-annotator disagreement, valid low confidence, formatting-only raw response differences or reason-only rewrites."],
        "before": before, "retained_dialogues": len(keep_ids),
        "retained_turns": sum(len(row["turns"]) for row in data["kept"]),
        "deleted_dialogues": len(deletions), "deletion_categories": dict(category),
        "invalid_annotation_dialogues": len(data["issues"]),
        "invalid_issue_categories_by_dialogue": dict(Counter(code for details in data["issues"].values() for code in {detail["code"] for detail in details})),
        "retained_reason_only_rewrites": [identifier for identifier in data["reason_only"] if identifier in keep_ids]}
    if remove_exact_prefixes:
        report.update(operation="remove_exact_dialogue_prefixes", exact_prefix_pairs=pairs,
                      comparison="Exact original strings, entire turns in order, and strictly fewer turns. No whitespace normalization or fuzzy matching.")
        report["rules"].append("Remove only the shorter proper-prefix dialogue; retain all longer continuations and equal-length records.")
    if not apply:
        return report
    if not keep_ids:
        raise ValueError("清理将删除全部数据，停止操作")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    backup = root / "backups" / f"before_{'prefix_cleaning' if remove_exact_prefixes else 'cleaning'}_{timestamp}"
    backup.mkdir(parents=True)
    original_hashes = {path.name: sha256(path) for path in source.iterdir() if path.is_file()}
    archive_hash = sha256(root / "data.7z") if (root / "data.7z").exists() else None
    for name in ("statistics.json", "annotation_coverage_audit.json", "example_dialogue.json", "cleaning_report.json", "dialogue_id_mapping.jsonl", "deleted_dialogues.jsonl", "dialogue_prefix_audit.json", "prefix_cleaning_report.json"):
        if (root / name).is_file():
            shutil.copy2(root / name, backup / name)
    counters = Counter()
    mapping = {}
    mapping_rows = []
    for row in data["kept"]:
        identifier = row["dialogue_id"]
        split = row["source_split"]
        new_id = f"lccc-{split}-{counters[split]:08d}"
        counters[split] += 1
        mapping[identifier] = new_id
        mapping_rows.append({"old_dialogue_id": identifier, "new_dialogue_id": new_id,
                             "original_dialogue_id": row.get("original_dialogue_id", identifier), "source_split": split})
    def renumber(row):
        identifier = row["dialogue_id"]
        return {**row, "dialogue_id": mapping[identifier], "original_dialogue_id": row.get("original_dialogue_id", identifier)}
    with tempfile.TemporaryDirectory(prefix=".clean-stage-", dir=root) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        for path in source.iterdir():
            if path.is_file() and path.name not in {"dialogues_pool.jsonl", *(f"annotations_dialogue_pool_{a}.jsonl" for a in ANNOTATORS), *(f"annotations_pool_{a}.jsonl" for a in ANNOTATORS)}:
                shutil.copy2(path, stage / path.name)
        write_rows(stage / "dialogues_pool.jsonl", (renumber(row) for row in data["kept"]))
        for annotator in ANNOTATORS:
            write_rows(stage / f"annotations_dialogue_pool_{annotator}.jsonl", (renumber(data["indexed"][annotator][row["dialogue_id"]]) for row in data["kept"]))
            legacy_name = f"annotations_pool_{annotator}.jsonl"
            if (source / legacy_name).is_file():
                old_rows = [renumber(row) for row in data["legacy"][annotator] if row["dialogue_id"] in keep_ids]
                old_rows.sort(key=lambda row: (row["dialogue_id"], row["turn_id"]))
                write_rows(stage / legacy_name, old_rows)
        checked = audit(stage)
        if checked["issues"] or len(checked["kept"]) != len(keep_ids):
            raise ValueError("暂存数据未通过清理后完整性校验，原始数据未修改")
        if remove_exact_prefixes and prefix_pairs(checked["raw"]):
            raise ValueError("暂存数据中仍有严格前缀，原始数据未修改")
        for split, count in counters.items():
            ids = sorted(row["dialogue_id"] for row in checked["raw"] if row["source_split"] == split)
            if ids != [f"lccc-{split}-{i:08d}" for i in range(count)]:
                raise ValueError("重编号不连续")
        report["after"] = statistics(checked)
        # Original files are moved intact, not rewritten in place. Roll back
        # immediately if installing the already validated stage fails.
        source.rename(backup / "data")
        try:
            stage.rename(source)
        except BaseException:
            (backup / "data").rename(source)
            raise
    for name, digest in original_hashes.items():
        if sha256(backup / "data" / name) != digest:
            raise RuntimeError("备份文件校验失败")
    if archive_hash and sha256(root / "data.7z") != archive_hash:
        raise RuntimeError("原压缩包发生变化")
    (backup / "SHA256.json").write_text(json.dumps(original_hashes, indent=2) + "\n", encoding="utf-8")
    report.update(status="complete", backup_directory=str(backup), archive_sha256_unchanged=archive_hash,
                  numbering="Zero-based 8-digit dialogue IDs within each original source_split; turn IDs and meme IDs unchanged.")
    write_rows(root / "dialogue_id_mapping.jsonl", mapping_rows)
    write_rows(root / "deleted_dialogues.jsonl", deletions)
    (root / "dialogue_prefix_audit.json").write_text(json.dumps(prefix_audit(checked["raw"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    after = report["after"]
    (root / "statistics.json").write_text(json.dumps({"status": "cleaned", **after}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "annotation_coverage_audit.json").write_text(json.dumps({"status": "cleaned", "expected_annotators": list(ANNOTATORS),
        "total_dialogues": after["dialogues"], "total_turns": after["turns"],
        "dialogue_coverage": {str(i): after["dialogues"] if i == 4 else 0 for i in range(5)},
        "turn_coverage": {str(i): after["turns"] if i == 4 else 0 for i in range(5)},
        "partial_dialogues": [], "validation_issues": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = next((row for row in data["kept"] if row.get("original_dialogue_id", row["dialogue_id"]) == "lccc-dev-00000004"), data["kept"][0])
    annotation = data["indexed"]["paper1"][selected["dialogue_id"]]
    labels = {turn["turn_id"]: turn for turn in annotation["turns"]}
    example = {"dialogue_id": mapping[selected["dialogue_id"]], "original_dialogue_id": selected.get("original_dialogue_id", selected["dialogue_id"]),
               "source_split": selected["source_split"], "annotator": "paper1", "messages": [
                   {"turn_id": i, "role": "A" if i % 2 == 0 else "B", "text": re.sub(r"(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])", "", text),
                    **{field: labels[i][field] for field in ("meme_id", "confidence", "reason")}} for i, text in enumerate(selected["turns"])]}
    (root / "example_dialogue.json").write_text(json.dumps(example, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if remove_exact_prefixes:
        (root / "prefix_cleaning_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "dataset/v.1.0.0")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--remove-exact-prefixes", action="store_true", help="删除原文逐轮完全一致的较短前缀，不做归一化")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.apply, args.remove_exact_prefixes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
