"""Restore Git-packaged datasets without downloading excluded model weights.

Run ``python model/tools/restore_git_assets.py`` from any working directory.
The default repository is this script's repository; ``--repo-root`` selects a
different checkout containing ``model/git_upload_manifest.json``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sys
import tempfile
from typing import BinaryIO
import zlib


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = "model/git_upload_manifest.json"
CHUNK_SIZE = 1024 * 1024


class AssetRestoreError(ValueError):
    """A manifest, package or existing file cannot be restored safely."""


def _repo_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise AssetRestoreError(f"Invalid repository path: {value!r}")
    relative = Path(value)
    windows = PureWindowsPath(value)
    if (relative.is_absolute() or windows.drive or windows.is_absolute()
            or ".." in relative.parts or ".." in windows.parts):
        raise AssetRestoreError(f"Path must be relative and cannot contain '..': {value!r}")
    resolved = (root / relative).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise AssetRestoreError(f"Path escapes repository root: {value!r}")
    return resolved


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise AssetRestoreError(f"Invalid SHA-256 for {label}")
    return value.lower()


def _metadata(entry: object, label: str) -> tuple[int, str]:
    if not isinstance(entry, dict):
        raise AssetRestoreError(f"{label} must be an object")
    size = entry.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise AssetRestoreError(f"Invalid byte count for {label}")
    return size, _digest(entry.get("sha256"), label)


def _read_manifest(root: Path) -> dict:
    path = _repo_path(root, MANIFEST_PATH)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise AssetRestoreError(f"Cannot read manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AssetRestoreError("Upload manifest must be an object")
    # Validate every path and all metadata before creating any output files.
    for section, fields in (
        ("packed_data", ("source_path", "packed_path")),
        ("duplicate_data", ("source_path", "target_path")),
        ("excluded_large_weights", ("path",)),
    ):
        entries = manifest.get(section)
        if not isinstance(entries, list):
            raise AssetRestoreError(f"Manifest {section} must be a list")
        for number, entry in enumerate(entries):
            label = f"{section}[{number}]"
            _metadata(entry, label)
            for field in fields:
                _repo_path(root, entry.get(field))
            if section == "packed_data":
                _digest(entry.get("packed_sha256"), label + ".packed_sha256")
    return manifest


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _already_present(path: Path, size: int, digest: str) -> bool:
    if not path.exists():
        return False
    if not path.is_file() or _file_identity(path) != (size, digest):
        raise AssetRestoreError(f"Refusing to overwrite conflicting existing file: {path}")
    return True


def _atomic_restore(source: BinaryIO, target: Path, size: int, expected_digest: str) -> bool:
    """Verify a sibling temporary file before publishing it with os.replace."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.restore-", delete=False) as output:
            temporary = Path(output.name)
            count = 0
            digest = hashlib.sha256()
            while chunk := source.read(CHUNK_SIZE):
                count += len(chunk)
                if count > size:
                    raise AssetRestoreError(f"Restored byte count exceeds manifest for {target}")
                digest.update(chunk)
                output.write(chunk)
            if count != size or digest.hexdigest() != expected_digest:
                raise AssetRestoreError(f"Restored bytes or SHA-256 mismatch for {target}")
            output.flush()
            os.fsync(output.fileno())
        # A matching file may have appeared while preparing the temporary file.
        # Never replace an existing file with conflicting contents.
        if _already_present(target, size, expected_digest):
            return False
        os.replace(temporary, target)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_assets(repo_root: str | Path = DEFAULT_REPO_ROOT) -> dict:
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise AssetRestoreError(f"Repository root is not a directory: {root}")
    manifest = _read_manifest(root)
    restored = unpacked = copied = skipped = 0
    for entry in manifest["packed_data"]:
        target = _repo_path(root, entry["source_path"])
        size, digest = _metadata(entry, entry["source_path"])
        if _already_present(target, size, digest):
            skipped += 1
            continue
        packed = _repo_path(root, entry["packed_path"])
        if not packed.is_file():
            raise AssetRestoreError(f"Missing packaged data: {entry['packed_path']}")
        if _file_identity(packed)[1] != _digest(entry["packed_sha256"], entry["packed_path"]):
            raise AssetRestoreError(f"Packed SHA-256 mismatch: {entry['packed_path']}")
        try:
            with gzip.open(packed, "rb") as source:
                changed = _atomic_restore(source, target, size, digest)
        except (OSError, EOFError, zlib.error) as exc:
            raise AssetRestoreError(f"Cannot decompress {entry['packed_path']}: {exc}") from exc
        unpacked += int(changed)
        restored += int(changed)
        skipped += int(not changed)
    for entry in manifest["duplicate_data"]:
        source = _repo_path(root, entry["source_path"])
        target = _repo_path(root, entry["target_path"])
        size, digest = _metadata(entry, entry["target_path"])
        if _already_present(target, size, digest):
            skipped += 1
            continue
        if not source.is_file():
            raise AssetRestoreError(f"Missing duplicate source: {entry['source_path']}")
        with source.open("rb") as stream:
            changed = _atomic_restore(stream, target, size, digest)
        copied += int(changed)
        restored += int(changed)
        skipped += int(not changed)
    missing_weights = []
    for entry in manifest["excluded_large_weights"]:
        path = _repo_path(root, entry["path"])
        size, digest = _metadata(entry, entry["path"])
        if not _already_present(path, size, digest):
            missing_weights.append({"path": entry["path"], "bytes": size, "sha256": digest})
    return {"repo_root": str(root), "restored_files": restored,
            "unpacked_files": unpacked, "copied_files": copied,
            "skipped_files": skipped, "missing_weights": missing_weights,
            "weights_downloaded": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验并恢复仓库 gzip 数据及归档副本；不下载模型权重")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                        help="包含 model/git_upload_manifest.json 的仓库根目录")
    args = parser.parse_args(argv)
    try:
        report = restore_assets(args.repo_root)
    except (AssetRestoreError, OSError) as exc:
        print(f"Asset restore failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
