#!/usr/bin/env python3
"""Download the pinned official Qwen3-8B checkpoint and verify every byte.

Run with the project's Python environment. Hugging Face's local_dir mode stores
the checkpoint once and keeps resumable partials/metadata inside that directory.
If HF's large-file service is unavailable, --source auto permits the official
Qwen ModelScope mirror, accepting it only against Hugging Face's pinned hashes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import struct
import threading
import time
import traceback
from datetime import datetime, timezone
from urllib.parse import quote

# Downloading is explicitly online, even when called from an offline training
# shell. Disable Xet so HTTP range downloads can resume without a second cache.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download
import httpx


REPO_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ROOT = Path(__file__).resolve().parents[1]
LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digests(path: Path, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest(), git_sha1.hexdigest()


def verify_file(path: Path, metadata: dict) -> dict:
    actual_size = path.stat().st_size
    if actual_size != metadata["size"]:
        raise ValueError(f"{path.name}: size {actual_size} != {metadata['size']}")
    sha256, git_sha1 = digests(path, actual_size)
    if metadata.get("sha256") and sha256 != metadata["sha256"]:
        raise ValueError(f"{path.name}: SHA256 mismatch")
    if metadata.get("git_blob_sha1") and git_sha1 != metadata["git_blob_sha1"]:
        raise ValueError(f"{path.name}: Git blob SHA1 mismatch")
    return {"size": actual_size, "sha256": sha256, "verified_at": now()}


def verify_safetensors(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise ValueError("config.json model_type is not qwen3")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError("The safetensors index has an empty weight_map")
    shards = set(weight_map.values())
    actual_shards = {path.name for path in directory.glob("*.safetensors")}
    if shards != actual_shards:
        raise ValueError(f"Shard/index mismatch: {sorted(shards ^ actual_shards)}")
    tensor_count, total_data_bytes = 0, 0
    for shard in sorted(shards):
        if Path(shard).name != shard:
            raise ValueError(f"Unsafe shard filename: {shard}")
        path = directory / shard
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                raise ValueError(f"{shard}: missing safetensors header")
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size > min(path.stat().st_size - 8, 100_000_000):
                raise ValueError(f"{shard}: invalid safetensors header length")
            header = json.loads(handle.read(header_size))
        tensor_names = set(header) - {"__metadata__"}
        expected_names = {key for key, value in weight_map.items() if value == shard}
        if tensor_names != expected_names:
            raise ValueError(f"{shard}: tensor names differ from the index")
        offsets = sorted(tuple(header[key]["data_offsets"]) for key in tensor_names)
        previous_end = 0
        for start, end in offsets:
            if start != previous_end or end < start:
                raise ValueError(f"{shard}: noncontiguous or invalid tensor offsets")
            previous_end = end
        if 8 + header_size + previous_end != path.stat().st_size:
            raise ValueError(f"{shard}: tensor payload length mismatch")
        tensor_count += len(tensor_names)
        total_data_bytes += previous_end
    if index.get("metadata", {}).get("total_size") != total_data_bytes:
        raise ValueError("The safetensors index total_size differs from shard payloads")
    return {"model_type": "qwen3", "shards": sorted(shards),
            "tensor_count": tensor_count, "total_tensor_bytes": total_data_bytes}


def partial_paths(directory: Path, filename: str, metadata: dict) -> list[Path]:
    cache = directory / ".cache/huggingface/download"
    paths = set(cache.glob(filename + ".*.incomplete"))
    # Recent huggingface_hub versions encode the filename in the cache path.
    etag = metadata.get("sha256") or metadata.get("git_blob_sha1")
    if etag:
        paths.update(cache.glob(f"*.{etag}.incomplete"))
    return list(paths)


def mirror_download(filename: str, directory: Path, expected_size: int, expected_sha256: str) -> str:
    """Resume an official mirror transfer; its bytes still need HF verification."""
    url = f"https://modelscope.cn/models/{REPO_ID}/resolve/master/{quote(filename)}"
    path = directory / filename
    partial = directory / (filename + ".partial")
    # Reuse HF's incomplete HTTP transfer instead of keeping two large partials.
    hf_partials = sorted(partial_paths(directory, filename, {"sha256": expected_sha256}),
                         key=lambda p: p.stat().st_size, reverse=True)
    if hf_partials and (not partial.exists() or hf_partials[0].stat().st_size > partial.stat().st_size):
        hf_partials[0].replace(partial)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    if offset == expected_size:
        partial.replace(path)
        return url
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(120, connect=30)) as client:
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            if offset and response.status_code == 206:
                if not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                    raise ValueError(f"{filename}: incorrect mirror Content-Range")
            elif offset:
                offset = 0  # Mirror ignored Range: restart this file safely.
            with partial.open("ab" if offset else "wb") as handle:
                for chunk in response.iter_bytes(chunk_size=8 * 1024 * 1024):
                    handle.write(chunk)
    if partial.stat().st_size != expected_size:
        raise ValueError(f"{filename}: mirror transfer incomplete")
    partial.replace(path)
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backbones/Qwen3-8B")
    parser.add_argument("--revision", default=REVISION, help="Full immutable HF commit SHA")
    parser.add_argument("--source", choices=("auto", "huggingface", "modelscope"), default="auto")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a full, immutable 40-character commit SHA")
    if args.workers < 1 or args.attempts < 1:
        parser.error("--workers and --attempts must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # A second pipeline invocation waits for the first download and then reuses
    # its immutable manifest. The lock itself never contains checkpoint bytes.
    process_lock = (directory / ".download.lock").open("a+")
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another downloader owns this checkpoint; waiting for its lock", flush=True)
        fcntl.flock(process_lock, fcntl.LOCK_EX)
    manifest_path = directory / "download_manifest.json"
    state_path = ROOT / "logs/download_state.json"
    previous_manifest = {}
    if manifest_path.exists():
        try:
            previous_manifest = json.loads(manifest_path.read_text())
        except (ValueError, OSError):
            pass
    if (previous_manifest.get("status") == "complete"
            and previous_manifest.get("repo_id") == REPO_ID
            and previous_manifest.get("revision") == args.revision):
        try:
            expected_files = previous_manifest["expected_files"]
            if not expected_files:
                raise ValueError("Completed manifest has no official file metadata")
            for filename, metadata in expected_files.items():
                verify_file(directory / filename, metadata)
            verify_safetensors(directory)
            print(f"Checkpoint already complete and verified: {directory}; manifest unchanged", flush=True)
            return 0
        except (KeyError, ValueError, OSError) as error:
            print(f"Existing completed checkpoint needs repair: {error}", flush=True)
    state = {"repo_id": REPO_ID, "revision": args.revision, "pid": os.getpid(),
             "output_dir": str(directory), "started_at": now(), "status": "fetching_metadata",
             "files": {}}
    manifest = {"repo_id": REPO_ID, "revision": args.revision,
                "official_metadata_source": "https://huggingface.co/" + REPO_ID,
                "status": "in_progress", "started_at": state["started_at"], "files": {}}
    progress_stop = threading.Event()

    def update(filename: str | None = None, **changes) -> None:
        with LOCK:
            if filename:
                state["files"].setdefault(filename, {}).update(changes)
            else:
                state.update(changes)
            state["updated_at"] = now()
            write_json(state_path, state)

    update()
    write_json(manifest_path, manifest)

    def interrupted(signum, _frame) -> None:
        progress_stop.set()
        message = f"Interrupted by signal {signum}; checkpoint partials are resumable"
        with LOCK:
            manifest.update(status="interrupted", interrupted_at=now(), error=message)
            write_json(manifest_path, manifest)
            update(status="interrupted", error=message)
        print(message, flush=True)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Never trust mirror metadata for the authoritative revision/hash list.
        info = HfApi(endpoint="https://huggingface.co").model_info(
            REPO_ID, revision=args.revision, files_metadata=True, timeout=60)
        if info.sha != args.revision:
            raise ValueError(f"HF resolved {args.revision} to a different commit: {info.sha}")
        metadata = {}
        for entry in info.siblings:
            if entry.rfilename.startswith("."):
                continue
            metadata[entry.rfilename] = {"size": entry.size,
                "sha256": entry.lfs.sha256 if entry.lfs else None,
                "git_blob_sha1": None if entry.lfs else entry.blob_id}
        for required in ("config.json", "tokenizer.json", "tokenizer_config.json",
                         "model.safetensors.index.json"):
            if required not in metadata:
                raise ValueError(f"Official checkpoint lacks required file {required}")
        manifest["expected_files"] = metadata
        manifest["expected_total_bytes"] = sum(item["size"] for item in metadata.values())
        write_json(manifest_path, manifest)
        update(status="downloading", expected_total_bytes=manifest["expected_total_bytes"])
        print(f"Official revision: {args.revision}; {len(metadata)} files, "
              f"{manifest['expected_total_bytes']:,} bytes", flush=True)

        def report_progress() -> None:
            while not progress_stop.wait(15):
                downloaded = 0
                for filename, expected in metadata.items():
                    candidates = [directory / filename, directory / (filename + ".partial")]
                    candidates.extend(partial_paths(directory, filename, expected))
                    sizes = []
                    for candidate in candidates:
                        try:
                            sizes.append(candidate.stat().st_size)
                        except FileNotFoundError:
                            pass  # The transfer just atomically renamed its partial.
                    downloaded += min(max(sizes, default=0), expected["size"])
                update(downloaded_bytes=downloaded)
                print(f"Download progress: {downloaded:,}/{manifest['expected_total_bytes']:,} bytes "
                      f"({100 * downloaded / manifest['expected_total_bytes']:.1f}%)", flush=True)

        threading.Thread(target=report_progress, name="download-progress", daemon=True).start()

        def download(filename: str) -> tuple[str, dict]:
            expected = metadata[filename]
            path = directory / filename
            update(filename, status="checking", expected_size=expected["size"])
            if path.exists():
                try:
                    verified = verify_file(path, expected)
                    previous_file = previous_manifest.get("files", {}).get(filename, {})
                    if previous_file.get("sha256") == verified["sha256"]:
                        for key in ("source", "source_url"):
                            if key in previous_file:
                                verified[key] = previous_file[key]
                    verified.setdefault("source", "existing_local_verified_against_official_hf")
                    update(filename, status="verified", **verified)
                    print(f"Verified existing {filename}", flush=True)
                    return filename, verified
                except (ValueError, OSError):
                    if args.verify_only:
                        raise
                    path.unlink()  # Invalid file: a resumable new download is required.
            if args.verify_only:
                raise FileNotFoundError(path)
            last_error = None
            sources = ["huggingface", "modelscope"] if args.source == "auto" else [args.source]
            # Mirror fallback is only valid for LFS files with official SHA256.
            if not expected["sha256"]:
                sources = ["huggingface"]
            for source in sources:
                for attempt in range(1, args.attempts + 1):
                    update(filename, status="downloading", source=source, attempt=attempt)
                    print(f"Downloading {filename} from {source}, attempt {attempt}", flush=True)
                    try:
                        if source == "huggingface":
                            hf_hub_download(REPO_ID, filename, revision=args.revision,
                                            local_dir=directory, endpoint="https://huggingface.co")
                            source_url = f"https://huggingface.co/{REPO_ID}/resolve/{args.revision}/{filename}"
                        else:
                            source_url = mirror_download(filename, directory, expected["size"],
                                                         expected["sha256"])
                        update(filename, status="verifying")
                        verified = verify_file(path, expected)
                        verified.update(source=source, source_url=source_url)
                        update(filename, status="verified", **verified)
                        print(f"Verified {filename}: {verified['sha256']}", flush=True)
                        return filename, verified
                    except Exception as error:
                        last_error = error
                        update(filename, status="retry_pending", error=f"{type(error).__name__}: {error}")
                        print(f"{filename}: {type(error).__name__}: {error}", flush=True)
                        if path.exists():
                            path.unlink()  # A failed completed checksum must never be reused.
                        if attempt < args.attempts:
                            time.sleep(min(10 * attempt, 30))
            update(filename, status="failed", error=str(last_error))
            raise RuntimeError(f"{filename}: all download attempts failed") from last_error

        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(download, name): name for name in metadata}
            for future in concurrent.futures.as_completed(futures):
                try:
                    filename, verified = future.result()
                    manifest["files"][filename] = verified
                    write_json(manifest_path, manifest)
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
        if errors:
            raise RuntimeError("\n".join(errors))
        update(status="validating_safetensors")
        manifest["validation"] = verify_safetensors(directory)
        manifest.update(status="complete", completed_at=now())
        write_json(manifest_path, manifest)
        update(status="complete", completed_at=manifest["completed_at"],
               verified_total_bytes=sum(item["size"] for item in manifest["files"].values()),
               validation=manifest["validation"])
        print(f"Checkpoint complete and verified: {directory}", flush=True)
        return 0
    except BaseException as error:
        manifest.update(status="failed", failed_at=now(), error=f"{type(error).__name__}: {error}")
        write_json(manifest_path, manifest)
        update(status="failed", error=manifest["error"])
        traceback.print_exc()
        return 1
    finally:
        progress_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
