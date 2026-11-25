"""Deterministic, reusable provenance fingerprints for local artifacts.

The full fingerprint hashes every file exactly once and stores per-file
records plus a path-bound content-tree digest.  Subsequent consumers can use
the fast verifier, which checks the complete path/size/mtime layout without
rereading large payloads, or request a full cryptographic recheck.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


FINGERPRINT_VERSION = 1
_HASH_BLOCK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_HASH_BLOCK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_exclusions(paths: Iterable[str | Path]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for value in paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Fingerprint exclusion must be relative to its root: {value}")
        text = relative.as_posix()
        if text in ("", "."):
            raise ValueError("Fingerprint exclusion cannot be the artifact root")
        normalized.add(text)
    return tuple(sorted(normalized))


def _candidate_files(root: Path, excluded_paths: tuple[str, ...]) -> list[Path]:
    excluded = set(excluded_paths)
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Artifact fingerprints do not follow symlinks: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def _layout_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def layout_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash the ordered path/size/mtime projection of fingerprint records."""

    layout = [
        {
            "path": str(record["path"]),
            "bytes": int(record["bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
        }
        for record in records
    ]
    encoded = json.dumps(layout, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def content_tree_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    """Hash ordered relative paths and their full file-content digests."""

    tree = hashlib.sha256()
    for record in records:
        path = str(record["path"])
        digest = str(record["sha256"])
        if len(digest) != 64:
            raise ValueError(f"Invalid SHA-256 record for {path!r}")
        tree.update(path.encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
    return tree.hexdigest()


def build_directory_fingerprint(
    root: Path,
    *,
    exclude_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Hash every non-excluded file below ``root`` and return full records."""

    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    excluded = _normalize_exclusions(exclude_paths)
    records: list[dict[str, Any]] = []
    for path in _candidate_files(root, excluded):
        before = _layout_record(path, root)
        digest = sha256_file(path)
        after = _layout_record(path, root)
        if before != after:
            raise RuntimeError(f"Artifact changed while it was being fingerprinted: {path}")
        records.append({**after, "sha256": digest})
    if not records:
        raise RuntimeError(f"Artifact contains no fingerprintable files: {root}")
    return {
        "version": FINGERPRINT_VERSION,
        "root": str(root),
        "excluded_paths": list(excluded),
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "layout_sha256": layout_sha256(records),
        "content_tree_sha256": content_tree_sha256(records),
        "records": records,
    }


def verify_directory_fingerprint(
    root: Path,
    fingerprint: Mapping[str, Any],
    *,
    full: bool = False,
) -> dict[str, Any]:
    """Verify a directory against a stored fingerprint.

    The default fast mode reads only directory metadata and compares every
    relative path, byte count, and nanosecond mtime.  ``full=True`` additionally
    rehashes all payloads and validates every stored per-file SHA-256 digest.
    A compact verified summary is returned; mismatches raise ``RuntimeError``.
    """

    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if fingerprint.get("version") != FINGERPRINT_VERSION:
        raise RuntimeError("Unsupported or missing artifact fingerprint version")
    try:
        excluded = _normalize_exclusions(fingerprint.get("excluded_paths", ()))
        stored_records = list(fingerprint["records"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed artifact fingerprint") from exc
    if not stored_records:
        raise RuntimeError("Artifact fingerprint has no content records")

    # First validate that the stored summary is internally consistent.  This
    # prevents a caller from trusting a tree/layout field detached from records.
    try:
        stored_layout = layout_sha256(stored_records)
        stored_tree = content_tree_sha256(stored_records)
        stored_bytes = sum(int(record["bytes"]) for record in stored_records)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed artifact fingerprint records") from exc
    if (
        fingerprint.get("files") != len(stored_records)
        or fingerprint.get("bytes") != stored_bytes
        or fingerprint.get("layout_sha256") != stored_layout
        or fingerprint.get("content_tree_sha256") != stored_tree
    ):
        raise RuntimeError("Artifact fingerprint summary is internally inconsistent")

    current_layout = [_layout_record(path, root) for path in _candidate_files(root, excluded)]
    stored_layout_records = [
        {
            "path": str(record["path"]),
            "bytes": int(record["bytes"]),
            "mtime_ns": int(record["mtime_ns"]),
        }
        for record in stored_records
    ]
    if current_layout != stored_layout_records:
        raise RuntimeError("Artifact path/size/mtime layout differs from its fingerprint")

    if full:
        current = build_directory_fingerprint(root, exclude_paths=excluded)
        if current["records"] != stored_records:
            raise RuntimeError("Artifact file content differs from its fingerprint")

    return {
        "verified": True,
        "mode": "full" if full else "fast_layout",
        "root": str(root),
        "files": len(stored_records),
        "bytes": stored_bytes,
        "layout_sha256": stored_layout,
        "content_tree_sha256": stored_tree,
    }
