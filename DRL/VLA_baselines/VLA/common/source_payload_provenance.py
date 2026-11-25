"""Full content provenance for exactly the NPZ episodes named by an index.

Unlike a recursive artifact-directory fingerprint, this role-specific digest
ignores logs, audit reports and unlisted episodes. Verification always rehashes
the indexed payloads; unchanged index bytes, file sizes or mtimes are not proof
that image/action payloads are unchanged. Index metadata itself remains the
responsibility of the caller's separate index fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from VLA.common.provenance import sha256_file


SOURCE_PAYLOAD_FINGERPRINT_VERSION = 1
SOURCE_PAYLOAD_ROLE = "indexed_npz_episode_payloads"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _index_rows(source: Path, index: Any) -> Sequence[Mapping[str, Any]]:
    if index is None or isinstance(index, (str, Path)):
        path = source / "index.json" if index is None else Path(index)
        index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, Sequence) or isinstance(index, (str, bytes)):
        raise ValueError("Source index must be a sequence of episode rows")
    if not index:
        raise ValueError("Source index contains no NPZ episode payloads")
    return index


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Indexed NPZ filename must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in (".", ""):
        raise ValueError(f"Indexed NPZ filename must be relative inside the source: {value!r}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Indexed payload is not an NPZ file: {value!r}")
    return path.as_posix()


def _indexed_paths(source: Path, index: Any) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    resolved_paths: set[Path] = set()
    for row in _index_rows(source, index):
        if not isinstance(row, Mapping) or "file" not in row:
            raise ValueError("Every source index row must contain an NPZ 'file'")
        name = _relative_name(row["file"])
        if name in paths:
            raise ValueError(f"Duplicate indexed NPZ filename: {name}")
        path = source / name
        resolved = path.resolve()
        try:
            resolved.relative_to(source)
        except ValueError as exc:
            raise ValueError(f"Indexed NPZ resolves outside the source: {name}") from exc
        if resolved in resolved_paths:
            raise ValueError(f"Duplicate indexed NPZ path alias: {name}")
        if not path.exists():
            raise FileNotFoundError(f"Missing indexed NPZ payload: {path}")
        if not path.is_file():
            raise ValueError(f"Indexed NPZ payload is not a regular file: {path}")
        paths[name] = path
        resolved_paths.add(resolved)
    return dict(sorted(paths.items()))


def _tree_sha256(payloads: Mapping[str, Any]) -> str:
    encoded = json.dumps(payloads, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stat_identity(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def source_payload_fingerprint(source: str | Path, index: Any = None) -> dict[str, Any]:
    """Hash exactly indexed NPZ files and return a filename-bound hash map.

    ``index`` may be loaded episode rows or a path to an index JSON file. An
    explicit relative index path is relative to the current working directory;
    omitted ``index`` reads ``source/index.json``. Payload filenames must remain
    relative inside ``source``. Missing/nonfile/duplicate/escaping paths fail.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    payloads: dict[str, dict[str, Any]] = {}
    for name, path in _indexed_paths(source, index).items():
        before = _stat_identity(path)
        digest = sha256_file(path)
        after = _stat_identity(path)
        if before != after:
            raise RuntimeError(f"Source payload changed during fingerprinting: {name}")
        payloads[name] = {"sha256": digest, "bytes": after[2]}
    return {
        "version": SOURCE_PAYLOAD_FINGERPRINT_VERSION,
        "role": SOURCE_PAYLOAD_ROLE,
        "files": len(payloads),
        "bytes": sum(record["bytes"] for record in payloads.values()),
        "payload_tree_sha256": _tree_sha256(payloads),
        "payloads": payloads,
    }


def verify_source_payload_fingerprint(saved: Any, source: str | Path,
                                      index: Any = None) -> dict[str, Any]:
    """Fully verify saved provenance; missing/malformed/stale data fails closed.

    Returns a compact verified summary. File-content verification is never
    replaced by a metadata-only shortcut, even for identical sizes and mtimes.
    The source directory may be relocated without changing this content digest.
    """
    if not isinstance(saved, Mapping):
        raise RuntimeError("Missing or malformed source payload fingerprint")
    if type(saved.get("version")) is not int or saved["version"] != SOURCE_PAYLOAD_FINGERPRINT_VERSION:
        raise RuntimeError("Unsupported or missing source payload fingerprint version")
    if saved.get("role") != SOURCE_PAYLOAD_ROLE:
        raise RuntimeError("Missing or mismatched source payload fingerprint role")
    payloads = saved.get("payloads")
    if not isinstance(payloads, Mapping) or not payloads:
        raise RuntimeError("Source payload fingerprint has no payload hash map")
    try:
        normalized = {}
        for name, record in payloads.items():
            if _relative_name(name) != name:
                raise ValueError("Noncanonical fingerprint filename")
            if not isinstance(record, Mapping) or set(record) != {"sha256", "bytes"}:
                raise ValueError("Malformed payload hash record")
            if not isinstance(record["sha256"], str) or not _SHA256.fullmatch(record["sha256"]):
                raise ValueError("Malformed payload SHA-256")
            if type(record["bytes"]) is not int or record["bytes"] < 0:
                raise ValueError("Malformed payload byte count")
            normalized[name] = dict(record)
        internally_valid = (
            type(saved.get("files")) is int and saved["files"] == len(normalized)
            and type(saved.get("bytes")) is int
            and saved["bytes"] == sum(record["bytes"] for record in normalized.values())
            and saved.get("payload_tree_sha256") == _tree_sha256(normalized)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed source payload fingerprint records") from exc
    if not internally_valid:
        raise RuntimeError("Source payload fingerprint summary/hash is internally inconsistent")
    try:
        current = source_payload_fingerprint(source, index)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Source payload fingerprint verification failed: {exc}") from exc
    if set(current["payloads"]) != set(normalized):
        raise RuntimeError("Source payload fingerprint filenames differ from the current index")
    differing = [name for name in normalized if normalized[name] != current["payloads"][name]]
    if differing:
        raise RuntimeError(f"Source NPZ payload content/hash differs: {', '.join(sorted(differing))}")
    return {"verified": True, "mode": "full_indexed_npz_content", "role": SOURCE_PAYLOAD_ROLE,
            "files": current["files"], "bytes": current["bytes"],
            "payload_tree_sha256": current["payload_tree_sha256"]}


def verify_source_payload_for_profile(report, source):
    """Require content-bound audits for V9, preserving legacy compatibility."""
    profile = json.loads((Path(source)/"dataset_manifest.json").read_text()).get("observation_profile")
    fingerprint = report.get("source_payload_fingerprint")
    if (profile or {}).get("gripper_controller") == "body_contact_v9" or fingerprint is not None:
        return verify_source_payload_fingerprint(fingerprint, source)
