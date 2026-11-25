#!/usr/bin/env python3
"""Create a reusable full-content fingerprint for a completed checkpoint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


REPORT_NAME = "benchmark_fingerprint.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-existing", action="store_true",
                        help="Fully verify an existing report without rewriting it; create only if absent")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def layout_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.relative_to(root).as_posix() == REPORT_NAME:
            continue
        stat = path.stat()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return records


def layout_sha256(records: list[dict]) -> str:
    encoded = json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_fingerprint(checkpoint: Path, *, full: bool = False) -> dict:
    """Verify the full-hash report plus current layout, optionally rehashing payloads."""

    from VLA.common.provenance import verify_directory_fingerprint

    root = Path(checkpoint).resolve()
    report_path = root / REPORT_NAME
    if not report_path.is_file():
        raise FileNotFoundError(f"Finalize a full checkpoint fingerprint first: {report_path}")
    report = json.loads(report_path.read_text())
    if Path(report.get("checkpoint", "")).resolve() != root:
        raise RuntimeError("Checkpoint fingerprint names a different directory")
    verify_directory_fingerprint(
        root, {**report, "excluded_paths": [REPORT_NAME]}, full=full
    )
    return report


def main() -> None:
    args = parse_args()
    root = args.checkpoint.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    output = root / REPORT_NAME
    if args.overwrite and args.verify_existing:
        raise ValueError("--overwrite and --verify-existing are mutually exclusive")
    if output.exists() and args.verify_existing:
        report = verify_fingerprint(root, full=True)
        print(json.dumps({"existing_fingerprint_fully_verified": True,
                          "content_tree_sha256": report["content_tree_sha256"],
                          "report_modified": False}, indent=2))
        return
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")

    records = layout_records(root)
    if not records:
        raise RuntimeError(f"Checkpoint contains no files: {root}")
    tree = hashlib.sha256()
    content_records = []
    for index, record in enumerate(records, start=1):
        path = root / record["path"]
        digest = sha256(path)
        tree.update(record["path"].encode("utf-8"))
        tree.update(b"\0")
        tree.update(digest.encode("ascii"))
        tree.update(b"\n")
        content_records.append({**record, "sha256": digest})
        print(f"hashed {index}/{len(records)} {record['path']}", flush=True)

    report = {
        "version": 1,
        "checkpoint": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "layout_sha256": layout_sha256(records),
        "content_tree_sha256": tree.hexdigest(),
        "records": content_records,
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
