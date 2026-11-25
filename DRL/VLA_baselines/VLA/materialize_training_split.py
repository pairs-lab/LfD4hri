"""Immutable, zero-copy NPZ views of a hashed, scene-disjoint pilot split.

Hard-linked payloads MUST NOT be edited. This never promotes pilot data to
production or uses validation episodes to compute training statistics.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize_split(args.readiness, args.split, args.output)


def materialize_split(readiness, split, output):
    from VLA.audit_training_readiness import freeze_split, source_reset_provenance, verify_readiness_source
    readiness, output = Path(readiness), Path(output)
    if split not in ("train", "validation"):
        raise ValueError("Unknown split")
    if output.exists():
        raise FileExistsError(output)
    report = json.loads(readiness.read_text())
    if not report["source_integrity_passed"] or report["failures"] or report["group_overlap_count"]:
        raise ValueError("Cannot materialize a failed or leaking split")
    split_path = readiness.parent / f"{split}_episodes.json"
    rows = json.loads(split_path.read_text())
    # A split file may not substitute unhashed/uninspected episodes.
    permitted = {(r["source"], r["file"]): r for r in report["payloads"]}
    source_data = {}
    canonical_sources = set()
    canonical_cache = {}
    for lineage in report["lineage"]:
        source = Path(lineage["source"])
        if sha256_file(source / "index.json") != lineage["source_index_sha256"] or sha256_file(source / "dataset_manifest.json") != lineage["source_manifest_sha256"]:
            raise ValueError("Source metadata changed after readiness audit")
        manifest = json.loads((source / "dataset_manifest.json").read_text())
        audit_path = Path(lineage["source_audit"])
        if sha256_file(audit_path) != lineage["source_audit_sha256"]:
            raise ValueError("Source audit changed after readiness audit")
        verify_readiness_source(source, manifest, json.loads(audit_path.read_text()))
        source_data[str(source)] = (manifest, {r["file"]: r for r in json.loads((source / "index.json").read_text())})
    keys = [(r["source"], r["file"]) for r in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("Empty split or duplicate payloads")
    if any(permitted.get(key) != row for key, row in zip(keys, rows, strict=True)):
        raise ValueError("Split changed after audit")
    for row in rows:
        if sha256_file(Path(row["source"]) / row["file"]) != row["sha256"]:
            raise ValueError("Source payload changed after audit")
    # Compare against the frozen deterministic assignment, not only membership.
    if rows != freeze_split(report["payloads"], seed=report["split_seed"],
                            preserve_training_vocabulary=report.get("preserve_training_vocabulary", False),
                            minimum_train_repetitions=report.get("minimum_train_repetitions", 1))[split]:
        raise ValueError("Split assignment changed")
    # Validate reset metadata before creating any partial output. RAW episodes
    # need no audit.json.source, and recovery episodes may have different parents.
    reset_metadata = {}
    for row in rows:
        original = source_data[row["source"]][1][row["file"]]
        reset = source_reset_provenance(row["source"], original, canonical_cache)
        if any(row.get(key) != value for key, value in reset.items()):
            raise ValueError("Canonical reset provenance changed after readiness audit")
        reset_metadata[(row["source"], row["file"])] = reset
        if reset:
            canonical_sources.add(reset["canonical_source"])
    output.mkdir(parents=True)
    index = []
    for row in rows:
        source = Path(row["source"])
        source_number = list(source_data).index(str(source))
        filename = f"source{source_number}_{row['file']}"
        os.link(source / row["file"], output / filename)
        original = source_data[str(source)][1][row["file"]]
        index.append({**original, **reset_metadata[(row["source"], row["file"])],
                      "file": filename, "scene_group": row["scene_group"],
                      "split_origin": {"source": str(source), "file": row["file"], "sha256": row["sha256"]}})
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    manifest = {**next(iter(source_data.values()))[0],
                "training_view": "vla_one_shot_success", "split": split,
                "parent_stratified_target_vocabulary": next(iter(source_data.values()))[0].get("stratified_target_vocabulary", False),
                "stratified_target_vocabulary": False,
                "vocabulary_contract": "Frozen grouped subset, not the parent's stratified collection schedule; exact coverage/counts in readiness report",
                "split_strategy": report.get("split_strategy", "hashed_scene_groups"),
                "candidate_only": True, "purpose": "Scene-disjoint native pipeline/fit canary, NOT final training sufficiency",
                "readiness_report": str(readiness.resolve()), "readiness_report_sha256": sha256_file(readiness),
                "declared_task_counts": dict(Counter(r["task"]["name"] for r in index)),
                "episodes": len(index), "episodes_per_task": None,
                "frames": sum(r["steps"] for r in index), "index_sha256": sha256_file(output / "index.json"),
                "recovery_start_steps": "per-episode, reset-only; excluded from training labels", "immutable_hardlinked_payloads": True}
    # Parent collection/camera evidence does not become a split source audit.
    for key in ("paired_audit_sha256", "camera_report_sha256"):
        manifest.pop(key, None)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "audit.json").write_text(json.dumps({
        **({"source": next(iter(canonical_sources))} if len(canonical_sources) == 1 else {}),
        "canonical_sources": sorted(canonical_sources),
        "scope": "Per-row canonical reset provenance ONLY; NOT camera/data or model-success audit",
        "parents": report["lineage"], "final_test_used": False,
    }, indent=2) + "\n")
    print(json.dumps({"split": split, "episodes": len(index), "frames": manifest["frames"], "output": str(output)}))


if __name__ == "__main__":
    main()
