"""Eight audited TRAIN episodes for memorization sanity checks, not evaluation.

Select one regular scene with both text-selected targets per primitive. Keep
original payloads immutable; never draw from validation/final scenes.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from VLA.common.provenance import sha256_file


def select_pairs(index):
    selected = []
    for name in ("touch", "pick", "move", "pick_place"):
        rows = [r for r in index if r["task"]["name"] == name
                and not r.get("recovery_start_actions_not_training_labels")]
        if not rows:
            raise ValueError(f"No regular training scenes for {name}")
        source = rows[0]["paired_scene_source"]
        pair = [r for r in rows if r["paired_scene_source"] == source]
        if len(pair) != 2 or {r["variant"] for r in pair} != {0, 1}:
            raise ValueError("Each tiny-fit scene must have two target instructions")
        selected.extend(pair)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    audit = json.loads(args.source_audit.read_text())
    if (manifest.get("split") != "train" or not audit.get("passed") or audit.get("failures")
            or Path(audit.get("dataset", "")).resolve() != source
            or audit.get("index_sha256") != sha256_file(source / "index.json")
            or audit.get("manifest_sha256") != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Tiny fit requires immutable qualified TRAIN data")
    index = select_pairs(json.loads((source / "index.json").read_text()))
    for row in index:
        if sha256_file(source / row["file"]) != row["split_origin"]["sha256"]:
            raise ValueError("Source payload changed since audited scene split")
    args.output.mkdir(parents=True)
    for row in index:
        os.link(source / row["file"], args.output / row["file"])
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    child = {**manifest, "purpose": "Tiny training-set fit sanity check, NOT a benchmark or generalization dataset",
             "tiny_fit_only": True, "candidate_only": True,
             "parent_source": str(source), "parent_source_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
             "parent_source_audit_sha256": sha256_file(args.source_audit),
             "declared_task_counts": dict(Counter(r["task"]["name"] for r in index)),
             "frames": sum(r["steps"] for r in index), "index_sha256": sha256_file(args.output / "index.json")}
    (args.output / "dataset_manifest.json").write_text(json.dumps(child, indent=2) + "\n")
    reset_provenance = json.loads((source / "audit.json").read_text())
    (args.output / "audit.json").write_text(json.dumps({
        "source": reset_provenance["source"], "scope": "Canonical reset provenance only; not the new source audit",
        "parent": str(source), "final_test_used": False}, indent=2) + "\n")
    print(json.dumps({"episodes": len(index), "frames": child["frames"], "source": str(args.output)}))


if __name__ == "__main__":
    main()
