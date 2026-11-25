"""Add qualified corrective demos without moving any frozen scene across splits.

Uses immutable hardlinks. Unknown geometry groups are rejected, not assigned
after seeing model outcomes. New source views require fresh audits/statistics.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--augmentation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base = {name: path.resolve() for name, path in (("train", args.train), ("validation", args.validation))}
    index = {name: json.loads((path / "index.json").read_text()) for name, path in base.items()}
    groups = {name: {physical_scene_key(r["task"]) for r in rows} for name, rows in index.items()}
    if groups["train"] & groups["validation"]:
        raise ValueError("Frozen base split already leaks scenes")
    aug = args.augmentation.resolve()
    audit = json.loads(args.augmentation_audit.read_text())
    aug_manifest = json.loads((aug / "dataset_manifest.json").read_text())
    train_manifest = json.loads((base["train"] / "dataset_manifest.json").read_text())
    if (not audit.get("passed") or audit.get("failures")
            or Path(audit.get("dataset", "")).resolve() != aug
            or audit.get("index_sha256") != sha256_file(aug / "index.json")
            or audit.get("manifest_sha256") != sha256_file(aug / "dataset_manifest.json")
            or aug_manifest["observation_profile_sha256"] != train_manifest["observation_profile_sha256"]):
        raise ValueError("Corrective augmentation is failed, stale or from another controller/sensor profile")
    augmented = json.loads((aug / "index.json").read_text())
    assigned = {name: [] for name in base}
    for row in augmented:
        group = physical_scene_key(row["task"])
        split = next((name for name, keys in groups.items() if group in keys), None)
        if split is None:
            raise ValueError("New geometry requires an explicit new split plan; do not leak it here")
        assigned[split].append(row)
    args.output.mkdir(parents=True)
    counts = {}
    for split, path in base.items():
        folder = args.output / split
        folder.mkdir()
        merged = []
        for origin, rows, prefix in ((path, index[split], "base"), (aug, assigned[split], "corrective")):
            for row in rows:
                digest = sha256_file(origin / row["file"])
                if origin == path and digest != row["split_origin"]["sha256"]:
                    raise ValueError("Frozen base payload changed")
                filename = prefix + "_" + row["file"]
                os.link(origin / row["file"], folder / filename)
                merged.append({**row, "file": filename, "split_origin": {"source": str(origin), "file": row["file"], "sha256": digest}})
        (folder / "index.json").write_text(json.dumps(merged, indent=2) + "\n")
        manifest = {**json.loads((path / "dataset_manifest.json").read_text()),
            "purpose": "Corrective-data diagnostic; frozen original geometry split retained",
            "candidate_only": True, "augmentation_source": str(aug),
            "augmentation_audit_sha256": sha256_file(args.augmentation_audit),
            "base_split_source": str(path), "base_split_index_sha256": sha256_file(path / "index.json"),
            "base_split_manifest_sha256": sha256_file(path / "dataset_manifest.json"),
            "declared_task_counts": dict(Counter(r["task"]["name"] for r in merged)),
            "frames": sum(r["steps"] for r in merged), "index_sha256": sha256_file(folder / "index.json"),
            "recovery_start_steps": "per_episode", "frozen_scene_split_preserved": True}
        (folder / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        canonical = json.loads((path / "audit.json").read_text())["source"]
        (folder / "audit.json").write_text(json.dumps({"source": canonical, "scope": "Reset provenance only", "final_test_used": False}, indent=2) + "\n")
        counts[split] = {"episodes": len(merged), "added_corrective_episodes": len(assigned[split]),
                         "frames": manifest["frames"], "layouts": len(groups[split]), "task_counts": manifest["declared_task_counts"]}
    (args.output / "split_extension.json").write_text(json.dumps({"splits": counts, "group_overlap": 0,
        "source_profiles_unchanged": True, "final_test_used": False,
        "scope": "Immutable split extension, NOT passed source/native-conversion audit"}, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
