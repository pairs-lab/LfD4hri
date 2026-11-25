"""Immutable full-size view replacing only a verified bad recovery scene pair."""
from collections import Counter
import json
import os
from pathlib import Path

from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "VLA/datasets/candidates/wrist_v8_closed_empty_visible_64_allvocab_20261008"
FIX = ROOT / "VLA/datasets/candidates/wrist_v8_closed_empty_goalfix_regression_20261008"
NEW = ROOT / "VLA/datasets/candidates/wrist_v8_closed_empty_visible_goalfix_64_20261008"


def main():
    if NEW.exists():
        raise FileExistsError(NEW)
    before_audit = json.loads((ROOT / "VLA/audits/finetune_v8_closed_empty_full_20261008/source/audit.json").read_text())
    if before_audit["failures"] != ["physical place target has <25 pixels across both cameras at reset"]:
        raise ValueError("Unexpected failures require investigation, not this narrow repair")
    fixed_audit = json.loads((FIX / "audit.json").read_text())
    if not fixed_audit["passed"]:
        raise ValueError("Regression collection did not pass")
    rows = json.loads((OLD / "index.json").read_text())
    corrected = json.loads((FIX / "index.json").read_text())
    replacements = {r["variant"]: r for r in corrected if r["task"]["name"] == "pick_place"}
    changed = []
    merged = []
    NEW.mkdir(parents=True)
    for row in rows:
        if row["paired_scene_source"] == "pick_place_00060.npz":
            replacement = replacements[row["variant"]]
            if (physical_scene_key(row["task"]) != physical_scene_key(replacement["task"])
                    or row["task"] != replacement["task"]
                    or row["observation_profile_sha256"] != replacement["observation_profile_sha256"]):
                raise ValueError("Repair changed task/geometry/sensor profile")
            origin, filename = FIX, replacement["file"]
            metadata = {**replacement, "file": row["file"]}
            changed.append(row["file"])
        else:
            origin, filename, metadata = OLD, row["file"], row
        os.link(origin / filename, NEW / row["file"])
        merged.append({**metadata, "repair_origin": {"source": str(origin), "file": filename,
                                                    "sha256": sha256_file(origin / filename)}})
    if len(changed) != 2 or len(merged) != len(rows):
        raise ValueError("Must replace exactly the identified pair, not discard failures")
    (NEW / "index.json").write_text(json.dumps(merged, indent=2) + "\n")
    paired = json.loads((OLD / "audit.json").read_text())
    fixed_pair = next(p for p in fixed_audit["pairs"] if p["source_episode"] == "pick_place_00060.npz")
    paired.update(pairs=[fixed_pair if p["source_episode"] == "pick_place_00060.npz" else p for p in paired["pairs"]],
        immutable_repair_of=str(OLD), changed_files=changed, repair_collection_audit_sha256=sha256_file(FIX / "audit.json"))
    (NEW / "audit.json").write_text(json.dumps(paired, indent=2) + "\n")
    manifest = json.loads((OLD / "dataset_manifest.json").read_text())
    manifest.update(index_sha256=sha256_file(NEW / "index.json"), paired_audit_sha256=sha256_file(NEW / "audit.json"),
        purpose="Goal-visibility corrective candidate; requires fresh exhaustive qualification",
        immutable_repair_of=str(OLD), repair_collection=str(FIX), changed_files=changed,
        declared_task_counts=dict(Counter(r["task"]["name"] for r in merged)))
    (NEW / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(merged), "changed_files": changed, "old_view_unchanged": True}))


if __name__ == "__main__":
    main()
