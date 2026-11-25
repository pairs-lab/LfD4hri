"""Create an immutable, explicitly bounded fit-pilot view; NOT final sufficiency.

No candidate is edited in place. Source/camera, grouped split, physical native
replay and current controller checks must pass before even this small pilot.
Model-specific conversion/statistics/preprocessing gates remain in its trainer.
"""
import argparse
import json
import os
from pathlib import Path

from VLA.audit_training_readiness import verify_readiness_source
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_source_environment_runtime
from VLA.common.source_payload_provenance import verify_source_payload_for_profile


def qualify(source, source_audit, replay_reports, output, model="smolvla", max_updates=500):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if model != "smolvla" or type(max_updates) is not int or not 1 <= max_updates <= 1000:
        raise ValueError("This initial qualification is only for a bounded 1..1000-update dual-view SmolVLA pilot")
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    index = json.loads((source / "index.json").read_text())
    audit = json.loads(Path(source_audit).read_text())
    if (manifest.get("split") != "train" or manifest.get("required_success_steps") != 1
            or not audit.get("passed") or audit.get("failures")
            or Path(audit["dataset"]).resolve() != source
            or audit["index_sha256"] != sha256_file(source / "index.json")
            or audit["manifest_sha256"] != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Requires the exact successful one-shot TRAIN source audit")
    if not verify_readiness_source(source, manifest, audit):
        raise ValueError("This qualifier is specific to the corrected V9 controller")
    readiness_path = Path(manifest["readiness_report"])
    if sha256_file(readiness_path) != manifest["readiness_report_sha256"]:
        raise ValueError("Frozen readiness report changed")
    readiness = json.loads(readiness_path.read_text())
    if (not readiness["source_integrity_passed"] or readiness["failures"]
            or readiness["group_overlap_count"] or not readiness["task_independent_grasp_candidates"]):
        raise ValueError("Failed or leaking source split")
    if any(task["missing_pick_names"] or task["missing_destination_names"]
           for task in readiness["splits"]["train"]["tasks"].values()):
        raise ValueError("TRAIN vocabulary is incomplete even for this fit pilot")
    # Full original-source physical replays cover every payload contributing to
    # this TRAIN subset, not merely an easy selected task.
    replay_by_source, proofs = {}, []
    for path in map(Path, replay_reports):
        replay = json.loads(path.read_text())
        parent = Path(replay["source"]).resolve()
        native = replay.get("summaries", {}).get("native", {})
        if (not replay.get("passed") or replay.get("failures") or not replay.get("all_episodes")
                or native.get("episodes", 0) != native.get("successes")
                or native.get("initial_false_positives") != 0
                or not replay.get("rows")
                or any(not row.get("ever_strict_success") or row.get("initial_strict_success")
                       for row in replay["rows"] if row["mode"] == "native")
                or replay["source_index_sha256"] != sha256_file(parent / "index.json")
                or replay["source_manifest_sha256"] != sha256_file(parent / "dataset_manifest.json")):
            raise ValueError(f"Incomplete/stale physical native replay: {path}")
        verify_source_environment_runtime(replay, parent)
        replay_by_source[str(parent)] = replay
        proofs.append({"path": str(path.resolve()), "sha256": sha256_file(path), "role": "full_parent_native_physics_replay"})
    for lineage in readiness["lineage"]:
        parent = Path(lineage["source"])
        if str(parent) not in replay_by_source:
            raise ValueError(f"Missing native physical replay for parent: {parent}")
        parent_audit_path = Path(lineage["source_audit"])
        if sha256_file(parent_audit_path) != lineage["source_audit_sha256"]:
            raise ValueError("Parent source audit changed")
        verify_source_payload_for_profile(json.loads(parent_audit_path.read_text()), parent)
        proofs.append({"path": str(parent_audit_path.resolve()), "sha256": sha256_file(parent_audit_path), "role": "parent_camera_and_payload_audit"})
    proofs.extend([{"path": str(Path(source_audit).resolve()), "sha256": sha256_file(source_audit), "role": "pilot_train_camera_and_payload_audit"},
                   {"path": str(readiness_path.resolve()), "sha256": sha256_file(readiness_path), "role": "frozen_scene_disjoint_readiness"}])
    output.mkdir(parents=True)
    for item in index:
        os.link(source / item["file"], output / item["file"])
    (output / "index.json").write_bytes((source / "index.json").read_bytes())
    if (source / "audit.json").is_file():
        (output / "audit.json").write_bytes((source / "audit.json").read_bytes())
    qualified = {**manifest, "candidate_only": False, "training_certified": False,
                 "qualification_parent": str(source),
                 "qualification_parent_index_sha256": sha256_file(source / "index.json"),
                 "qualification_parent_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
                 "purpose": "Bounded pipeline/fit pilot, NOT sufficient final-benchmark training data",
                 "training_qualification": {"scope": "bounded_pipeline_pilot", "allowed_models": [model],
                     "max_optimizer_updates": max_updates, "proofs": proofs,
                     "data_sufficiency_established": False, "final_test_used": False,
                     "known_limits": ["Few independent examples per object/destination pair",
                         "Recovery remains sparse; closed-empty restart is not universal missed-grasp recovery",
                         "Destination can briefly be occluded during pickup; replan1 and short future horizon required",
                         "No stable learned-policy success is established"]}}
    (output / "dataset_manifest.json").write_text(json.dumps(qualified, indent=2) + "\n")
    return {"output": str(output), "episodes": len(index), "frames": manifest["frames"],
            "scope": "bounded_pipeline_pilot", "max_optimizer_updates": max_updates, "training_certified": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--native-replay-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(qualify(args.source, args.source_audit, args.native_replay_report, args.output, max_updates=args.max_updates), indent=2))


if __name__ == "__main__":
    main()
