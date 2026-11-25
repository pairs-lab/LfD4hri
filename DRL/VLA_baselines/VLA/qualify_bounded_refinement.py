"""Qualify a finite larger-data experiment, NOT final data sufficiency.

Inherits ALL native-physics/payload/split gates of the initial pilot, then adds
predeclared independent-layout coverage and actual native-image review. Old
qualified sources are never edited or allowed to exceed their original cap.
"""
import argparse
import json
from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.prepare_v9_candidate_fit import require_visual_receipt
from VLA.qualify_bounded_training_pilot import qualify


def qualify_refinement(source, audit, replay, output, receipt, native, combined, combined_audit, max_updates):
    if type(max_updates) is not int or not 1001 <= max_updates <= 20000:
        raise ValueError("Refinement is a finite1001..20000-update experiment")
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    readiness = json.loads(Path(manifest["readiness_report"]).read_text())
    if readiness.get("minimum_train_repetitions", 1) < 3 or not readiness.get("preserve_training_vocabulary"):
        raise ValueError("Refinement requires predeclared independent-group target repetitions, not augmented copy counts")
    tasks = readiness["splits"]["train"]["tasks"]
    if set(tasks) != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("Missing four-task coverage")
    for task in tasks.values():
        if (task["physical_layouts"] < 100 or task["missing_pick_names"] or task["missing_destination_names"]
                or min(task["pick_counts"].values(), default=0) < 3
                or (task["destination_counts"] and min(task["destination_counts"].values()) < 3)):
            raise ValueError("Predeclared larger-data experiment coverage not met; no promotion")
    require_visual_receipt(receipt, combined, native, combined_audit)
    from VLA.audit_transport_visibility import audit as audit_transport
    phase_report = Path(combined_audit).parent/"per_episode_transport_visibility.json"
    phase = audit_transport(combined, phase_report, max_transient_frames=1)
    if not phase["passed"] or not phase["all_episodes"]:
        raise ValueError("Complete per-trajectory carry visibility failed; no refinement qualification")
    precondition_path = output.with_name(output.name+"_refinement_preconditions.json")
    if output.exists() or precondition_path.exists():
        raise FileExistsError("Use a fresh refinement qualification")
    # This call verifies the exact full native replay, every lineage/payload
    # audit, immutable readiness and unchanged environment before creating data.
    result = qualify(source, audit, [replay], output, max_updates=1000)
    preconditions = {"passed": True, "scope": "bounded_refinement_experiment",
        "source_index_sha256": sha256_file(source/"index.json"), "train_coverage": tasks,
        "minimum_independent_layouts_per_task": 100, "minimum_train_repetitions_per_target": 3,
        "max_optimizer_updates": max_updates, "visual_review_sha256": sha256_file(receipt),
        "data_sufficiency_established": False, "final_test_used": False,
        "note": "Engineering gates for a bounded learning experiment, not universal dataset-sufficiency thresholds"}
    precondition_path.write_text(json.dumps(preconditions, indent=2)+"\n")
    qualified = json.loads((output/"dataset_manifest.json").read_text())
    qualification = qualified["training_qualification"]
    qualification.update(scope="bounded_refinement_experiment", max_optimizer_updates=max_updates)
    qualification["known_limits"].append("Explicit single100ms goal-observation outages can be accepted for bounded fit; not all manipulation frames are fully observable. Persistent outages still stop qualification.")
    qualification["proofs"].extend([
        {"path": str(precondition_path), "sha256": sha256_file(precondition_path), "role": "bounded_refinement_preconditions"},
        {"path": str(Path(receipt).resolve()), "sha256": sha256_file(receipt), "role": "actual_four_task_native_image_review"},
        {"path": str(phase_report.resolve()), "sha256": sha256_file(phase_report), "role": "all_episode_transport_phase_visibility"},
        {"path": str(Path(native).resolve()/"audit.json"), "sha256": sha256_file(Path(native)/"audit.json"), "role": "actual_smol_native_image_geometry"}])
    qualified["purpose"] = "Finite larger-data refinement, NOT final-benchmark sufficiency or model-success guarantee"
    (output/"dataset_manifest.json").write_text(json.dumps(qualified, indent=2)+"\n")
    return {**result, "scope": "bounded_refinement_experiment", "max_optimizer_updates": max_updates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "native-replay-report", "output", "visual-review-receipt",
                 "native-camera-dir", "combined-source", "combined-source-audit"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--max-updates", type=int, default=5000)
    a = parser.parse_args()
    print(json.dumps(qualify_refinement(a.source, a.source_audit, a.native_replay_report, a.output,
        a.visual_review_receipt, a.native_camera_dir, a.combined_source, a.combined_source_audit, a.max_updates), indent=2))


if __name__ == "__main__":
    main()
