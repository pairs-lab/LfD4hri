"""Reproduce the preserved near-body recovery hole with actual images/baselines.

Original rejection gates remain unchanged. A no-motion baseline repeats the
same number of physics/control ticks and gripper commands to distinguish
gravity settling from robot-induced object displacement. Diagnostic only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.near_object_recovery_start import closed_empty_near_object_start


def object_poses(env):
    return {name: [list(v) for v in p.getBasePositionAndOrientation(body)] for name, body in env.object_ids.items()}


def rotation_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    cosine = abs(float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))))
    return float(2*np.arccos(np.clip(cosine, 0, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate = args.failed_candidate.resolve()
    plan = json.loads((candidate/"collection_plan.json").read_text())
    state = json.loads((candidate/"collection_state.json").read_text())
    if (state["status"] != "stopped_error_NOT_training" or state["completed_episodes"]%2
            or "No safe visible near-object missed-grasp startup" not in state["error"]
            or plan.get("final_test_used") is not False):
        raise ValueError("Preserved completed-pair boundary startup exception required")
    source = Path(plan["source"])
    if (sha256_file(source/"index.json") != plan["source_index_sha256"]
            or sha256_file(source/"dataset_manifest.json") != plan["source_manifest_sha256"]):
        raise ValueError("Original canonical TRAIN changed")
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    if manifest.get("split") != "train" or manifest.get("final_test_used"):
        raise ValueError("Never diagnose with validation/final-test labels")
    filename = plan["selected_source_files"][state["completed_episodes"]//2]
    row = next(r for r in json.loads((source/"index.json").read_text()) if r["file"] == filename)
    task = TaskSpec.from_dict(row["task"])
    protected = [candidate/"collection_plan.json", candidate/"collection_state.json", candidate/"index.json",
        source/"index.json", source/"dataset_manifest.json", source/filename,
        Path(__file__), Path(__file__).with_name("near_object_recovery_start.py")]
    hashes = {str(path.resolve()): sha256_file(path) for path in protected}
    verify_environment_runtime_fingerprint(manifest["environment_runtime_fingerprint"], manifest["observation_profile"])
    args.output.mkdir(parents=True)
    events = []
    def callback(env, proposal, actions, original, visibility):
        folder = args.output/f"physical_start{len(events):02d}"
        folder.mkdir()
        paths = {}
        for camera in ("front", "wrist"):
            path = folder/f"{camera}_raw.png"
            Image.fromarray(env.render(camera)).save(path)
            paths[camera] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        events.append({"proposal": proposal, "controls_NOT_labels": [a.tolist() for a in actions],
            "initial_object_poses": {k: [list(v) for v in pose] for k, pose in original.items()},
            "actual_object_poses": object_poses(env), "visibility_original_224_gate": visibility,
            "image_files": paths, "actual_closure": env.measured_gripper_closure(),
            "gripper_contact_count_at_start": len(p.getContactPoints(bodyA=env.gripper.body)),
            "held_object": env.gripper.grasped_object})
        (args.output/"physical_start_progress.json").write_text(json.dumps(events, indent=2)+"\n")
    exception, startup_passed = None, False
    with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
        reset_source_episode(env, source, row)
        try:
            closed_empty_near_object_start(env, task, audit_callback=callback)
            startup_passed = True
        except RuntimeError as exc:
            exception = str(exc)
    # Every physically closed-empty start observed before any original image/
    # drift rejection gets a matched-time baseline, not a hand-picked success.
    for event in events:
        with audit_environment(source, image_size=32, required_success_steps=1,
                               use_semantic_success=False, render_observations=False) as env:
            env.reset(task)
            baseline_initial = object_poses(env)
            for action in event["controls_NOT_labels"]:
                idle = np.array([0, 0, 0, 0, 0, 0, action[6]], dtype=np.float32)
                _, _, done, info = env.step(idle)
                if done or info.get("safety_violation") or env.gripper.grasped_object is not None:
                    raise RuntimeError("No-motion baseline terminated/touched/held; do not compare")
            baseline = object_poses(env)
        comparisons = {}
        for name, actual in event["actual_object_poses"].items():
            initial = event["initial_object_poses"][name]
            comparisons[name] = {
                "actual_translation_from_reset_m": float(np.linalg.norm(np.asarray(actual[0])-initial[0])),
                "actual_rotation_from_reset_rad": rotation_distance(actual[1], initial[1]),
                "baseline_translation_from_reset_m": float(np.linalg.norm(np.asarray(baseline[name][0])-baseline_initial[name][0])),
                "actual_minus_no_motion_translation_m": float(np.linalg.norm(np.asarray(actual[0])-baseline[name][0])),
                "actual_minus_no_motion_rotation_rad": rotation_distance(actual[1], baseline[name][1])}
        event["matched_physics_time_baseline"] = {"control_steps": len(event["controls_NOT_labels"]),
            "gripper_commands_matched": True, "object_comparisons": comparisons,
            "max_robot_specific_translation_m": max(v["actual_minus_no_motion_translation_m"] for v in comparisons.values()),
            "max_robot_specific_rotation_rad": max(v["actual_minus_no_motion_rotation_rad"] for v in comparisons.values())}
    report = {"scope": "Exact excluded TRAIN near-miss scene image/settling diagnostic_NOT_fit_data_or_neural_success",
        "failed_candidate": str(candidate), "source": str(source), "source_episode": filename,
        "task": row["task"], "protected_hashes": hashes, "startup_passed_original_gates": startup_passed,
        "exception": exception, "original_rejection_message_identical": exception == state["error"],
        "all_physically_valid_start_events_included": True, "events": events,
        "runtime_success_safety_or_camera_changed": False, "training_performed": False, "final_test_used": False}
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Preserved source/instrumentation changed during audit")
    print(json.dumps({"events": len(events), "original_rejection_identical": report["original_rejection_message_identical"],
        "baseline_max_robot_specific_translation_m": [e["matched_physics_time_baseline"]["max_robot_specific_translation_m"] for e in events]}))


if __name__ == "__main__":
    main()
