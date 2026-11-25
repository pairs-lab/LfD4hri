"""Attach immutable source metadata to a completed camera/data diagnostic.

This permits the normal numeric/camera/conversion audits to inspect a candidate.
It does NOT promote it to production or certify data/training sufficiency.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from VLA.common.observation_profile import validate_profile, profile_hash
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--camera-report", required=True, type=Path)
    args = parser.parse_args()
    root = args.candidate.resolve()
    output = root / "dataset_manifest.json"
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((root / "collection_state.json").read_text())
    paired = json.loads((root / "audit.json").read_text())
    index = json.loads((root / "index.json").read_text())
    camera = json.loads(args.camera_report.read_text())
    if state["status"] != "completed_candidate_audit_passed" or not paired["passed"]:
        raise ValueError("Paired collection must finish and pass before manifest creation")
    if len(index) != state["planned_episodes"] or not all(row["success"] for row in index):
        raise ValueError("Incomplete/failed candidate")
    profile = validate_profile(paired["observation_profile"])
    identity = profile_hash(profile)
    if any(row.get("observation_profile_sha256") != identity for row in index):
        raise ValueError("Mixed sensor/home domains in candidate")
    summary = next((s for s in camera["summary"] if s["profile_sha256"] == identity), None)
    if (summary is None or summary["teacher_success"] != summary["teacher_trials"]
            or summary["min_reset_table_corners"] != 4 or summary["all_frames_both_views_absent"] != 0):
        raise ValueError("Camera/teacher qualification does not match candidate profile")
    manifest = {"robot": "ur5e", "robot_urdf": "asset/ur5e/ur5e.urdf", "gripper": "Robotiq 2F-85",
        "gripper_asset": "asset/robotiq_2f_85", "image_size": 384, "physics_hz": 240, "control_hz": 10,
        "required_success_steps": 1, "scene_profile": "rl_random_env", "scene_source": "utils/random_env.py",
        "stratified_target_vocabulary": False,
        "candidate_only": True, "purpose": "native pipeline/conditioning canary, NOT sufficient final benchmark training",
        "declared_task_counts": dict(Counter(row["task"]["name"] for row in index)),
        "original_source_index_sha256": paired["source_index_sha256"],
        "index_sha256": sha256_file(root/"index.json"), "paired_audit_sha256": sha256_file(root/"audit.json"),
        "camera_report_sha256": sha256_file(args.camera_report), "observation_profile": profile,
        "front_camera_role": profile.get("front_camera_role", "robot_overview"),
        "overview_supplied_to_policy": False,
        "observation_profile_sha256": identity, "cameras": profile["cameras"], "home_joints": profile["home_joints"],
        "recovery_start_steps": paired["recovery_start_steps"],
        "recovery_bad_start_actions_in_training_labels": False,
        "named_regions_included": paired["includes_all_named_regions"],
        "all_destination_names_included": paired.get("includes_all_destinations", False),
        "grasp_candidate_scope": profile.get("grasp_candidate_scope", "selected_target_legacy"),
        "privileged_pose_in_policy_input": False, "policy_task_input": "natural-language prompt only",
        "action_contract": "[dx,dy,dz,droll,dpitch,dyaw,gripper_-1_open_+1_close]",
        "gripper_state_contract": "commanded closure, NOT calibrated measured physical aperture",
        "native_joint_action_kind": paired.get("native_joint_action_kind", "measured_next_joint_position"),
        "production_dataset_changed": False, "final_test_used": False}
    output.write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps({"manifest": str(output), "episodes": len(index), "profile_sha256": identity}))


if __name__ == "__main__":
    main()
