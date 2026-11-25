"""Create an immutable manifest for an audited V9 closed-empty recovery pilot.

This finalizes candidate metadata only. It never promotes training certification,
rewrites NPZ/index/audit files or claims broad natural-failure recovery.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from VLA.common.observation_profile import profile_hash, validate_profile
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.source_payload_provenance import (
    source_payload_fingerprint,
    verify_source_payload_fingerprint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_COUNTS = {name: 2 for name in ("touch", "pick", "move", "pick_place")}
JOINT_ACTION_KIND = "requested_joint_position_command"
RECOVERY_KINDS = ("closed_empty_workspace", "closed_empty_near_object_missed_close")


def _validate_near_object_evidence(row):
    evidence = row.get("near_object_recovery_start_evidence", {})
    replay = row.get("near_object_recovery_replay_evidence", {})
    anchor = evidence.get("anchor_name")
    if (anchor not in row["task"].get("config_pick_items", ())
            or evidence.get("protocol") != "near_body_contact_free_attempted_close_v1"
            or evidence.get("startup_controls_are_training_labels") is not False
            or evidence.get("source_object_poses_modified") is not False
            or evidence.get("safety_guards_modified") is not False):
        raise ValueError(f"Missing independently bounded near-body miss evidence: {row['file']}")
    checks = row.get("near_object_recovery_checks")
    required_checks = {"initial_actual_closure_at_least_95pct", "first_expert_label_reopens",
                       "no_false_grasp_on_first_action", "verified_carry_at_task_end", "no_safety_abort"}
    if not isinstance(checks, dict) or set(checks) != required_checks or any(value is not True for value in checks.values()):
        raise ValueError(f"Near-object recovery checks failed or incomplete: {row['file']}")
    for recorded in (evidence, replay):
        actual = recorded.get("actual_state", {})
        numeric_fields = ("actual_closure", "actual_main_angle_rad", "surface_clearance_m",
                          "closest_object_clearance_m", "anchor_aabb_distance_m")
        if any(not isinstance(actual.get(name), (float, int)) or not np.isfinite(actual[name])
               for name in numeric_fields):
            raise ValueError(f"Near-object t0 has invalid physical feedback: {row['file']}")
        if (actual.get("grasped_object", "missing") is not None
                or actual.get("gripper_contact_count") != 0 or actual.get("bilateral_contact_count") != 0
                or actual.get("actual_closure", -1) < .95
                or actual.get("actual_main_angle_rad", -1) < .95*.78
                or actual.get("surface_clearance_m", -1) < .003
                or actual.get("closest_object_clearance_m", -1) < .003
                or not .015 <= actual.get("anchor_aabb_distance_m", -1) <= .11):
            raise ValueError(f"Near-object t0 is not physically safe/closed/empty: {row['file']}")
        tcp, aabb = np.asarray(actual.get("tcp_world")), np.asarray(actual.get("anchor_aabb"))
        if (tcp.shape != (3,) or aabb.shape != (2, 3)
                or not np.isfinite(tcp).all() or not np.isfinite(aabb).all()
                or tcp[2] > aabb[1, 2]+.020):
            raise ValueError(f"Near-object t0 is not at the physical object body height: {row['file']}")
        visibility = recorded.get("visibility", {})
        front = visibility.get("front", {})
        wrist_anchor = visibility.get("wrist", {}).get(anchor, {})
        from VLA.common.spatial import FIXED_DESTINATION
        expected = set((*row["task"].get("config_pick_items", ()),
                        *row["task"].get("config_place_items", ()))) - set(FIXED_DESTINATION)
        if set(front) != expected:
            raise ValueError(f"Near-object camera evidence omits a configured object: {row['file']}")
        for cell in (*front.values(), wrist_anchor):
            if (cell.get("passes") is not True or cell.get("pixels_224", -1) < 48
                    or min(cell.get("width_224", -1), cell.get("height_224", -1)) < 7
                    or cell.get("contrast", -1) < 25):
                raise ValueError(f"Near-object model-resolution camera gate failed: {row['file']}")
    for key, maximum in (("source_object_translation_drift_m", .001),
                         ("source_object_rotation_drift_rad", .01)):
        drift = evidence.get(key)
        if (not isinstance(drift, dict) or not drift
                or any(not np.isfinite(value) or value < 0 or value > maximum for value in drift.values())):
            raise ValueError(f"Near-object startup source-pose drift is invalid: {row['file']}")
    return anchor


def finalize(candidate: Path, *, recovery_kind="closed_empty_workspace", expected_per_task=2,
             expected_transitions=None) -> dict:
    if recovery_kind not in RECOVERY_KINDS or type(expected_per_task) is not int or expected_per_task < 1:
        raise ValueError("Declare a supported recovery kind and positive expected task count")
    task_counts = {name: expected_per_task for name in TASK_COUNTS}
    expected_episodes = sum(task_counts.values())
    # The original immutable workspace8/301 pilot remains backward compatible.
    if expected_transitions is None and recovery_kind == "closed_empty_workspace" and expected_per_task == 2:
        expected_transitions = 301
    candidate = candidate.resolve()
    output = candidate / "dataset_manifest.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate manifest: {output}")
    report_path = candidate / "audit.json"
    index_path = candidate / "index.json"
    state_path = candidate / "collection_state.json"
    report = json.loads(report_path.read_text())
    index = json.loads(index_path.read_text())
    state = json.loads(state_path.read_text())
    if report.get("passed") is not True or report.get("episodes") != expected_episodes:
        raise ValueError("Recovery collection report must pass with the declared episode count")
    if (state.get("status") != "completed_candidate_audit_passed"
            or state.get("completed_episodes") != expected_episodes or state.get("planned_episodes") != expected_episodes):
        raise ValueError("Recovery collection state is incomplete or failed")
    if report.get("recovery_start_kind") != recovery_kind:
        raise ValueError("Recovery kind differs from the explicit finalizer contract")
    if len(index) != expected_episodes or dict(Counter(row["task"]["name"] for row in index)) != task_counts:
        raise ValueError("Recovery pilot counts differ from the declared four-primitive contract")
    near_object = recovery_kind == "closed_empty_near_object_missed_close"
    if near_object and report.get("near_object_recovery_checks_passed") is not True:
        raise ValueError("Near-object recovery collection checks did not pass")
    profile = validate_profile(report["observation_profile"])
    identity = profile_hash(profile)
    if (profile.get("gripper_controller") != "body_contact_v9"
            or profile.get("gripper_state_contract") != "encoder_fk_aperture_closure_v1"
            or profile.get("native_gripper_action_kind") != "requested_close_intent"):
        raise ValueError("Recovery pilot requires actual V9 feedback and separate close-intent labels")
    if report.get("observation_profile_sha256") != identity:
        raise ValueError("Recovery collection profile hash is inconsistent")
    if report.get("native_joint_action_kind") != JOINT_ACTION_KIND:
        raise ValueError("Recovery native actions must be requested joint commands")

    parent = Path(report["source"]).resolve()
    parent_index = parent / "index.json"
    parent_manifest = parent / "dataset_manifest.json"
    parent_index_hash = sha256_file(parent_index)
    if report.get("source_index_sha256") != parent_index_hash:
        raise ValueError("Paired replay parent index changed after collection")
    manifest = copy.deepcopy(json.loads(parent_manifest.read_text()))
    if manifest.get("robot") != "ur5e" or manifest.get("scene_profile") != "rl_random_env":
        raise ValueError("Recovery pilot must preserve the parent UR5e/random-env scene")
    protected = {path: sha256_file(path) for path in
                 (report_path, index_path, state_path, parent_index, parent_manifest)}
    payloads = source_payload_fingerprint(candidate, index)
    frames = 0
    closure_at_start = []
    max_joint_sync_error = 0.
    max_gripper_sync_error = 0.
    near_object_anchors = set()
    for row in index:
        if near_object:
            near_object_anchors.add(_validate_near_object_evidence(row))
        if (row.get("success") is not True or row.get("safety_violation") is not None
                or row.get("safety_rollback_restored") is not None
                or row.get("observation_profile_sha256") != identity
                or row.get("native_joint_action_kind") != JOINT_ACTION_KIND
                or row.get("gripper_state_contract") != "encoder_fk_aperture_closure_v1"
                or row.get("native_gripper_action_kind") != "requested_close_intent"):
            raise ValueError(f"Failed/unsafe/mixed-contract episode: {row['file']}")
        if (row.get("paired_reset_strategy") != "canonical_spawn_order_then_teacher_retarget"
                or row.get("success_contract", {}).get("required_consecutive_steps") != 1):
            raise ValueError(f"Episode reset/success contract differs: {row['file']}")
        startup = np.asarray(row.get("recovery_start_actions_not_training_labels"), dtype=float)
        if (startup.ndim != 2 or startup.shape[1] != 7 or not np.isfinite(startup).all()
                or (startup[:, -1] > .2).sum() < 5):
            raise ValueError(f"Missing physically settled startup controls: {row['file']}")
        with np.load(candidate / row["file"], allow_pickle=False) as episode:
            actions = episode["canonical_actions"]
            closure = episode["gripper_closure"]
            next_closure = episode["next_gripper_closure"]
            native_intent = episode["next_requested_gripper_closure"]
            verified = episode["next_verified_grasp"]
            bilateral = episode["next_bilateral_pad_contact"] if near_object else None
            joints = episode["joint_positions"]
            next_joints = episode["next_observed_joint_positions"]
            length = len(actions)
            if (not length or length != row["steps"] or actions.shape != (length, 7)
                    or any(len(array) != length for array in
                           (closure, next_closure, native_intent, verified, joints, next_joints))):
                raise ValueError(f"Malformed corrective transition lengths: {row['file']}")
            if (not all(np.isfinite(array).all() for array in
                        (actions, closure, next_closure, native_intent, joints, next_joints))
                    or np.any(closure < 0) or np.any(closure > 1)
                    or np.any(next_closure < 0) or np.any(next_closure > 1)):
                raise ValueError(f"Invalid actual feedback values: {row['file']}")
            if (float(closure[0]) < .95 or float(actions[0, -1]) != -1.
                    or float(native_intent[0]) != 0. or bool(verified[0])):
                raise ValueError(f"Not measured-closed / first corrective OPEN-unheld: {row['file']}")
            if near_object and abs(float(closure[0]) - row["near_object_recovery_replay_evidence"]["actual_state"]["actual_closure"]) > 1e-6:
                raise ValueError(f"Near-object replay feedback differs from recorded t0 observation: {row['file']}")
            if near_object and row["task"]["name"] != "touch" and (
                    len(bilateral) != length or not bool(verified[-1]) or not bool(bilateral[-1])):
                raise ValueError(f"Near-object recovery carry ends without verified bilateral grasp: {row['file']}")
            if not np.allclose(native_intent, (actions[:, -1] + 1) / 2, atol=1e-7, rtol=0):
                raise ValueError(f"Requested intent is not the applied command label: {row['file']}")
            if length > 1:
                joint_error = float(np.max(np.abs(next_joints[:-1] - joints[1:])))
                gripper_error = float(np.max(np.abs(next_closure[:-1] - closure[1:])))
                if joint_error > 1e-5 or gripper_error > 1e-6:
                    raise ValueError(f"Actual feedback is not transition-synchronized: {row['file']}")
                max_joint_sync_error = max(max_joint_sync_error, joint_error)
                max_gripper_sync_error = max(max_gripper_sync_error, gripper_error)
            closure_at_start.append(float(closure[0]))
            frames += length
    if expected_transitions is not None and frames != expected_transitions:
        raise ValueError(f"Expected {expected_transitions} corrective transitions, got {frames}")
    verify_source_payload_fingerprint(payloads, candidate, index)
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise RuntimeError("Protected collection/parent metadata changed during finalization")

    # Clear inherited labels that describe a different collection/source.
    manifest.pop("repaired_from", None)
    manifest.update(
        required_success_steps=1, seed=20261007, episodes_per_task=expected_per_task,
        stratified_target_vocabulary=False, declared_task_counts=task_counts,
        task_counts=task_counts, episodes=expected_episodes, frames=frames, transitions=frames,
        independent_physical_layouts=len(report.get("pairs", [])), layouts_per_task=report.get("layouts_per_task", 1),
        observation_profile=profile, observation_profile_sha256=identity,
        cameras=profile["cameras"], home_joints=profile["home_joints"],
        front_camera_role=profile.get("front_camera_role", "robot_overview"),
        grasp_candidate_scope=profile.get("grasp_candidate_scope"),
        overview_supplied_to_policy=False, privileged_pose_in_policy_input=False,
        policy_task_input="natural-language prompt only",
        gripper_state_contract="encoder_fk_aperture_closure_v1",
        native_gripper_action_kind="requested_close_intent",
        native_joint_action_kind=JOINT_ACTION_KIND,
        candidate_only=True, training_certified=False,
        purpose=f"{expected_episodes} paired {recovery_kind} corrective episodes; NOT sufficient final training data",
        recovery_start_kind=recovery_kind,
        recovery_start_steps="per_episode", recovery_bad_start_actions_in_training_labels=False,
        recovery_scope=("Deliberate closed-empty near-object missed close; NOT contact-slip/all natural grasp failures"
                        if near_object else "Physically closed empty off-path workspace reset and corrective reopening ONLY"),
        near_object_missed_grasp_recovery_certified=False,
        dropped_or_wrong_object_recovery_certified=False,
        learned_vla_performance_certified=False,
        parent_source=str(parent), original_source_index_sha256=parent_index_hash,
        parent_source_index_sha256=parent_index_hash,
        parent_manifest_sha256=protected[parent_manifest],
        index_sha256=protected[index_path], paired_audit_sha256=protected[report_path],
        collection_state_sha256=protected[state_path], source_payload_fingerprint=payloads,
        environment_runtime_fingerprint=environment_runtime_fingerprint(profile),
        teacher_source_sha256=sha256_file(PROJECT_ROOT / "VLA/collision_planned_teacher.py"),
        planner_source_sha256=sha256_file(PROJECT_ROOT / "VLA/collision_grasp_planner.py"),
        collection_helper_source_sha256=sha256_file(PROJECT_ROOT / "VLA/audit_counterfactual_collection.py"),
        measured_t0_closure_range=[min(closure_at_start), max(closure_at_start)],
        first_corrective_transition_verified_grasp=False,
        actual_feedback_max_joint_sync_error_rad=max_joint_sync_error,
        actual_feedback_max_gripper_sync_error=max_gripper_sync_error,
        production_dataset_changed=False, final_test_used=False,
    )
    if near_object:
        manifest.update(
            near_object_deliberate_missed_close_pilot_audited=True,
            near_object_startup_unique_anchor_names=sorted(near_object_anchors),
            near_object_startup_unique_anchor_count=len(near_object_anchors),
            near_object_startup_source_sha256=sha256_file(PROJECT_ROOT / "VLA/near_object_recovery_start.py"),
        )
    # Exclusive creation: existing manifests are never silently overwritten.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    summary = {"manifest": str(output), "episodes": len(index), "frames": frames,
               "profile_sha256": identity, "candidate_only": True, "training_certified": False,
               "measured_t0_closure_range": manifest["measured_t0_closure_range"],
               "actual_feedback_max_joint_sync_error_rad": max_joint_sync_error,
               "actual_feedback_max_gripper_sync_error": max_gripper_sync_error}
    print(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--recovery-kind", choices=RECOVERY_KINDS, default="closed_empty_workspace")
    parser.add_argument("--expected-per-task", type=int, default=2)
    parser.add_argument("--expected-transitions", type=int)
    args = parser.parse_args()
    finalize(args.candidate, recovery_kind=args.recovery_kind, expected_per_task=args.expected_per_task,
             expected_transitions=args.expected_transitions)


if __name__ == "__main__":
    main()
