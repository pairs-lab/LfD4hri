"""Recover feedback by replaying immutable RECORDED neural actions, CPU only.

Never imports/calls a policy, uses teacher controls, repairs actions, or creates
additional evaluation trials. Diagnostic target/goal identities are kept only
in output records. Replay must match the original terminal outcomes exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pybullet as p

from config import BOUNDS
from VLA.common.action import CanonicalAction
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.physics_safety import physics_diagnostics, surface_clearance
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import safety_abort_info, verify_source_audit


def action_binding(row):
    payload = {"file": row["file"], "trace": row["trace"]}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def sensor(env):
    held = env.gripper.grasped_object
    names = {body: name for name, body in env.object_ids.items()}
    mount = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True)
    tcp = np.asarray(env.gripper.grasp_center(), dtype=float)
    target = np.asarray(p.getBasePositionAndOrientation(env.pick_id)[0], dtype=float)
    measured = float(env.normalized_gripper_closure())
    diagnostic = physics_diagnostics(env)
    return {
        "mount_position_m": list(mount[4]), "mount_quaternion": list(mount[5]),
        "tcp_position_m": tcp.tolist(), "target_position_m": target.tolist(),
        "measured_encoder_fk_closure": measured,
        "requested_binary_command_before_next_action": float(env.gripper.command),
        "jaw_gap_m": float(env.gripper.width()), "held_body_id": held,
        "held_body_name": names.get(held), "held_correct_target": held == env.pick_id,
        "closed_empty_proxy": bool(measured >= .95 and held is None),
        "ee_target_distance_m": float(env.ee_object_distance()),
        "tcp_target_xy_distance_m": float(np.linalg.norm(tcp[:2] - target[:2])),
        "object_goal_distance_m": float(env.object_goal_distance()),
        "target_bilateral_pad_contact": bool(env.gripper._bilateral(env.pick_id)),
        "surface_clearance_m": float(surface_clearance(env)),
        "joint_positions_rad": [float(p.getJointState(env.robot_id, joint)[0]) for joint in env.robot_spec.arm_joint_ids],
        "gripper_contacts": diagnostic["contacts"],
        "gripper_follower_errors_rad": diagnostic["follower_errors_rad"],
    }


def teacher_summary(source, item):
    with np.load(source / item["file"], allow_pickle=False) as episode:
        actions = np.asarray(episode["canonical_actions"], dtype=float)
        measured = np.asarray(episode["gripper_closure"], dtype=float)
        requests = native_gripper_closure(episode)
        verified = np.r_[False, episode["next_verified_grasp"][:-1]].astype(bool)
        close = np.flatnonzero(actions[:, 6] > .2)
        hold = np.flatnonzero(verified)
        transport = np.flatnonzero(verified & (np.linalg.norm(actions[:, :2], axis=1) >= .005))
        return {
            "scope": "Stored teacher reference ONLY; no teacher action is replayed or fed to neural policy",
            "observations": len(actions), "first_action": actions[0].tolist(),
            "first_close_action_frame": int(close[0]) if len(close) else None,
            "first_verified_observation_frame": int(hold[0]) if len(hold) else None,
            "first_verified_horizontal_transport_frame": int(transport[0]) if len(transport) else None,
            "close_requested_frames": int((requests > .5).sum()),
            "measured_closure_min_max": [float(measured.min()), float(measured.max())],
            "source_success": bool(item["success"]),
        }


def replay_job(job):
    source, item, original = job
    trace = original["trace"]
    if len(trace) != original["steps"] or not trace:
        raise ValueError("Incomplete recorded action trace")
    feedback = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.0, use_semantic_success=True,
                           render_observations=False) as env:
        reset_source_episode(env, source, item)
        before = sensor(env)
        if env.is_semantic_success():
            raise ValueError("Replay begins with an initial false positive")
        for index, record in enumerate(trace):
            if record["step"] != index + 1:
                raise ValueError("Nonconsecutive action timestamps")
            action = np.asarray(record["action"], dtype=np.float32)
            _obs, _reward, done, info = env.step(action)
            aborted, reason = safety_abort_info(info)
            executed = np.asarray(info["executed_canonical_action"], dtype=np.float32)
            if not np.array_equal(executed, np.asarray(record["executed_action"], dtype=np.float32)):
                error = float(np.max(np.abs(executed - np.asarray(record["executed_action"], dtype=np.float32))))
                raise ValueError(f"{item['file']}: recorded execution/safety clamp differs at step {record['step']}; max_abs_error={error:.9g}")
            for key, actual in (("semantic_success", info["semantic_raw_success"]),
                                ("strict_success", info["strict_raw_success"]),
                                ("safety_abort", aborted),
                                ("safety_clamped_action", info["safety_clamped_action"])):
                if bool(record[key]) != bool(actual):
                    raise ValueError(f"Recorded {key} differs at step {record['step']}")
            if record.get("safety_reason") != reason:
                raise ValueError("Safety reason mismatch")
            after = sensor(env)
            normalized = CanonicalAction.from_array(executed).values
            current_mount = np.asarray(before["mount_position_m"], dtype=np.float32)
            desired_mount = np.clip(current_mount + normalized[:3],
                    np.array((BOUNDS[0, 0] - .12, BOUNDS[1, 0] - .05, .10)),
                    np.array((BOUNDS[0, 1] + .12, BOUNDS[1, 1] + .05, .72)))
            actual_delta = np.asarray(after["mount_position_m"]) - current_mount
            desired_delta = desired_mount - current_mount
            before_rotation = np.asarray(p.getMatrixFromQuaternion(before["mount_quaternion"])).reshape(3, 3)
            after_rotation = np.asarray(p.getMatrixFromQuaternion(after["mount_quaternion"])).reshape(3, 3)
            relative = before_rotation.T @ after_rotation
            # Quaternion conversion is diagnostic only; no control is produced.
            actual_rotation = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
            feedback.append({"step": record["step"], "observation_frame_before_action": index,
                "neural_action": action.tolist(), "executed_precanonical_clip_action": executed.tolist(),
                "effective_canonical_after_clip": normalized.tolist(),
                "desired_mount_translation_after_safety_actionclip_workspaceclip_m": desired_delta.tolist(),
                "actual_mount_translation_m": actual_delta.tolist(),
                "actual_mount_rotation_angle_rad": actual_rotation,
                "translation_tracking_error_m": float(np.linalg.norm(actual_delta - desired_delta)),
                "tcp_delta_m": (np.asarray(after["tcp_position_m"]) - before["tcp_position_m"]).tolist(),
                "before": before, "after": after,
                "safety_abort": aborted, "safety_abort_reason": reason,
                "safety_diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
            })
            before = after
            if done:
                if index != len(trace) - 1:
                    raise ValueError("Replay terminates earlier than original recorded trace")
                break
        if not done:
            raise ValueError("Original terminal state was not reproduced")
        if int(info["step"]) != original["steps"]:
            raise ValueError("Terminal step mismatch")
        terminal_error = abs(float(info["ee_object_distance"]) - original["final_ee_object_distance_m"])
        goal_actual, goal_expected = info.get("goal_xy_distance_m"), original.get("final_goal_xy_distance_m")
        goal_error = None if goal_actual is None and goal_expected is None else abs(float(goal_actual) - float(goal_expected))
        if terminal_error > 1e-6 or (goal_error is not None and goal_error > 1e-6):
            raise ValueError(f"Terminal metric mismatch: EE={terminal_error}, goal={goal_error}")
        if bool(info["success"] and not aborted) != original["success"] or aborted != original["safety_abort"]:
            raise ValueError("Terminal outcome mismatch")
        if original.get("safety_diagnostics_before_rollback") != info.get("safety_diagnostics_before_rollback"):
            raise ValueError("Pre-rollback safety evidence differs")
    close = [row for row in feedback if row["neural_action"][6] > .2]
    lifts = [row for row in feedback if row["neural_action"][2] >= .005]
    held = [row for row in feedback if row["after"]["held_body_id"] is not None]
    return {"file": item["file"], "task": item["task"]["name"], "prompt": item["task"]["prompt"],
            "action_trace_sha256": action_binding(original), "steps": len(feedback),
            "matched_original_terminal": True, "terminal_ee_distance_abs_error_m": terminal_error,
            "terminal_goal_xy_abs_error_m": goal_error,
            "teacher_reference": teacher_summary(source, item),
            "summary": {"first_neural_close_action_step": close[0]["step"] if close else None,
                "first_close_before_sensor": close[0]["before"] if close else None,
                "close_requested_steps": len(close), "lift_requested_steps": len(lifts),
                "lift_requested_without_held_object_steps": sum(row["before"]["held_body_id"] is None for row in lifts),
                "fully_closed_empty_poststeps": sum(row["after"]["closed_empty_proxy"] for row in feedback),
                "any_held_body_names": sorted({row["after"]["held_body_name"] for row in held}),
                "minimum_ee_target_distance_m": min(row["after"]["ee_target_distance_m"] for row in feedback),
                "minimum_tcp_target_xy_distance_m": min(row["after"]["tcp_target_xy_distance_m"] for row in feedback),
                "translation_tracking_error_p50_p90_max_m": np.quantile([row["translation_tracking_error_m"] for row in feedback], [.5, .9, 1]).tolist()},
            "feedback": feedback}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise ValueError("Use only 1 or 2 CPU workers")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    _audit, manifest = verify_source_audit(source, args.audit_report)
    source_sha = (sha256_file(source / "index.json"), sha256_file(source / "dataset_manifest.json"))
    index = json.loads((source / "index.json").read_text())
    by_file = {item["file"]: item for item in index}
    payload = source_payload_fingerprint(source, index)
    runtime = environment_runtime_fingerprint(manifest["observation_profile"])
    progress_bytes = args.progress.read_bytes()
    recorded = json.loads(progress_bytes)
    if not 1 <= len(recorded) <= 8 or len({row["file"] for row in recorded}) != len(recorded):
        raise ValueError("Bound to at most 8 completed original episodes")
    jobs = [(source, by_file[row["file"]], row) for row in recorded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        replayed = list(executor.map(replay_job, jobs))
    verify_source_audit(source, args.audit_report)
    verify_environment_runtime_fingerprint(runtime, manifest["observation_profile"])
    if payload != source_payload_fingerprint(source, index) or source_sha != (sha256_file(source / "index.json"), sha256_file(source / "dataset_manifest.json")):
        raise RuntimeError("Source payload/metadata changed")
    # Appended completed episodes are expected while the original neural run is
    # active. Existing recorded traces, however, must never be silently edited.
    current = {row["file"]: row for row in json.loads(args.progress.read_text())}
    for row in recorded:
        if row["file"] not in current or action_binding(row) != action_binding(current[row["file"]]):
            raise RuntimeError("Original recorded neural action trace changed during replay")
    report = {"passed": True,
        "scope": "POSTHOC CPU recorded-neural-action replay ONLY; not a new neural rollout, additional trial, learned SR, or teacher fallback",
        "progress": str(args.progress.resolve()),
        "progress_sha256_at_snapshot": hashlib.sha256(progress_bytes).hexdigest(),
        "completed_episodes_at_snapshot": len(recorded), "source": str(source),
        "source_index_sha256": source_sha[0], "source_manifest_sha256": source_sha[1],
        "source_payload_fingerprint": payload, "environment_runtime_fingerprint": runtime,
        "model_or_policy_called": False, "teacher_action_used_for_controls": False,
        "source_or_original_evaluation_modified": False,
        "replay_pybullet_binary": {"path": str(Path(p.__file__).resolve()), "sha256": sha256_file(Path(p.__file__))},
        "diagnostic_privileged_fields": "Target, goal, masks/contact, distances and held IDs are output diagnostics only; no policy/model/controller target override",
        "metric_tolerance_m": 1e-6, "rows": replayed}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "episodes": len(replayed), "rows": [{k: v for k, v in row.items() if k != "feedback"} for row in replayed]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
