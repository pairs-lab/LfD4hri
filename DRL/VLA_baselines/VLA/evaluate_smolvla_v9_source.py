"""Bounded learned-policy source-scene validation, never a final benchmark.

Reset metadata reconstructs the scene only. Controls use the normal deployed
RGB/proprio/text adapter, with no teacher actions, coordinates or masks and no
oracle fallback. HELDOUT diagnostics always deploy TRAIN normalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.source_payload_provenance import verify_source_payload_for_profile

TASKS = ("touch", "pick", "move", "pick_place")


def safety_abort_info(info):
    """The V9 controller exposes safety_violation, not just safety_abort."""
    return bool(info.get("safety_violation") or info.get("safety_abort")), info.get("safety_violation", info.get("safety_abort_reason"))


def verify_source_audit(source, audit_path):
    source = Path(source).resolve()
    audit = json.loads(Path(audit_path).read_text())
    if (not audit.get("passed") or audit.get("failures")
            or Path(audit.get("dataset", "")).resolve() != source
            or audit.get("index_sha256") != sha256_file(source / "index.json")
            or audit.get("manifest_sha256") != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Source audit is failed, stale or for a different dataset")
    verify_source_payload_for_profile(audit, source)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    if ((manifest.get("observation_profile") or {}).get("gripper_controller") != "body_contact_v9"
            or manifest.get("required_success_steps") != 1
            or manifest.get("privileged_pose_in_policy_input") is not False):
        raise ValueError("Requires the V9 one-shot nonprivileged source contract")
    verify_environment_runtime_fingerprint(manifest.get("environment_runtime_fingerprint"),
                                           manifest["observation_profile"])
    return audit, manifest


def reset_observation_errors(observation, measured_closure, recorded):
    """Compare real deployment sensors to the recorded initial observation.

    Source scene metadata is used only to reconstruct and check the reset;
    neither these diagnostics nor recorded labels enter policy prediction.
    """
    joints = np.asarray(observation.joint_positions)
    reference = np.asarray(recorded["joint_positions"][0])
    if joints.shape != reference.shape or joints.shape != (6,):
        raise ValueError("Reset joint observation shape differs from source")
    errors = {"joint_max_abs_error": float(np.abs(joints - reference).max()),
              "measured_closure_abs_error": abs(float(measured_closure) - float(recorded["gripper_closure"][0]))}
    for camera in ("front", "wrist"):
        actual = np.asarray(getattr(observation, camera + "_rgb"))
        expected = np.asarray(recorded[camera + "_rgb"][0])
        if (actual.shape != expected.shape or actual.ndim != 3 or actual.shape[-1] != 3
                or actual.dtype != np.uint8 or expected.dtype != np.uint8):
            raise ValueError(f"Reset {camera} RGB shape/dtype differs from source")
        delta = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
        errors[camera + "_rgb_max_abs_error"] = int(delta.max())
        errors[camera + "_rgb_mean_abs_error"] = float(delta.mean())
        errors[camera + "_rgb_changed_pixel_fraction"] = float(np.mean(np.any(delta != 0, axis=-1)))
    return errors


def validate_reset_observation(errors):
    """Fail closed using the frozen derivative's numeric/RGB rounding gates."""
    limits = {"joint_max_abs_error": 1e-6, "measured_closure_abs_error": 1e-6}
    for camera in ("front", "wrist"):
        limits[camera + "_rgb_max_abs_error"] = 1
        limits[camera + "_rgb_changed_pixel_fraction"] = .001
    failures = [key for key, limit in limits.items()
                if key not in errors or not np.isfinite(errors[key]) or errors[key] < 0 or errors[key] > limit]
    if failures:
        raise ValueError("Deployment reset observation differs from audited source: " + ", ".join(failures))
    return {"passed": True, "limits": limits, **errors}


def select_episodes(rows, per_task=2, filenames=None):
    if type(per_task) is not int or not 1 <= per_task <= 4:
        raise ValueError("A bounded diagnostic accepts 1..4 episodes per task")
    if filenames is not None:
        names = list(filenames)
        if len(names) != len(set(names)):
            raise ValueError("Duplicate selected episode filenames")
        selected = [row for row in rows if row["file"] in names]
        if len(selected) != len(names):
            raise ValueError("Selected source episode is missing")
        if any(sum(row["task"]["name"] == task for row in selected) != per_task for task in TASKS):
            raise ValueError("Explicit subset must contain exactly per_task episodes for all four tasks")
        return selected
    from VLA.audit_training_readiness import physical_scene_key
    selected = []
    for task in TASKS:
        candidates = [row for row in rows if row["task"]["name"] == task]
        if len(candidates) < per_task:
            raise ValueError(f"Insufficient source episodes for {task}")
        # Spread across frozen source order without selecting on learned
        # outcomes, target masks or easy geometry. Prefer distinct layouts.
        indices = np.linspace(0, len(candidates) - 1, len(candidates), dtype=int)
        order = list(np.linspace(0, len(candidates) - 1, per_task, dtype=int))
        order.extend(int(i) for i in indices if int(i) not in order)
        chosen, seen = [], set()
        for i in order:
            row = candidates[i]
            key = physical_scene_key(row["task"])
            if key in seen:
                continue
            chosen.append(row); seen.add(key)
            if len(chosen) == per_task:
                break
        if len(chosen) != per_task:
            raise ValueError(f"Not enough independent physical layouts for {task}")
        selected.extend(chosen)
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("heldout", "train_diagnostic"), required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--training-audit-report", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--episode-files", nargs="+")
    parser.add_argument("--seed", type=int, default=20261009)
    return parser.parse_args()


def main():
    from PIL import Image
    from VLA.common.observation_profile import profile_hash
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.recorder import VideoRecorder
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    from VLA.common.tasks import TaskSpec
    from VLA.dataset_replay_context import audit_environment, reset_source_episode
    from VLA.fingerprint_checkpoint import verify_fingerprint
    from VLA.smolvla.adapter import SmolVLAPolicyAdapter
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout, validate_saved_normalization

    args = parse_args()
    source, training = args.source.resolve(), args.training_source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit, manifest = verify_source_audit(source, args.audit_report)
    train_audit, train_manifest = verify_source_audit(training, args.training_audit_report)
    if train_manifest.get("split") != "train":
        raise ValueError("Normalization source must be the qualified TRAIN split")
    if args.split_mode == "heldout":
        if manifest.get("split") not in {"val", "validation"} or manifest.get("final_test_used"):
            raise ValueError("Only the frozen VALIDATION split, never final TEST, may be used")
        validate_holdout(training, source)
    elif source != training:
        raise ValueError("TRAIN diagnostic must use the actual training source")
    bound_profile = checkpoint_profile(args.checkpoint)
    if bound_profile is None or profile_hash(bound_profile) != manifest["observation_profile_sha256"]:
        raise ValueError("Checkpoint and source controller/camera/home profiles differ")
    identity = verify_fingerprint(args.checkpoint)
    normalization = validate_saved_normalization(args.checkpoint.resolve(), training, train_audit, identity)
    runtime = environment_runtime_fingerprint(bound_profile)
    rows = json.loads((source / "index.json").read_text())
    selected = select_episodes(rows, args.episodes_per_task, args.episode_files)
    args.output.mkdir(parents=True)
    # Freeze this exact subset before neural predictions, including startup
    # reset metadata. It cannot be selected based on outcome after the fact.
    (args.output / "selected_episodes.json").write_text(json.dumps(selected, indent=2) + "\n")
    results = []
    scope = "heldout_source_scene_diagnostic_not_final_benchmark" if args.split_mode == "heldout" else "training_scene_closed_loop_diagnostic_not_generalization"
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.0, use_semantic_success=True) as env:
        policy = SmolVLAPolicyAdapter(env, checkpoint=str(args.checkpoint.resolve()), replan_steps=1, seed=args.seed)
        recorder = VideoRecorder(args.output / "rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
        try:
            for row in selected:
                task = TaskSpec.from_dict(row["task"])
                observation = reset_source_episode(env, source, row)
                policy.reset(task)
                folder = args.output / Path(row["file"]).stem
                folder.mkdir()
                initial_success = bool(env.is_semantic_success())
                with np.load(source / row["file"], allow_pickle=False) as recorded:
                    reset_errors = reset_observation_errors(observation, env.normalized_gripper_closure(), recorded)
                (folder / "reset_observation.json").write_text(json.dumps(reset_errors, indent=2) + "\n")
                reset_errors = validate_reset_observation(reset_errors)
                if initial_success:
                    raise ValueError("Invalid learned evaluation: task already succeeds at reset")
                trace, info, correct, wrong = [], None, False, False
                for step in range(task.max_steps):
                    if step % 10 == 0:
                        for camera in ("front", "wrist"):
                            Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"step_{step:04d}_{camera}.png")
                    recorder.add(observation, f"{args.split_mode} | {task.prompt}")
                    # The only control-producing call: normal deployed VLA
                    # observation. Source labels never enter this call.
                    action = policy.predict(observation)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid learned action")
                    observation, _, done, info = env.step(action)
                    grasped = env.gripper.grasped_object
                    correct |= grasped == env.pick_id
                    wrong |= grasped is not None and grasped != env.pick_id
                    aborted, abort_reason = safety_abort_info(info)
                    trace.append({"step": int(info["step"]), "action": action.tolist(),
                                  "executed_action": info.get("executed_canonical_action"),
                                  "safety_clamped_action": bool(info.get("safety_clamped_action", False)),
                                  "semantic_success": bool(info["semantic_raw_success"]),
                                  "strict_success": bool(info["strict_raw_success"]),
                                  "safety_abort": aborted, "safety_reason": abort_reason})
                    if done:
                        break
                if info is None:
                    raise ValueError("Empty learned episode")
                for camera in ("front", "wrist"):
                    Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"terminal_{info['step']:04d}_{camera}.png")
                aborted, abort_reason = safety_abort_info(info)
                success = bool(info["success"] and not aborted and not initial_success)
                recorder.add(observation, f"{args.split_mode} | success={success}")
                result = {"file": row["file"], "task": task.name, "prompt": task.prompt,
                          "steps": int(info["step"]), "success": success,
                          "initial_semantic_success": initial_success,
                          "correct_target_grasp_seen": correct, "wrong_target_grasp_seen": wrong,
                          "final_ee_object_distance_m": info["ee_object_distance"],
                          "final_goal_xy_distance_m": info.get("goal_xy_distance_m"),
                          "strict_success": bool(info["strict_raw_success"]),
                          "reset_observation_check": reset_errors,
                          "safety_abort": aborted, "safety_abort_reason": abort_reason,
                          "safety_diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
                          "safety_clamped_steps": sum(frame["safety_clamped_action"] for frame in trace), "trace": trace}
                results.append(result)
                (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{row['file']}: success={success} steps={info['step']} safety_abort={result['safety_abort']}", flush=True)
        finally:
            recorder.close(); policy.close()
    verify_source_audit(source, args.audit_report)
    verify_source_audit(training, args.training_audit_report)
    verify_environment_runtime_fingerprint(runtime, bound_profile)
    verify_fingerprint(args.checkpoint)
    report = {"scope": scope, "split_mode": args.split_mode, "source": str(source),
              "source_index_sha256": audit["index_sha256"], "source_manifest_sha256": audit["manifest_sha256"],
              "source_payload_fingerprint": audit["source_payload_fingerprint"],
              "environment_runtime_fingerprint": runtime, "training_normalization_source": str(training),
              "saved_normalization_audit": normalization, "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_content_sha256": identity["content_tree_sha256"], "seed": args.seed,
              "replan_steps": 1, "required_success_steps": 1,
              "semantic_success_thresholds": env.semantic_success_thresholds,
              "trials": len(results), "successes": sum(row["success"] for row in results),
              "safety_aborts": sum(row["safety_abort"] for row in results),
              "initial_false_positives": sum(row["initial_semantic_success"] for row in results),
              "privileged_information_in_controls": False, "oracle_fallback": False, "final_test_used": False,
              "results": results}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
