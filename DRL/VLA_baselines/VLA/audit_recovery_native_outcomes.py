"""Supplemental FULL native replay: distinguish command drift from failure.

Never overwrites/waives the failed strict command-parity audit. Runs all
recorded labels, retains contact/state discrepancies and requires ORIGINAL
terminal task/body-grasp/touch-quality/safety outcomes. No model SR claimed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.benchmark import _interaction_audit
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
from VLA.simstate_scene64 import scene_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--strict-replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve every prior supplemental audit")
    args.output.mkdir(parents=True)
    plan = json.loads((args.candidate / "plan.json").read_text())
    rows = json.loads((args.candidate / "index.json").read_text())
    strict = json.loads(args.strict_replay.read_text())
    canonical = [r for r in strict["rows"] if r["mode"] == "canonical"]
    if len(rows) != 40 or len(canonical) != 38 or not all(r["passed"] for r in canonical):
        raise ValueError("All40 attempts plus38 EXACT canonical paths required")
    source = Path(plan["source"])
    originals = {r["file"]: r for r in json.loads((source / "index.json").read_text())}
    hashes = {str(path.resolve()): sha256_file(path) for path in (
        args.candidate / "plan.json", args.candidate / "index.json", args.strict_replay, Path(__file__))}
    reports = []
    for number, row in enumerate(rows):
        if not row.get("file"): continue
        path = args.candidate / row["file"]; digest = sha256_file(path)
        if digest != row["payload_sha256"]: raise ValueError("Candidate was modified")
        with np.load(path, allow_pickle=False) as archive: data = {k: archive[k] for k in archive.files}
        closures = native_gripper_closure(data)
        result = {"attempt": number, "file": row["file"], "passed": False, "error": None,
            "steps": 0, "command_error_max_rad": 0., "command_error_above_original_1e4_gate_frames": 0,
            "measured_joint_error_max_rad": 0., "body_contact_timeline_differing_frames": 0,
            "held_identity_timeline_differing_frames": 0, "scene64_max_abs_errors_by_feature": [0.] * 64}
        env = audit_environment(source, image_size=384, control_hz=10, seed=0,
            required_success_steps=1, use_semantic_success=False)
        wrapper = NearSurfaceTouchCollection(env, None, None) if row["task"]["name"] == "touch" else env
        try:
            obs = reset_source_episode(env, source, originals[row["source_file"]])
            for action in row["recovery_start_actions_not_training_labels"]:
                obs, _, done, info = env.step(np.asarray(action, np.float32))
                if done or info.get("safety_violation"): raise ValueError("Saved prefix changed")
            if np.max(np.abs(scene_state(env, obs) - row["initial_scene64"])) > 1e-6:
                raise ValueError("Initial actual handoff changed")
            env.step_count, env.success_streak, env._last_success = 0, 0, False
            for frame, reference in enumerate(data["canonical_actions"]):
                obs = env.observe()
                error = np.abs(scene_state(env, obs) - data["audit_pre_action_PRIVILEGED_state64"][frame])
                result["scene64_max_abs_errors_by_feature"] = np.maximum(result["scene64_max_abs_errors_by_feature"], error).tolist()
                result["measured_joint_error_max_rad"] = max(result["measured_joint_error_max_rad"],
                    float(np.max(np.abs(obs.joint_positions - data["joint_positions"][frame]))))
                action = env.joint_targets_to_canonical(data["next_joint_targets"][frame], float(closures[frame]))
                obs, _, done, info = wrapper.step(action)
                error = float(np.max(np.abs(env.requested_arm_targets - data["next_joint_targets"][frame])))
                result["command_error_max_rad"] = max(result["command_error_max_rad"], error)
                result["command_error_above_original_1e4_gate_frames"] += int(error > 1e-4)
                result["held_identity_timeline_differing_frames"] += int(
                    (_interaction_audit(env)["held_object"] or "") != str(data["audit_any_held_object_after"][frame]))
                result["body_contact_timeline_differing_frames"] += int(
                    bool(env.gripper._bilateral(env.pick_id)) != bool(data["next_bilateral_pad_contact"][frame]))
                result["steps"] += 1
                if info.get("safety_violation") or info.get("expert_quality_failure"):
                    raise ValueError("Original native safety/contact-quality outcome differs")
                # The unchanged source uses first-hit semantics. Preserve the
                # whole commanded trajectory rather than drop later labels.
            result["terminal_strict_success"] = bool(info["strict_raw_success"])
            if not result["terminal_strict_success"]: raise ValueError("Native terminal task outcome differs")
            if row["task"]["name"] == "touch":
                result["terminal_near_surface_gap_and_force"] = wrapper.terminal_geometry()
                if not wrapper._last_success: raise ValueError("Original near-surface quality differs")
            elif env.gripper.grasped_object != env.pick_id or not env.gripper._bilateral(env.pick_id):
                raise ValueError("Original terminal correct body grasp differs")
            result["passed"] = True
        except Exception as error: result["error"] = f"{type(error).__name__}: {error}"
        finally: env.close()
        if sha256_file(path) != digest: raise ValueError("Source labels changed")
        reports.append(result)
        atomic_json(args.output / "progress.json", {"rows": reports, "completed_native_paths": len(reports)})
        print(json.dumps({k: v for k, v in result.items() if k != "scene64_max_abs_errors_by_feature"}), flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()): raise ValueError("Source/code changed")
    proof = {"passed": all(r["passed"] for r in reports), "rows": reports, "paths": len(reports),
        "scope": "supplemental_native_complete_physical_outcomes_NOT_strict_command_parity_waiver",
        "strict_failed_report_preserved": str(args.strict_replay.resolve()), "source_hashes": hashes,
        "training_qualified": False, "new_model_trials": 0, "new_model_success_rate": None,
        "source_labels_or_physics_or_success_thresholds_modified": False}
    atomic_json(args.output / "audit.json", proof)


if __name__ == "__main__": main()
