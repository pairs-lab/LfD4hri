"""Replay every labeled recovery attempt, retain unlabeled/failed attempts.

No neural reroll, no automatic selection for training. Canonical sensors use
the ALREADY established numerical tolerance; no threshold/physics edit. The
native joint-command path independently checks command/held/outcome parity.
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
from VLA.evaluate_smolvla_v9_source import reset_observation_errors, validate_reset_observation
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
from VLA.simstate_scene64 import SCHEMA, scene_state


def replay(candidate, output):
    if output.exists(): raise FileExistsError("Preserve previous replay failures")
    output.mkdir(parents=True)
    plan = json.loads((candidate / "plan.json").read_text())
    rows = json.loads((candidate / "index.json").read_text())
    if plan["schema"] != SCHEMA or not plan["candidate_only"] or len(rows) != 40:
        raise ValueError("All40 predeclared attempts, not just a success subset, required")
    if [r["source_file"] for r in rows] != plan["selected_files"]:
        raise ValueError("Attempt order/scene selection changed")
    source = Path(plan["source"])
    originals = {r["file"]: r for r in json.loads((source / "index.json").read_text())}
    frozen = {str(p): sha256_file(p) for p in (candidate / "plan.json", candidate / "index.json",
        source / "index.json", Path(__file__), Path(__file__).with_name("simstate_scene64.py"))}
    reports = []; no_labels = []
    for number, row in enumerate(rows):
        if not row.get("file"):
            no_labels.append({"attempt": number, "source_file": row["source_file"],
                "error": row.get("error"), "prefix_trace_retained": True})
            continue
        path = candidate / row["file"]; frozen[str(path)] = sha256_file(path)
        if frozen[str(path)] != row["payload_sha256"]: raise ValueError("Candidate labels changed")
        with np.load(path, allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        closures = native_gripper_closure(data)
        if len(data["canonical_actions"]) != row["steps"]:
            raise ValueError("Incomplete saved correction")
        for mode in ("canonical", "native_joint"):
            result = {"attempt": number, "file": row["file"], "mode": mode,
                "original_expert_success_NOT_model_SR": bool(row["success"]), "passed": False,
                "frames_checked": 0, "scene64_max_abs_error": 0., "sensor_maxima": {}, "error": None}
            env = audit_environment(source, image_size=384, control_hz=10, seed=0,
                required_success_steps=1, use_semantic_success=False)
            correction = NearSurfaceTouchCollection(env, None, None) if row["task"]["name"] == "touch" else env
            try:
                obs = reset_source_episode(env, source, originals[row["source_file"]])
                prefix = row["recovery_start_actions_not_training_labels"]
                if len(prefix) != row["prefix_controls"]: raise ValueError("Incomplete predeclared prefix")
                for action in prefix:
                    obs, _, done, info = env.step(np.asarray(action, np.float32))
                    if done or info.get("safety_violation"): raise ValueError("Saved student prefix changed")
                error = float(np.max(np.abs(scene_state(env, obs) - row["initial_scene64"])))
                if error > 1e-6: raise ValueError("On-policy handoff state no longer replays")
                if _interaction_audit(env) != row["recovery_start_evidence"]:
                    raise ValueError("Actual held/contact identity at handoff changed")
                env.step_count, env.success_streak, env._last_success = 0, 0, False
                if env.is_strict_success(): raise ValueError("Initially completed correction")
                for frame, reference in enumerate(data["canonical_actions"]):
                    obs = env.observe()
                    error = float(np.max(np.abs(scene_state(env, obs) - data["audit_pre_action_PRIVILEGED_state64"][frame])))
                    result["scene64_max_abs_error"] = max(result["scene64_max_abs_error"], error)
                    if mode == "canonical":
                        if error > 1e-6: raise ValueError("Causal before-label state64 changed")
                        errors = reset_observation_errors(obs, env.normalized_gripper_closure(),
                            {k: data[k][frame:frame+1] for k in ("joint_positions", "gripper_closure", "front_rgb", "wrist_rgb")})
                        gate = validate_reset_observation(errors)
                        result["existing_sensor_tolerance"] = gate["limits"]
                        for key, value in errors.items():
                            result["sensor_maxima"][key] = max(result["sensor_maxima"].get(key, 0), value)
                        for camera in ("front", "wrist"):
                            for role in ("pick", "place"):
                                if not np.array_equal(env.target_mask(camera, role), data[f"{camera}_{role}_mask"][frame]):
                                    raise ValueError("Requested object/goal silhouette changed")
                    if (_interaction_audit(env)["held_object"] or "") != str(data["audit_any_held_object_before"][frame]):
                        raise ValueError("Held identity before label changed")
                    action = reference if mode == "canonical" else env.joint_targets_to_canonical(
                        data["next_joint_targets"][frame], float(closures[frame]))
                    obs, _, done, info = correction.step(action)
                    if mode == "native_joint":
                        command_error = float(np.max(np.abs(env.requested_arm_targets - data["next_joint_targets"][frame])))
                        result["native_command_max_error_rad"] = max(result.get("native_command_max_error_rad", 0.), command_error)
                        if command_error > 1e-4: raise ValueError("Native requested joint command parity lost")
                    if (_interaction_audit(env)["held_object"] or "") != str(data["audit_any_held_object_after"][frame]):
                        raise ValueError("Held identity after label changed")
                    if bool(info["raw_success"]) != bool(data["next_raw_success"][frame]):
                        raise ValueError("Recorded success timeline changed")
                    if bool(env.gripper._bilateral(env.pick_id)) != bool(data["next_bilateral_pad_contact"][frame]):
                        raise ValueError("Bilateral-contact timeline changed")
                    result["frames_checked"] += 1
                    if done:
                        if frame != row["steps"] - 1: raise ValueError("Another termination frame")
                        break
                if result["frames_checked"] != row["steps"] or bool(info["success"]) != bool(row["success"]):
                    raise ValueError("Full attempt/outcome not reproduced")
                if row["success"] and row["task"]["name"] != "touch" and (
                    env.gripper.grasped_object != env.pick_id or not env.gripper._bilateral(env.pick_id)):
                    raise ValueError("Terminal requested body grasp missing")
                if row["success"] and row["task"]["name"] == "touch" and not correction._last_success:
                    raise ValueError("Near-surface expert quality missing")
                result["passed"] = True
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            finally:
                env.close()
            reports.append(result)
            atomic_json(output / "progress.json", {"replayed_paths": len(reports), "rows": reports,
                "unlabeled_attempts_retained": no_labels, "training_qualified": False})
            print(json.dumps(result), flush=True)
    if any(sha256_file(Path(p)) != digest for p, digest in frozen.items()): raise ValueError("Source/code changed")
    passed = bool(reports) and all(r["passed"] for r in reports)
    proof = {"passed": passed, "scope": "all_predeclared_TRAIN_recovery_attempts_physics_NOT_model_SR",
        "declared_attempts": len(rows), "replayed_paths": len(reports), "rows": reports,
        "unlabeled_attempts_retained": no_labels, "source_hashes": frozen, "training_qualified": False,
        "automatic_failed_attempt_filtering": False, "remaining_gates": [
            "Review failed/early-complete prefixes and correction-label quality", "Full source64 extraction",
            "New64 input/action normalization and native batched loader parity",
            "Actual CUDA updates/saved precision/exact reload and independent neural evaluation"]}
    atomic_json(output / "audit.json", proof)
    if not passed: raise ValueError("Recovery replay has failures; never silently train")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); replay(args.candidate.resolve(), args.output.resolve())


if __name__ == "__main__": main()
