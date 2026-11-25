"""Replay stored student prefix and every correction, canonical/native paths.

No student rerun, no new VLA score and no source promotion. Byte-exact
canonical sensor checks remain strict; report discrepancies rather than waive.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError("Preserve prior replay")
    plan = json.loads((candidate / "plan.json").read_text())
    index = json.loads((candidate / "index.json").read_text())
    if len(index) != 4 or not plan["candidate_only"] or any(row.get("error") or not row["success"] for row in index):
        raise ValueError("Requires complete four-case candidate, never filter failures")
    source = Path(plan["source"])
    originals = {row["file"]: row for row in json.loads((source / "index.json").read_text())}
    paths = [candidate / "plan.json", candidate / "index.json", Path(__file__), source / "index.json", source / "dataset_manifest.json"]
    hashes = {str(path): sha256_file(path) for path in paths}
    results = []
    for row in index:
        path = candidate / row["file"]
        hashes[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        closures = native_gripper_closure(arrays)
        for mode in ("canonical", "native_joint"):
            env = audit_environment(source, robot="ur5e", image_size=384, control_hz=10, seed=0,
                required_success_steps=1, use_semantic_success=False)
            correction = NearSurfaceTouchCollection(env, None, None) if row["task"]["name"] == "touch" else env
            result = {"file": row["file"], "task": row["task"]["name"], "mode": mode, "passed": False, "error": None}
            maximum = {"joint_rad": 0., "closure": 0., "front_pixel": 0, "wrist_pixel": 0}
            try:
                reset_source_episode(env, source, originals[row["source_file"]])
                if len(row["recovery_start_actions_not_training_labels"]) != 20:
                    raise ValueError("Predeclared full student prefix missing")
                for action in row["recovery_start_actions_not_training_labels"]:
                    _, _, done, info = env.step(np.asarray(action, np.float32))
                    if done or info.get("safety_violation"): raise ValueError("Stored student prefix aborted")
                if _interaction_audit(env)["held_object"] is not None: raise ValueError("Unsupported held start")
                env.step_count, env.success_streak, env._last_success = 0, 0, False
                if env.is_strict_success(): raise ValueError("Initial false-positive success")
                for frame, reference in enumerate(arrays["canonical_actions"]):
                    if mode == "canonical":
                        obs = env.observe()
                        maximum["joint_rad"] = max(maximum["joint_rad"], float(np.max(np.abs(obs.joint_positions-arrays["joint_positions"][frame]))))
                        maximum["closure"] = max(maximum["closure"], abs(env.measured_gripper_closure()-float(arrays["gripper_closure"][frame])))
                        for camera in ("front", "wrist"):
                            error = int(np.max(np.abs(getattr(obs,camera+"_rgb").astype(np.int16)-arrays[camera+"_rgb"][frame].astype(np.int16))))
                            maximum[camera+"_pixel"] = max(maximum[camera+"_pixel"], error)
                        if (_interaction_audit(env)["held_object"] or "") != str(arrays["audit_any_held_object_before"][frame]):
                            raise ValueError("Held identity differs before label")
                    action = reference if mode == "canonical" else env.joint_targets_to_canonical(arrays["next_joint_targets"][frame], float(closures[frame]))
                    _, _, done, info = correction.step(action)
                    if info.get("safety_violation") or info.get("expert_quality_failure"):
                        raise ValueError("Correction safety/quality abort: " + str(info))
                    if mode == "canonical" and (_interaction_audit(env)["held_object"] or "") != str(arrays["audit_any_held_object_after"][frame]):
                        raise ValueError("Held identity differs after label")
                    result.update(steps=frame+1, strict_success=bool(info["strict_raw_success"]))
                    if done:
                        if frame != len(arrays["canonical_actions"])-1: raise ValueError("Replay ends at another label")
                        break
                result["maximum_sensor_errors"] = maximum
                if mode == "canonical" and (maximum["joint_rad"] > 1e-6 or maximum["closure"] > 1e-6 or maximum["front_pixel"] or maximum["wrist_pixel"]):
                    raise ValueError("Exact canonical sensors differ")
                if row["task"]["name"] == "touch":
                    gap, force = correction.terminal_geometry()
                    result.update(near_surface_gap_m=gap, normal_force_n=force)
                    if not correction._last_success: raise ValueError("Near-surface touch quality failed")
                elif not env.gripper._bilateral(env.pick_id) or env.gripper.grasped_object != env.pick_id:
                    raise ValueError("Terminal correct body grasp not verified")
                if not result["strict_success"] or result["steps"] != len(arrays["canonical_actions"]):
                    raise ValueError("Full correction did not replay to strict success")
                result["passed"] = True
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                env.close()
            results.append(result)
            print(json.dumps(result),flush=True)
    if any(sha256_file(Path(path)) != digest for path,digest in hashes.items()): raise ValueError("Candidate changed")
    output.mkdir(parents=True)
    (output / "audit.json").write_text(json.dumps({"scope": "four_TRAIN_onpolicy_correction_physics_NOT_training_qualification",
        "passed": all(row["passed"] for row in results), "rows": results, "source_hashes": hashes,
        "neural_weights_loaded": False, "new_VLA_trials": 0, "training_qualified": False},indent=2)+"\n")


if __name__ == "__main__":
    main()
