"""Replay every label of the isolated near-surface touch repair.

The expert's quality continuation is separate from deployed first-hit success.
No qualification, learned-model result, or change to deployed scoring follows.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.dataset_replay_context import audit_environment
from VLA.replay_wrong_held_recovery_pilot import stored_reset
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate, output = args.candidate.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve previous audits")
    plan = json.loads((candidate / "plan.json").read_text())
    index = json.loads((candidate / "index.json").read_text())
    touches = [row for row in index if row["task"]["name"] == "touch"]
    if not plan["candidate_only"] or len(touches) != 1:
        raise ValueError("Only the isolated four-task pilot's touch refinement is covered")
    row = touches[0]
    if not row["near_surface_touch_quality"]["strict_expert_success"]:
        raise ValueError("Near-surface teacher quality is required")
    files = [candidate / "plan.json", candidate / "index.json", candidate / row["file"], Path(__file__).resolve()]
    hashes = {str(path): sha256_file(path) for path in files}
    with np.load(candidate / row["file"], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    closures = native_gripper_closure(data)
    results = []
    for mode in ("canonical", "native_joint"):
        env = audit_environment(Path(plan["source"]), robot="ur5e", image_size=384,
            control_hz=10, seed=0, required_success_steps=1, use_semantic_success=False)
        quality = NearSurfaceTouchCollection(env, None, None)
        env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(row)
        result = {"mode": mode, "passed": False, "error": None}
        maximum = {"joint_rad": 0., "closure": 0., "front_pixel": 0, "wrist_pixel": 0}
        try:
            stored_reset(env, row)
            if env.is_strict_success():
                raise ValueError("Initial false-positive success")
            for frame, reference in enumerate(data["canonical_actions"]):
                if mode == "canonical":
                    obs = env.observe()
                    maximum["joint_rad"] = max(maximum["joint_rad"], float(np.max(np.abs(obs.joint_positions - data["joint_positions"][frame]))))
                    maximum["closure"] = max(maximum["closure"], abs(env.measured_gripper_closure() - float(data["gripper_closure"][frame])))
                    for camera in ("front", "wrist"):
                        error = int(np.max(np.abs(getattr(obs, camera+"_rgb").astype(np.int16) - data[camera+"_rgb"][frame].astype(np.int16))))
                        maximum[camera+"_pixel"] = max(maximum[camera+"_pixel"], error)
                    if (_interaction_audit(env)["held_object"] or "") != str(data["audit_any_held_object_before"][frame]):
                        raise ValueError("Held identity differs before label")
                action = reference if mode == "canonical" else env.joint_targets_to_canonical(data["next_joint_targets"][frame], float(closures[frame]))
                _, _, done, info = quality.step(action)
                if info.get("safety_violation") or info.get("expert_quality_failure"):
                    raise ValueError(str(info))
                if mode == "canonical" and (_interaction_audit(env)["held_object"] or "") != str(data["audit_any_held_object_after"][frame]):
                    raise ValueError("Held identity differs after label")
                result["steps"] = frame + 1
                if done:
                    if frame != len(data["canonical_actions"]) - 1:
                        raise ValueError("Quality replay terminates at a different label")
                    break
            gap, force = quality.terminal_geometry()
            result.update(strict_success=env.is_strict_success(), surface_gap_m=gap, max_normal_force_n=force,
                maximum_sensor_errors=maximum, all_labels_replayed=result["steps"] == len(data["canonical_actions"]))
            if mode == "canonical" and (maximum["joint_rad"] > 1e-6 or maximum["closure"] > 1e-6 or maximum["front_pixel"] or maximum["wrist_pixel"]):
                raise ValueError("Exact canonical sensor replay differs")
            if not result["all_labels_replayed"] or not quality._last_success or gap is None or gap > .0002 or force > 5:
                raise ValueError("Terminal expert near-surface quality failed")
            result["passed"] = True
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            env.close()
        results.append(result)
        print(json.dumps(result), flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError("Candidate/code changed during audit")
    output.mkdir(parents=True)
    report = {"scope": "isolated_TRAIN_touch_quality_replay_NOT_qualification", "passed": all(r["passed"] for r in results),
        "rows": results, "qualified_for_training": False, "new_VLA_trials": 0,
        "deployed_success_unchanged": True, "source_hashes": hashes}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
