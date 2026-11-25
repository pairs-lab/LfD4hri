"""Exact canonical and real native-joint replay of the isolated TRAIN pilot.

Explicit stored physical startup, never object teleport or teacher regeneration.
Not qualification: source/readiness/quantity/native-camera gates remain required.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.benchmark import _interaction_audit
from VLA.collect_wrong_held_recovery_pilot import PhysicalWrongHeldReset
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment


def stored_reset(env, row):
    task = TaskSpec.from_dict(row["task"])
    env.reset(task)
    saved = {name: copy.deepcopy(getattr(env, name)) for name in
             ("initial_pick_z", "initial_pick_position", "initial_pick_top", "initial_pick_bottom")}
    proxy = PhysicalWrongHeldReset(env)
    proxy.retarget(TaskSpec.from_dict(row["startup_task_not_training_instruction"]))
    for action in row["recovery_start_actions_not_training_labels"]:
        _, _, done, info = env.step(np.asarray(action, np.float32))
        if done or info.get("safety_violation"):
            raise ValueError("Stored physical startup terminated or safety-aborted")
    proxy.retarget(task)
    for name, value in saved.items(): setattr(env, name, value)
    actual = _interaction_audit(env)
    evidence = row["recovery_start_evidence"]
    if (actual["held_object"] != evidence["held_object"] or actual["correct_object_held"]
            or not env.gripper._bilateral(env.gripper.grasped_object)):
        raise ValueError("Actual wrong-held identity/bilateral reset evidence differs")
    env.step_count, env.success_streak, env._last_success = 0, 0, False
    return env.observe()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Keep each failed replay")
    candidate = args.candidate.resolve()
    plan = json.loads((candidate / "plan.json").read_text())
    index = json.loads((candidate / "index.json").read_text())
    if not plan["candidate_only"] or len(index) != plan["planned_episodes"]:
        raise ValueError("Complete isolated candidate required")
    bound = {str(candidate / "plan.json"): sha256_file(candidate / "plan.json"),
             str(candidate / "index.json"): sha256_file(candidate / "index.json"),
             str(Path(__file__).resolve()): sha256_file(Path(__file__))}
    source = Path(plan["source"])
    results = []
    for row in index:
        if row.get("error") or not row.get("teacher_success_and_feedback_checks"):
            raise ValueError("Pilot includes unsuccessful/unverified corrections; retain and repair")
        path = candidate / row["file"]
        bound[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
            closures = native_gripper_closure(arrays)
        for mode in ("canonical", "native_joint"):
            env = audit_environment(source, robot="ur5e", image_size=384, control_hz=10, seed=0,
                                    required_success_steps=1, use_semantic_success=False)
            env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(row)
            maximum = {"joint_rad": 0., "closure": 0., "front_pixel": 0, "wrist_pixel": 0}
            result = {"file": row["file"], "task": row["task"]["name"], "mode": mode,
                      "strict_success": False, "initial_strict_success": None, "error": None}
            try:
                stored_reset(env, row)
                result["initial_strict_success"] = bool(env.is_strict_success())
                if result["initial_strict_success"]: raise ValueError("Initial false-positive task success")
                for frame, reference in enumerate(arrays["canonical_actions"]):
                    obs = env.observe()
                    if mode == "canonical":
                        maximum["joint_rad"] = max(maximum["joint_rad"], float(np.max(np.abs(obs.joint_positions - arrays["joint_positions"][frame]))))
                        maximum["closure"] = max(maximum["closure"], abs(env.measured_gripper_closure() - float(arrays["gripper_closure"][frame])))
                        for camera in ("front", "wrist"):
                            difference = int(np.max(np.abs(getattr(obs, camera+"_rgb").astype(np.int16) - arrays[camera+"_rgb"][frame].astype(np.int16))))
                            maximum[camera+"_pixel"] = max(maximum[camera+"_pixel"], difference)
                        if (_interaction_audit(env)["held_object"] or "") != str(arrays["audit_any_held_object_before"][frame]):
                            raise ValueError("Held identity differs before a correction label")
                    action = reference if mode == "canonical" else env.joint_targets_to_canonical(arrays["next_joint_targets"][frame], float(closures[frame]))
                    _, _, done, info = env.step(action)
                    if info.get("safety_violation"): raise ValueError(info["safety_violation"])
                    result.update(steps=frame+1, strict_success=bool(info["strict_raw_success"]),
                                  correct_object_held=_interaction_audit(env)["correct_object_held"])
                    if mode == "canonical" and (_interaction_audit(env)["held_object"] or "") != str(arrays["audit_any_held_object_after"][frame]):
                        raise ValueError("Held identity differs after a correction label")
                    if done:
                        if mode == "canonical" and frame != len(arrays["canonical_actions"])-1:
                            raise ValueError("Canonical replay terminates at a different correction label")
                        break
                result["maximum_sensor_errors"] = maximum
                if mode == "canonical" and (maximum["joint_rad"] > 1e-6 or maximum["closure"] > 1e-6
                        or maximum["front_pixel"] or maximum["wrist_pixel"]):
                    raise ValueError("Exact canonical image/feedback replay differs")
                contacts = p.getContactPoints(env.gripper.body, env.pick_id)
                closest = p.getClosestPoints(env.gripper.body, env.pick_id, distance=.025)
                result.update(terminal_surface_gap_m=min((c[8] for c in closest), default=None),
                              terminal_max_normal_force_n=max((c[9] for c in contacts), default=0.))
            except Exception as exc:
                result["error"] = str(exc)
            finally:
                env.close()
            results.append(result)
            print(json.dumps(result), flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in bound.items()):
        raise ValueError("Candidate/replay evidence changed")
    args.output.mkdir(parents=True)
    report = {"scope": "isolated4_TRAIN_wrong_held_correction_physics_NOT_training_qualification",
        "passed": all(r["strict_success"] and not r["error"] for r in results), "rows": results,
        "neural_weights_loaded": False, "new_VLA_trials": 0, "training_qualified": False,
        "candidate_source_unchanged": True, "source_hashes": bound, "final_test_used": False}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
