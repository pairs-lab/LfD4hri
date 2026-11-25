"""CPU expert-feasibility probe from recorded wrong-hold state, NOT training.

Never a learned-controller fallback. Replays original neural prefix unchanged,
then tests explicit OPEN/retreat/retarget expert labels. No teleport, gripper
guard relaxation, dataset promotion, extra VLA trial or learned-SR claim.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.benchmark import _interaction_audit
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher


class WrongHeldOpeningProbe:
    """Offline expert only; explicit held-wrong, not contact-free empty reset."""
    def __init__(self, env):
        self.env = env

    def reset(self, task):
        self.task = task
        self.wrong_body = self.env.gripper.grasped_object
        if self.wrong_body is None or self.wrong_body == self.env.pick_id:
            raise ValueError("Probe needs an actual physically held distractor")
        self.opening = True
        self.delegate = StagedAcquisitionTeacher(self.env)
        self.open_steps = 0
        self.retreat_height = max(.22, float(self.env.gripper.grasp_center()[2]) + .08)

    def predict(self, observation):
        if self.opening:
            held = self.env.gripper.grasped_object
            if held is not None and held != self.wrong_body:
                raise ValueError("Unexpected new held body during corrective opening")
            self.open_steps += 1
            if self.open_steps > 40:
                raise ValueError("Corrective OPEN did not converge in bounded steps")
            center = self.env.gripper.grasp_center()
            if held is None and self.env.measured_gripper_closure() < .02 and center[2] >= self.retreat_height - .008:
                self.opening = False
                self.delegate.reset(self.task)
            else:
                # First release without dragging a held distractor upward.
                dz = 0. if held is not None else float(np.clip(.65 * (self.retreat_height - center[2]), 0, .015))
                return np.asarray([0, 0, dz, 0, 0, 0, -1], np.float32)
        return self.delegate.predict(observation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve every probe outcome")
    run, output = args.run.resolve(), args.output.resolve()
    paths = [run / p for p in ("manifest.json", "results.json", "trajectory.json")]
    fixed = {str(p): sha256_file(p) for p in paths}
    manifest, _, traces = [json.loads(p.read_text()) for p in paths]
    task = TaskSpec.from_dict(next(t for t in manifest["tasks"] if t["name"] == args.task))
    trace = traces[args.task]
    first = next((i for i, row in enumerate(trace) if row["interaction_after_audit_only"]["held_object"] is not None
                  and not row["interaction_after_audit_only"]["correct_object_held"]), None)
    if first is None:
        raise ValueError("Original episode lacks a recorded actual wrong hold")
    output.mkdir(parents=True)
    report = {"source_run": str(run), "source_hashes": fixed, "task": task.to_dict(),
        "scope": "expert_recovery_feasibility_only_NOT_dataset_or_learned_success", "neural_weights_loaded": False,
        "training_performed": False, "new_VLA_trial": False, "final_test_used": False,
        "runtime_safety_or_cameras_changed": False, "recoverable": False, "error": None, "steps": []}
    profile = manifest["observation_profile"]
    env = environment_class_for_profile(profile)(observation_profile=profile, robot=manifest["robot"],
        image_size=manifest["image_size"], control_hz=manifest["control_hz"], seed=manifest["seed"],
        required_success_steps=1, success_threshold_scale=manifest["reward_contract"]["threshold_scale"],
        use_semantic_success=True, semantic_success_thresholds=manifest["reward_contract"]["semantic_success_thresholds"])
    def pictures(stem):
        obs = env.observe()
        for view in ("front", "wrist"):
            Image.fromarray(getattr(obs, view + "_rgb")).save(output / (stem + "_" + view + ".png"))
    try:
        env.reset(task)
        for row in trace[:first + 1]:
            obs = env.observe()
            if not np.array_equal(obs.joint_positions, np.asarray(row["policy_observation"]["joint_positions_rad"], np.float32)):
                raise ValueError("Original replay proprioception differs")
            _, _, done, info = env.step(np.asarray(row["action"], np.float32))
            if done or info.get("safety_violation"):
                raise ValueError("Original prefix terminated before recovery state")
        state = _interaction_audit(env)
        if state["held_object"] != trace[first]["interaction_after_audit_only"]["held_object"]:
            raise ValueError("Original wrong-held body identity differs")
        report["initial_wrong_hold"] = state
        report["prefix_steps_not_training_labels"] = first + 1
        pictures("wrong_hold_start")
        env.step_count, env.success_streak, env._last_success = 0, 0, False
        teacher = WrongHeldOpeningProbe(env)
        teacher.reset(task)
        for index in range(task.max_steps):
            action = teacher.predict(env.observe())
            _, _, done, info = env.step(action)
            report["steps"].append({"step": index + 1, "expert_action_NOT_policy": action.tolist(),
                "interaction": _interaction_audit(env), "success": info["success"],
                "strict_success": info["strict_raw_success"], "safety_violation": info.get("safety_violation")})
            if index in (0, 10): pictures(f"correction_{index+1:03}")
            if done:
                report["recoverable"] = bool(info["success"] and not info.get("safety_violation"))
                break
        pictures("probe_terminal")
        if any(sha256_file(Path(p)) != digest for p, digest in fixed.items()):
            raise ValueError("Original evidence changed")
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        env.close()
        (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"recoverable_expert_only": report["recoverable"], "error": report["error"],
        "correction_steps": len(report["steps"])}, indent=2))


if __name__ == "__main__":
    main()
