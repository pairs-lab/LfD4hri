"""CPU-only replay of one recorded action trace; not a new learned trial.

Verifies original proprioception, requested/held object and terminal outcomes.
Names pre-rollback collision bodies; never alters commands, guards or data.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.benchmark import _interaction_audit
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve recorded diagnostics")
    run = args.run.resolve()
    paths = [run / p for p in ("manifest.json", "results.json", "trajectory.json")]
    fixed = {str(p): sha256_file(p) for p in paths}
    manifest, result, trace = [json.loads(p.read_text()) for p in paths]
    task = next(x for x in manifest["tasks"] if x["name"] == args.task)
    original = next(x for x in result["tasks"] if x["name"] == args.task)
    steps = trace[args.task]
    if not original.get("safety_violation") or len(steps) != original["steps"]:
        raise ValueError("Complete safety-aborted original trace required")
    profile = manifest["observation_profile"]
    cls = environment_class_for_profile(profile)
    env = cls(observation_profile=profile, robot=manifest["robot"], image_size=manifest["image_size"],
        control_hz=manifest["control_hz"], seed=manifest["seed"], required_success_steps=1,
        success_threshold_scale=manifest["reward_contract"]["threshold_scale"], use_semantic_success=True,
        semantic_success_thresholds=manifest["reward_contract"]["semantic_success_thresholds"],
        render_observations=False)
    maximum_errors = {"joint_rad": 0., "pad_position_m": 0., "object_position_m": 0.}
    feedback = []
    try:
        env.reset(TaskSpec.from_dict(task))
        names = {body: name for name, body in env.object_ids.items()}
        for row in steps:
            state = _interaction_audit(env)
            obs = env.observe()
            maximum_errors["joint_rad"] = max(maximum_errors["joint_rad"], float(np.max(np.abs(
                obs.joint_positions - row["policy_observation"]["joint_positions_rad"]))))
            for key, target in (("pad_position_m", "pad_grasp_center_xyz_m"), ("object_position_m", "object_base_xyz_m")):
                maximum_errors[key] = max(maximum_errors[key], float(np.max(np.abs(
                    np.asarray(state[target]) - row["interaction_before_audit_only"][target]))))
            _, _, done, info = env.step(np.asarray(row["action"], dtype=np.float32))
            if (info.get("safety_violation") != row.get("safety_violation")
                    or bool(info["semantic_raw_success"]) != bool(row["semantic_raw_success"])):
                raise ValueError("Recorded safety/success differs; no causal claim from mismatched replay")
            diagnostics = info.get("safety_diagnostics_before_rollback")
            if diagnostics:
                diagnostics = {**diagnostics, "contacts": [{**c,
                    "other_body_name": names.get(c["other_body"], "robot_or_static_environment"),
                    "is_requested_object": c["other_body"] == env.pick_id} for c in diagnostics["contacts"]]}
            feedback.append({"step": row["step"], "requested_object": env.task.pick_name,
                "safety_violation": info.get("safety_violation"),
                "pre_rollback_diagnostics": diagnostics})
            if done and row is not steps[-1]:
                raise ValueError("Replay terminated before recorded final step")
        if any(v > 1e-6 for v in maximum_errors.values()):
            raise ValueError(f"Recorded sensor/geometry differs: {maximum_errors}")
        if any(sha256_file(Path(p)) != h for p, h in fixed.items()):
            raise ValueError("Original trace changed during diagnostic")
        args.output.mkdir(parents=True)
        report = {"passed_exact_recorded_replay": True, "task": args.task, "source_run": str(run),
            "source_hashes": fixed, "maximum_replay_errors": maximum_errors, "steps": len(steps),
            "feedback": feedback, "neural_weights_loaded": False, "new_evaluation_trial": False,
            "actions_or_guards_changed": False, "training_performed": False, "final_test_used": False}
        (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": True, "errors": maximum_errors, "terminal": feedback[-1]}, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
