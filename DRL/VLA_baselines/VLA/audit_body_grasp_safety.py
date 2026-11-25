"""Adversarial downward-command audit; never scores it as a task success."""
import argparse
import json
from pathlib import Path
import numpy as np

from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.physics_safety import surface_clearance
from VLA.common.tasks import TASKS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    profile = json.loads(Path("VLA/profiles/wrist_v8_h65_f80_workspace_scene_grasp.json").read_text())
    profile.update(name="body_v9_safety_audit", gripper_controller="body_contact_v9")
    records = []
    with BodyGraspEnvironment(profile, render_observations=False, required_success_steps=1, use_semantic_success=True) as env:
        env.reset(TASKS["touch"])
        for step in range(100):
            _, _, done, info = env.step(np.array([0, 0, -.05, 0, 0, 0, -1], dtype=np.float32))
            records.append({"step": step, "clearance_m": surface_clearance(env),
                "clamped": info["safety_clamped_action"], "executed_dz": info["executed_canonical_action"][2],
                "safety_abort": info.get("safety_violation"), "rollback": info.get("rollback_restored")})
            if done:
                break
        minimum = min(r["clearance_m"] for r in records)
        report = {"passed": minimum >= -.0008 and any(r["clamped"] or r["safety_abort"] for r in records),
            "minimum_restored_surface_clearance_m": minimum, "maximum_mimic_error_rad": env.safety_max_mimic_error,
            "records": records, "scope": "Repeated unsafe downward action test, not task/learned success",
            "legacy_or_rl_changed": False, "final_test_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("Downward safety guard failed; do not deploy")


if __name__ == "__main__":
    main()
