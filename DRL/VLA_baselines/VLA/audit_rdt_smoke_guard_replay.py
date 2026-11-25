"""Replay all recorded untrained RDT smoke commands; verify guard rollback.

No model loading/new scores, no disabling guards or changing commands. A
two-update model's unsafe command is a failed manipulation, not an API defect.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def state(env):
    return np.asarray([p.getJointStates(body, list(range(p.getNumJoints(body))))[j][:2]
        for body in (env.robot_id, env.gripper.body) for j in range(p.getNumJoints(body))], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve earlier guard replays")
    report = json.loads(args.integration.read_text())
    source = Path(report["source"])
    rows = {row["file"]: row for row in json.loads((source / "index.json").read_text())}
    hashes = {str(path.resolve()): sha256_file(path) for path in (args.integration, source / "index.json", source / "dataset_manifest.json", Path(__file__))}
    results = []
    for reference in report["results"]:
        env = audit_environment(source, image_size=384, required_success_steps=1, success_threshold_scale=1., use_semantic_success=True)
        result = {"file": reference["file"], "task": reference["task"], "passed": False, "error": None}
        try:
            reset_source_episode(env, source, rows[reference["file"]])
            for step in reference["trace"]:
                before = state(env)
                _, _, done, info = env.step(np.asarray(step["action"], np.float32))
                actual_abort = bool(info.get("safety_violation"))
                if actual_abort != step["safety_abort"]:
                    raise ValueError("Recorded abort does not reproduce")
                if actual_abort:
                    diagnostics = info["safety_diagnostics_before_rollback"]
                    expected = reference["safety_diagnostics_before_rollback"]
                    error = float(np.max(np.abs(np.asarray(diagnostics["joint_positions_rad"])-expected["joint_positions_rad"])))
                    velocity_error = float(np.max(np.abs(np.asarray(diagnostics["joint_velocities_rad_s"])-expected["joint_velocities_rad_s"])))
                    restored_error = float(np.max(np.abs(state(env)-before)))
                    result.update(pre_rollback_joint_error_rad=error, pre_rollback_velocity_error=velocity_error,
                        restored_state_max_error=restored_error, rollback_restored=info.get("rollback_restored"),
                        guard_reason=info["safety_violation"], clearance_m=diagnostics["surface_clearance_m"], contacts=diagnostics["contacts"])
                    if error > 1e-6 or velocity_error > 1e-5 or restored_error > 1e-9 or info.get("rollback_restored") is not True or info["success"]:
                        raise ValueError("Recorded guard/rollback differs")
                    if not done: raise ValueError("Guard did not terminate unsafe rollout")
                elif done:
                    raise ValueError("Unexpected terminal event")
            result.update(passed=True, reproduced_safety_abort=reference["safety_abort"], steps=len(reference["trace"]))
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            env.close()
        results.append(result)
        print(json.dumps(result), flush=True)
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()): raise ValueError("Proof changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output).write_text(json.dumps({"passed": all(row["passed"] for row in results), "rows": results,
        "scope": "all_recorded_smoke_commands_guard_rollback_replay_NOT_learned_success", "bound_files": hashes,
        "new_neural_trials": 0, "safety_guards_changed": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
