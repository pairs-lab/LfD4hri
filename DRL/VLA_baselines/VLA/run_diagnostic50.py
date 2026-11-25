"""User-requested50/action development diagnostics, not held-out paper test.

Uses the unchanged learned adapter and benchmark controller/scoring. No oracle
pre-filter, rerolls, early stopping for poor scores, or checkpoint selection.
Seeds explicitly become DEVELOPMENT after failures are used for repairs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from VLA.common.provenance import sha256_file
from VLA.common.evaluation_statistics import success_statistics
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("touch", "pick", "move", "pick_place")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.run_tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Fresh output and simple immutable tag required")
    checkpoint = args.checkpoint.resolve()
    identity = verify_fingerprint(checkpoint, full=True)
    template_path = ROOT / "VLA/evaluation_protocol_v9_postfit_10_per_action.json"
    template = json.loads(template_path.read_text())
    if profile_hash(checkpoint_profile(str(checkpoint))) != template["observation_profile_sha256"]:
        raise ValueError("Checkpoint and common robot/camera profile differ")
    seeds = list(range(20267001, 20267051))
    reserved = json.loads((ROOT / template["reserved_final_protocol"]).read_text())
    if set(seeds) & set(reserved.get("final_test_seeds", [])):
        raise ValueError("Diagnostic seeds overlap reserved final benchmark")
    output = args.output.resolve()
    output.mkdir(parents=True)
    logs = output / "logs"
    logs.mkdir()
    frozen = {str(path): sha256_file(path) for path in
              (Path(__file__).resolve(), ROOT / "VLA/benchmark.py", template_path)}
    plan = {"model": "smolvla", "checkpoint": str(checkpoint),
            "checkpoint_content_sha256": identity["content_tree_sha256"],
            "tasks": list(TASKS), "seeds": seeds, "trials_per_action": 50,
            "planned_trials": 200, "run_tag": args.run_tag, "frozen_files": frozen,
            "success": template["success"], "observation_profile_sha256": template["observation_profile_sha256"],
            "scope": "error_discovery_DEVELOPMENT_NOT_final_benchmark",
            "final_test_used": False, "oracle_fallback": False,
            "oracle_outcome_seed_filter": False, "video_seed": seeds[0],
            "integration_error_is_task_failure": False}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    def save(status, error=None):
        by_task = {task: success_statistics([r for r in rows if r["task"] == task])
                   for task in TASKS if any(r["task"] == task for r in rows)}
        (output / "progress.json").write_text(json.dumps({**plan, "pid": os.getpid(),
            "status": status, "error": error, "completed_trials": len(rows),
            "counts": dict(Counter(r["task"] for r in rows)), "by_task": by_task,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2) + "\n")
    try:
        save("running")
        for seed in seeds:
            if any(sha256_file(Path(p)) != digest for p, digest in frozen.items()):
                raise ValueError("Locked runner/config changed; do not mix policies")
            test_id = f"{args.run_tag}_seed{seed}_rp1"
            result_path = ROOT / "VLA/smolvla" / test_id / "results.json"
            if result_path.parent.exists():
                raise FileExistsError("Never overwrite or silently resume an old trial")
            command = [sys.executable, "-m", "VLA.benchmark", "--model", "smolvla",
                "--robot", "ur5e", "--tasks", ",".join(TASKS), "--test-id", test_id,
                "--checkpoint", str(checkpoint), "--seed", str(seed), "--scene-profile", "rl_random_env",
                "--replan-steps", "1", "--control-hz", "10", "--required-success-steps", "1",
                "--success-mode", "semantic", "--success-threshold-scale",
                str(template["success"]["reward_xml_threshold_scale"]), "--audit-observation-every", "10"]
            for key, value in template["success"]["semantic_thresholds"].items():
                command += ["--semantic-" + key.replace("_", "-"), str(value)]
            if seed != seeds[0]: command.append("--no-video")
            with (logs / f"seed{seed}.log").open("x") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result_path.exists():
                report = json.loads(result_path.read_text())
                for task in report["tasks"]:
                    rows.append({"seed": seed, "task": task["name"], "success": task["success"],
                        "safety_violation": task.get("safety_violation"), "error": task.get("error"),
                        "initial_semantic_success": task.get("initial_semantic_success"),
                        "result": str(result_path), "trajectory": str(result_path.parent / "trajectory.json"),
                        "video": str(result_path.parent / "rollout.mp4") if seed == seeds[0] else None})
                save("running")
            if result.returncode:
                raise RuntimeError(f"Integration failed on seed{seed}; preserved partial evidence, no reroll")
            print(f"Completed {len(rows)}/200 learned trials", flush=True)
        verify_fingerprint(checkpoint, full=True)
        if Counter(r["task"] for r in rows) != {task: 50 for task in TASKS}:
            raise ValueError("Incomplete four-task50/action diagnostic")
        save("completed200_development_trials")
    except Exception as exc:
        save("stopped_integration_error_no_reroll", str(exc))
        raise


if __name__ == "__main__":
    main()
