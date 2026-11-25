"""Ten fixed development trials/action, optionally adopting prior exact runs.

User-requested budget revision, never outcome filtering. Preserve an in-flight
old child and its results. No fallback, rerolls, shortened task horizons or
checkpoint/scoring changes. No training data derived from evaluation labels.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.common.evaluation_statistics import success_statistics
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.runtime_provenance import environment_runtime_fingerprint

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("touch", "pick", "move", "pick_place")
SEEDS = list(range(20267001, 20267011))


def rows_from_result(path, seed, video):
    result = json.loads(path.read_text())
    tasks = result["tasks"]
    if Counter(t["name"] for t in tasks) != {t: 1 for t in TASKS} or any(t.get("error") for t in tasks):
        raise ValueError("Incomplete or integration-failed trial; preserve, no reroll")
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (manifest["seed"] != seed or result["model"] != "smolvla" or manifest["replan_steps"] != 1
            or manifest.get("privileged_object_pose_given_to_vla") is not False):
        raise ValueError("Trial identity differs")
    return [{"seed": seed, "task": t["name"], "success": t["success"],
        "safety_violation": t.get("safety_violation"), "error": t.get("error"),
        "initial_semantic_success": t.get("initial_semantic_success"),
        "result": str(path), "trajectory": str(path.parent / "trajectory.json"),
        "video": str(path.parent / "rollout.mp4") if video else None} for t in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--adopt-from", type=Path)
    parser.add_argument("--inflight-pid", type=int)
    args = parser.parse_args()
    if args.output.exists() or not args.run_tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("New output/simple run tag required")
    cp = args.checkpoint.resolve()
    identity = verify_fingerprint(cp, full=True)
    template_path = ROOT / "VLA/evaluation_protocol_v9_postfit_10_per_action.json"
    template = json.loads(template_path.read_text())
    profile = checkpoint_profile(str(cp))
    if profile_hash(profile) != template["observation_profile_sha256"]:
        raise ValueError("Camera/controller profile mismatch")
    reserved = json.loads((ROOT / template["reserved_final_protocol"]).read_text())
    if set(SEEDS) & set(reserved.get("final_test_seeds", [])):
        raise ValueError("Reserved test leakage")
    old = None
    if args.adopt_from:
        old = json.loads((args.adopt_from / "progress.json").read_text())
        if (Path(old["checkpoint"]).resolve() != cp
                or old["checkpoint_content_sha256"] != identity["content_tree_sha256"]
                or old["success"] != template["success"] or old["seeds"][:10] != SEEDS
                or max((r["seed"] for r in old["rows"]), default=SEEDS[0]) > SEEDS[-1]):
            raise ValueError("Budget revision cannot change checkpoint/rules/seed prefix")
        for path, digest in old["frozen_files"].items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("Original rollout code/config changed")
    elif args.inflight_pid:
        raise ValueError("In-flight adoption requires exact old run provenance")
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    frozen = {str(p): sha256_file(p) for p in (Path(__file__).resolve(), ROOT / "VLA/benchmark.py", template_path)}
    runtime = environment_runtime_fingerprint(profile)
    for name in (*runtime["sources_sha256"], "asset/params/reward.xml", "utils/random_env.py", "utils/camera.py"):
        frozen[str(ROOT / name)] = sha256_file(ROOT / name)
    if old:
        frozen[str(args.adopt_from.resolve() / "plan.json")] = sha256_file(args.adopt_from / "plan.json")
    plan = {"model": "smolvla", "checkpoint": str(cp), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "tasks": list(TASKS), "seeds": SEEDS, "trials_per_action": 10, "planned_trials": 40,
        "run_tag": args.run_tag, "frozen_files": frozen, "environment_runtime_fingerprint": runtime,
        "success": template["success"], "observation_profile_sha256": profile_hash(profile),
        "scope": "40_trial_DEVELOPMENT_repair_cycle_NOT_paper_test", "final_test_used": False,
        "oracle_fallback": False, "oracle_outcome_seed_filter": False, "video_seed": SEEDS[0],
        "user_requested_budget_revision_from200": bool(old),
        "adopted_prior_progress": str(args.adopt_from.resolve() / "progress.json") if old else None,
        "inflight_old_child_pid": args.inflight_pid, "episodes_rerolled": 0}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows, evidence = [], {}
    def verify():
        if any(sha256_file(Path(p)) != digest for p, digest in frozen.items()):
            raise ValueError("Controller/rules/runner changed during fixed40")
    def save(status, error=None):
        (output / "progress.json").write_text(json.dumps({**plan, "status": status, "pid": os.getpid(),
            "completed_trials": len(rows), "counts": dict(Counter(r["task"] for r in rows)),
            "by_task": {t: success_statistics([r for r in rows if r["task"] == t]) for t in TASKS
                        if any(r["task"] == t for r in rows)}, "result_hashes": evidence,
            "rows": rows, "error": error, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(ROOT / "VLA/checkpoints/huggingface"),
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}
    try:
        save("running")
        prior_seeds = {r["seed"] for r in old["rows"]} if old else set()
        for seed in SEEDS:
            verify()
            previous = ROOT / "VLA/smolvla" / f"{old['run_tag']}_seed{seed}_rp1" if old else None
            if old and previous.exists():
                result = previous / "results.json"
                if not result.exists():
                    if seed in prior_seeds or not args.inflight_pid:
                        raise ValueError("Unexpected incomplete old trial; no silent reroll")
                    argv_path = Path(f"/proc/{args.inflight_pid}/cmdline")
                    argv = argv_path.read_bytes().split(b"\0") if argv_path.exists() else []
                    if (b"VLA.benchmark" not in argv or str(seed).encode() not in argv
                            or previous.name.encode() not in argv):
                        raise ValueError("In-flight child no longer matches missing result")
                    deadline = time.monotonic() + 1800
                    while not result.exists() or not (previous / "trajectory.json").exists():
                        verify()
                        save("waiting_preserved_inflight_trial")
                        if (not argv_path.exists() or not argv_path.read_bytes()) and not result.exists():
                            raise RuntimeError("Preserved old child ended without results")
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Preserved child exceeded finite wait")
                        time.sleep(10)
                path = result
            else:
                directory = ROOT / "VLA/smolvla" / f"{args.run_tag}_seed{seed}_rp1"
                if directory.exists():
                    raise FileExistsError("Do not overwrite/reroll a new trial")
                command = [sys.executable, "-m", "VLA.benchmark", "--model", "smolvla", "--robot", "ur5e",
                    "--tasks", ",".join(TASKS), "--test-id", directory.name, "--checkpoint", str(cp),
                    "--seed", str(seed), "--scene-profile", "rl_random_env", "--replan-steps", "1",
                    "--control-hz", "10", "--required-success-steps", "1", "--success-mode", "semantic",
                    "--success-threshold-scale", str(template["success"]["reward_xml_threshold_scale"]),
                    "--audit-observation-every", "10"]
                for key, value in template["success"]["semantic_thresholds"].items():
                    command += ["--semantic-" + key.replace("_", "-"), str(value)]
                if seed != SEEDS[0]: command.append("--no-video")
                with (output / "logs" / f"seed{seed}.log").open("x") as stream:
                    done = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                if done.returncode:
                    raise RuntimeError(f"Integration failed on seed{seed}; no reroll")
                path = directory / "results.json"
            rows.extend(rows_from_result(path, seed, seed == SEEDS[0]))
            for p in (path, path.parent / "trajectory.json", path.parent / "manifest.json"):
                evidence[str(p)] = sha256_file(p)
            save("running")
            print(f"Completed {len(rows)}/40 neural trials", flush=True)
        verify_fingerprint(cp, full=True)
        verify()
        if Counter(r["task"] for r in rows) != {t: 10 for t in TASKS}:
            raise ValueError("Incomplete fixed40")
        save("completed40_development_trials")
    except Exception as exc:
        save("stopped_integration_error_no_reroll", str(exc))
        raise


if __name__ == "__main__":
    main()
