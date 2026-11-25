"""Fixed10/action learned RDT trials on the SAME SmolVLA development protocol.

The caller owns a verified target-domain RDT server. No oracle seed filtering,
fallback, retries or runtime/data edits. Source diagnostic rollouts are separate.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import requests

from VLA.analyze_diagnostic50 import analyze_trial
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.run_diagnostic40 import SEEDS, TASKS

ROOT = Path(__file__).resolve().parents[1]


def result_rows(path, seed, video):
    result = json.loads(path.read_text())
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if (result.get("model") != "rdt" or manifest["seed"] != seed or manifest["replan_steps"] != 1
            or manifest.get("privileged_object_pose_given_to_vla") is not False
            or Counter(row["name"] for row in result["tasks"]) != {t: 1 for t in TASKS}
            or any(row.get("error") for row in result["tasks"])):
        raise ValueError("Incomplete, privileged or integration-failed trial; preserve, no reroll")
    return [{"seed": seed, "task": row["name"], "success": row["success"],
        "safety_violation": row.get("safety_violation"), "error": row.get("error"),
        "result": str(path), "trajectory": str(path.parent / "trajectory.json"),
        "video": str(path.parent / "rollout.mp4") if video else None} for row in result["tasks"]]


def analyze_requests(result, trace):
    # The body-controller requests CLOSE at >0.2, not >0.5. Change ONLY the
    # analysis copy to feed the historical signature helper; raw evidence and
    # actually executed actions remain unchanged. Deadband is not a new request.
    diagnostic = [{**row, "action": [*row["action"][:6], 1. if row["action"][6] > .2 else -1.]}
                  for row in trace]
    return analyze_trial(result, diagnostic)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.run_tag.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Fresh output and simple run tag required")
    cp, output = args.checkpoint.resolve(), args.output.resolve()
    identity = verify_fingerprint(cp, full=True)
    profile = checkpoint_profile(str(cp))
    template_path = ROOT / "VLA/evaluation_protocol_v9_postfit_10_per_action.json"
    template = json.loads(template_path.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]:
        raise ValueError("RDT and shared camera/controller protocol differ")
    reserved = json.loads((ROOT / template["reserved_final_protocol"]).read_text())
    if set(SEEDS) & set(reserved.get("final_test_seeds", [])):
        raise ValueError("Reserved final-test leakage")
    client = requests.Session()
    client.trust_env = False
    response = client.get(f"http://127.0.0.1:{args.port}/metadata", timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if (metadata.get("model") != "rdt" or metadata.get("target_domain") is not True
            or metadata.get("control_hz") != 10
            or Path(metadata["checkpoint"]["path"]).resolve() != cp
            or metadata["checkpoint"].get("content_tree_sha256") != identity["content_tree_sha256"]):
        raise ValueError("Server/checkpoint identity differs")
    runtime = environment_runtime_fingerprint(profile)
    paths = [Path(__file__).resolve(), template_path, ROOT / "VLA/benchmark.py",
             ROOT / "VLA/rdt/adapter.py", ROOT / "VLA/rdt/server.py", ROOT / "VLA/analyze_diagnostic50.py"]
    paths += [ROOT / name for name in (*runtime["sources_sha256"], "asset/params/reward.xml",
                                      "utils/random_env.py", "utils/camera.py")]
    fixed = {str(p): sha256_file(p) for p in paths}
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    plan = {"model": "rdt", "checkpoint": str(cp), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "server_metadata": metadata, "tasks": list(TASKS), "seeds": SEEDS, "planned_trials": 40,
        "trials_per_action": 10, "success": template["success"], "scope": "40_trial_DEVELOPMENT_NOT_paper_test",
        "final_test_used": False, "oracle_fallback": False, "oracle_outcome_seed_filter": False,
        "episodes_rerolled": 0, "frozen_files": fixed, "observation_profile_sha256": profile_hash(profile)}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows, hashes, analyses = [], {}, []
    def save(status, error=None):
        (output / "progress.json").write_text(json.dumps({**plan, "status": status,
            "completed_trials": len(rows), "counts": dict(Counter(r["task"] for r in rows)),
            "by_task": {t: success_statistics([r for r in rows if r["task"] == t]) for t in TASKS
                        if any(r["task"] == t for r in rows)}, "rows": rows, "result_hashes": hashes,
            "pid": os.getpid(), "error": error, "updated_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n")
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    try:
        save("running")
        for seed in SEEDS:
            if any(sha256_file(Path(p)) != h for p, h in fixed.items()):
                raise ValueError("Fixed runtime/protocol changed")
            directory = ROOT / "VLA/rdt" / f"{args.run_tag}_seed{seed}_rp1"
            if directory.exists():
                raise FileExistsError("Existing trial preserved; no reroll")
            tokens = [sys.executable, "-m", "VLA.benchmark", "--model", "rdt", "--robot", "ur5e",
                "--tasks", ",".join(TASKS), "--test-id", directory.name, "--checkpoint", str(cp), "--port", str(args.port),
                "--seed", str(seed), "--scene-profile", "rl_random_env", "--replan-steps", "1", "--control-hz", "10",
                "--required-success-steps", "1", "--success-mode", "semantic", "--success-threshold-scale",
                str(template["success"]["reward_xml_threshold_scale"]), "--audit-observation-every", "10"]
            for key, value in template["success"]["semantic_thresholds"].items():
                tokens += ["--semantic-" + key.replace("_", "-"), str(value)]
            if seed != SEEDS[0]: tokens.append("--no-video")
            with (output / "logs" / f"seed{seed}.log").open("x") as stream:
                done = subprocess.run(tokens, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            if done.returncode:
                raise RuntimeError("Integration failure; no silent reroll")
            path = directory / "results.json"
            rows.extend(result_rows(path, seed, seed == SEEDS[0]))
            for file in (path, path.parent / "manifest.json", path.parent / "trajectory.json"):
                hashes[str(file)] = sha256_file(file)
            result = json.loads(path.read_text())
            traces = json.loads((path.parent / "trajectory.json").read_text())
            analyses += [{"seed": seed, **analyze_requests(task, traces[task["name"]])} for task in result["tasks"]]
            save("running")
            print(f"RDT completed {len(rows)}/40 learned trials", flush=True)
        if any(sha256_file(Path(p)) != h for p, h in {**fixed, **hashes}.items()):
            raise ValueError("Immutable runtime/result evidence changed")
        verify_fingerprint(cp, full=True)
        if Counter(r["task"] for r in rows) != {t: 10 for t in TASKS}:
            raise ValueError("Incomplete declared40")
        report = {"scope": "40_trial_learned_RDT_DEVELOPMENT_NOT_final_test", "complete40": True,
            "successes": sum(r["success"] for r in rows), "statistics": success_statistics(rows),
            "failure_signatures_overlap": dict(Counter(flag for row in analyses for flag in row["failure_signatures"])),
            "rows": analyses, "close_request_threshold": .2, "raw_actions_modified": False,
            "data_sufficiency_established": False, "stable_success_established": False, "final_test_used": False}
        (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
        save("completed40_development_trials")
    except Exception as exc:
        save("stopped_integration_error_no_reroll", str(exc))
        raise


if __name__ == "__main__":
    main()
