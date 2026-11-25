"""Replay ALL recorded safety aborts offline; never new neural evaluations.

Preserve failed reproduction and continue auditing the other recorded cases.
Safety signatures describe the exact terminal state, not an intrinsic-model
causal conclusion. No action, environment, checkpoint or training data edits.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

from VLA.common.provenance import sha256_file


def summarize(report):
    if not report.get("passed_exact_recorded_replay"):
        raise ValueError("An exact recorded-action replay is required")
    last = report["feedback"][-1]
    diagnostics = last.get("pre_rollback_diagnostics") or {}
    contacts = diagnostics.get("contacts", [])
    # A near-zero force contact is not evidence of a damaging collision.
    loaded = [c for c in contacts if float(c.get("normal_force_n", 0)) > 5.]
    return {"task": report["task"], "steps": report["steps"],
        "replay_errors": report["maximum_replay_errors"],
        "safety_violation": last["safety_violation"],
        "diagnostics": diagnostics,
        "loaded_contact_body_names": sorted({c["other_body_name"] for c in loaded}),
        "requested_body_loaded_contact": any(c["is_requested_object"] for c in loaded),
        "distractor_loaded_contact": any(not c["is_requested_object"] and
            c["other_body_name"] != "robot_or_static_environment" for c in loaded)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve every audit attempt")
    progress = json.loads(args.progress.read_text())
    if progress["status"] != "completed40_development_trials" or len(progress["rows"]) != 40:
        raise ValueError("Complete declared40-trial cohort required")
    cases = [r for r in progress["rows"] if r.get("safety_violation")]
    fixed = {str(args.progress.resolve()): sha256_file(args.progress)}
    args.output.mkdir(parents=True)
    rows = []
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    def save():
        (args.output / "audit.json").write_text(json.dumps({
            "scope": "CPU_only_recorded_safety_reproduction_NOT_neural_success",
            "expected_cases": len(cases), "completed_cases": len(rows),
            "all_exact_reproduced": len(rows) == len(cases) and all(r.get("reproduced") for r in rows),
            "new_neural_trials": 0, "actions_or_guards_changed": False,
            "data_promoted": False, "source_hashes": fixed, "rows": rows,
            "loaded_contact_signatures_overlap": dict(Counter(key for row in rows for key in
                ("distractor_loaded_contact", "requested_body_loaded_contact") if row.get(key)))}, indent=2) + "\n")
    for case in cases:
        directory = args.output / f"seed{case['seed']}_{case['task']}"
        log = args.output / (directory.name + ".log")
        tokens = [sys.executable, "-m", "VLA.audit_benchmark_safety_replay", "--run",
            str(Path(case["result"]).parent), "--task", case["task"], "--output", str(directory)]
        with log.open("x") as stream:
            completed = subprocess.run(tokens, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row = {"seed": case["seed"], "task": case["task"], "exit_code": completed.returncode,
               "reproduced": completed.returncode == 0, "log": str(log.resolve())}
        if not completed.returncode:
            report = json.loads((directory / "audit.json").read_text())
            row.update(summarize(report))
            fixed[str((directory / "audit.json").resolve())] = sha256_file(directory / "audit.json")
            fixed.update(report["source_hashes"])
        rows.append(row)
        save()
        print(json.dumps({"case": directory.name, "reproduced": row["reproduced"]}), flush=True)
    if any(sha256_file(Path(p)) != digest for p, digest in fixed.items()):
        raise ValueError("Immutable recorded evidence changed")
    save()


if __name__ == "__main__":
    main()
