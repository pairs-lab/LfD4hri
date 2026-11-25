"""Analyze completed neural trial traces; privileged state stays audit-only.

Failure signatures describe observations, not proven model/setup root causes.
Partial snapshots are explicitly provisional and never select/omit seeds.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file


def analyze_trial(result, trace):
    if not trace:
        raise ValueError("Missing completed-trial trajectory")
    xyz = [step["interaction_after_audit_only"] for step in trace]
    if any(row.get("privileged_policy_input") is not False for row in xyz):
        raise ValueError("Audit interaction must not be policy input")
    # Canonical Robotiq control closes at >0.2, not >0.5. A neural request
    # such as0.3 really closes the hand and must not be called "no close".
    # This changes diagnostic signatures ONLY, never success/scoring/control.
    close = np.array([step["action"][6] > .2 for step in trace])
    correct = np.array([row["correct_object_held"] for row in xyz])
    held = np.array([row["held_object"] is not None for row in xyz])
    separation = np.array([np.asarray(row["pad_grasp_center_xyz_m"]) -
                           np.asarray(row["object_center_xyz_m"]) for row in xyz])
    xy = np.linalg.norm(separation[:, :2], axis=1)
    # This is a camera/controller localization diagnostic, not a grasp-success
    # threshold: actual safe body height and object shape still matter.
    low = np.abs(separation[:, 2]) <= .06
    flags = []
    if not result["success"]:
        if result.get("safety_violation"):
            flags.append("safety_abort")
        if result["name"] != "touch" and not close.any():
            flags.append("no_close_request_in_grasp_task")
        if result["name"] != "touch" and not correct.any():
            flags.append("never_held_requested_object")
        if low.any() and xy[low].min() > .05:
            flags.append("low_approach_offset_over5cm")
        if np.any(held & ~correct):
            flags.append("held_wrong_object")
        if np.any(close & ~held):
            flags.append("close_requested_without_hold_after_step")
        if np.any(close & ~held & (np.array([step["action"][2] for step in trace]) > .005)):
            flags.append("close_and_lift_without_hold_after_step")
        if correct.any() and result["name"] in ("move", "pick_place"):
            flags.append("acquired_requested_object_but_destination_incomplete")
    closest = int(np.argmin(np.linalg.norm(separation, axis=1)))
    return {"task": result["name"], "prompt": result["prompt"], "success": result["success"],
        "steps": result["steps"], "failure_signatures": flags, "close_requests": int(close.sum()),
        "canonical_close_request_threshold": .2,
        "correct_hold_steps": int(correct.sum()), "wrong_hold_steps": int((held & ~correct).sum()),
        "min_pad_object_center_distance_m": float(np.linalg.norm(separation, axis=1).min()),
        "min_xy_error_at_low_approach_m": float(xy[low].min()) if low.any() else None,
        "nearest_pad_step": int(trace[closest]["step"]),
        "safety_violation": result.get("safety_violation"),
        "initial_semantic_success": result.get("initial_semantic_success")}


def analyze(progress_path, output, require_complete=False):
    progress_path, output = Path(progress_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve each analysis snapshot")
    progress = json.loads(progress_path.read_text())
    rows = progress["rows"]
    counts = Counter(row["task"] for row in rows)
    complete = (progress["status"] == "completed200_development_trials"
                and len(rows) == 200 and counts == {t: 50 for t in progress["tasks"]})
    if require_complete and not complete:
        raise ValueError("The50/action experiment is not complete")
    if not rows or len({(r["seed"], r["task"]) for r in rows}) != len(rows):
        raise ValueError("No completed trials or duplicate seed/task")
    reports, evidence, summaries = {}, {}, []
    for row in rows:
        path = Path(row["result"])
        trajectory = Path(row["trajectory"])
        if path not in reports:
            reports[path] = (json.loads(path.read_text()), json.loads(trajectory.read_text()))
            evidence[str(path)] = sha256_file(path)
            evidence[str(trajectory)] = sha256_file(trajectory)
        result, traces = reports[path]
        task = next(t for t in result["tasks"] if t["name"] == row["task"])
        if task["success"] != row["success"] or task.get("error"):
            raise ValueError("Progress disagrees with result or contains an integration error")
        summaries.append({"seed": row["seed"], **analyze_trial(task, traces[row["task"]])})
    report = {"scope": "DEVELOPMENT_error_discovery_NOT_final_paper_test", "complete50_per_action": complete,
        "provisional": not complete, "completed_trials": len(rows), "counts": dict(counts),
        "checkpoint_content_sha256": progress["checkpoint_content_sha256"],
        "progress_snapshot_sha256": sha256_file(progress_path), "evidence": evidence,
        "statistics": success_statistics(summaries), "rows": summaries,
        "failure_signature_counts_overlap": dict(Counter(flag for row in summaries for flag in row["failure_signatures"])),
        "videos": sorted({r["video"] for r in rows if r.get("video")}),
        "all_video_frames_visually_reviewed": False,
        "root_cause_not_established_by_signature_alone": True,
        "no_privileged_model_input_or_oracle_fallback": True,
        "repair_plan": [
            "Keep baseline checkpoint, scoring and controller fixed for all200 trials; do not reroll failures.",
            "Use separately recollected r12 data with exhaustive canonical/native strict replay; old checkpoint did NOT train on it.",
            "Fix readiness singleton-recovery classification; preserve real counterfactual-pair and measured feedback checks.",
            "Audit qualified TRAIN close-acquisition and recovery probability mass before optimization; do not fit on these development evaluation rows.",
            "Train fresh pretrained base with adapted visual connector/language layers and FP32 optimizer; preserve scene-disjoint validation.",
            "Re-evaluate the fixed new checkpoint on the same200 DEVELOPMENT tasks to measure change; final benchmark uses untouched reserved seeds.",
            "Review linkage safety and low approach offsets against successful native replay; do not disable rollback to inflate success."],
        "new_training_success_guaranteed": False, "final_test_used": False}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("completed_trials", "provisional", "failure_signature_counts_overlap")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    analyze(args.progress, args.output, args.require_complete)


if __name__ == "__main__":
    main()
