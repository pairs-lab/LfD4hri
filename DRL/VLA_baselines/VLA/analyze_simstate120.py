"""Post-rollout failure audit for the explicit privileged-input VLA variant.

Preserves causal-input/physics evidence. Signatures are NOT proven root causes
and no simulator-assisted result is presented as plain visual grounding.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.common.evaluation_statistics import success_statistics
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json

TASKS = ("touch", "pick", "move", "pick_place")


def audit_trace(row, trace):
    if not trace or len(trace) != row["steps"] or bool(trace[-1]["success"]) != row["success"]:
        raise ValueError("Completed trace/progress lengths or terminal outcome differ")
    flags, errors, offsets, low_offsets, held_steps = [], [], [], [], 0
    close_unheld, lift_unheld, wrong_hold, no_feedback_lift = 0, 0, 0, 0
    close_requests = 0
    for step in trace:
        before, after = step["interaction_before_audit"], step["interaction_after_audit"]
        state = np.asarray(step["pre_action_PRIVILEGED_state18"], dtype=np.float32)
        action = np.asarray(step["action"], dtype=np.float32)
        if state.shape != (18,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("Invalid actual learned input/action trace")
        pad = np.asarray(before["pad_grasp_center_xyz_m"], dtype=np.float32)
        target = np.asarray(before["object_center_xyz_m"], dtype=np.float32)
        goal = np.asarray(before["goal_xyz_m"], dtype=np.float32)
        base = np.asarray(before["object_base_xyz_m"], dtype=np.float32)
        expected = np.r_[pad, target-pad, goal-base]
        observed = np.r_[state[7:10], state[11:17]]
        error = float(np.abs(observed-expected).max())
        if error > 1e-6 or state[17] not in (0., 1.) or not 0 <= state[10] <= 1:
            raise ValueError("Deployed privileged input is not causal BEFORE-action physics")
        if state[17] and not before["correct_object_held"]:
            raise ValueError("Wrong held object advertised as verified target feedback")
        errors.append(error)
        delta = np.asarray(after["pad_grasp_center_xyz_m"]) - np.asarray(after["object_center_xyz_m"])
        offsets.append(float(np.linalg.norm(delta)))
        if abs(delta[2]) <= .06:
            low_offsets.append(float(np.linalg.norm(delta[:2])))
        correct, held = bool(after["correct_object_held"]), after["held_object"] is not None
        close = bool(action[6] > .2)  # actual canonical Robotiq command deadband
        close_requests += close
        held_steps += correct
        wrong_hold += held and not correct
        close_unheld += close and not held
        lift_unheld += close and not held and action[2] > .005
        no_feedback_lift += close and not state[17] and action[2] > .005
    if not row["success"]:
        if row.get("safety_violation"):
            flags.append("safety_abort")
        if row["task"] != "touch" and not held_steps:
            flags.append("never_held_requested_object")
        if row["task"] != "touch" and not close_requests:
            flags.append("no_close_request_in_grasp_task")
        if low_offsets and min(low_offsets) > .05:
            flags.append("low_approach_offset_over5cm")
        if wrong_hold:
            flags.append("held_wrong_object")
        if close_unheld:
            flags.append("close_requested_without_hold_after_step")
        if lift_unheld:
            flags.append("close_and_lift_without_hold_after_step")
        if held_steps and row["task"] in ("move", "pick_place"):
            flags.append("acquired_requested_object_but_destination_incomplete")
    return {**row, "failure_signatures": flags, "correct_hold_steps": int(held_steps),
        "canonical_close_request_threshold": .2,
        "wrong_hold_steps": int(wrong_hold), "close_requests": int(close_requests),
        "closing_lift_requests_with_pre_action_feedback_zero": int(no_feedback_lift),
        "min_pad_object_distance_m": min(offsets),
        "min_xy_error_at_low_approach_m": min(low_offsets) if low_offsets else None,
        "pre_action_privileged_pose_max_abs_error": max(errors)}


def analyze(progress_path, output):
    progress_path, output = Path(progress_path), Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior failure audit")
    progress = json.loads(progress_path.read_text())
    rows = progress["rows"]
    if (progress["status"] != "completed120_privileged_input_VLA" or len(rows) != 120
            or Counter(r["task"] for r in rows) != {t: 30 for t in TASKS}
            or len({(r["seed"], r["task"]) for r in rows}) != 120):
        raise ValueError("Do not turn incomplete/evaluation-error trials into SR120")
    reports, evidence, traces = [], {}, {}
    for row in rows:
        path = Path(row["trajectory"])
        if path not in traces:
            traces[path] = json.loads(path.read_text())
            evidence[str(path)] = sha256_file(path)
        reports.append(audit_trace(row, traces[path][row["task"]]))
    successes = sum(r["success"] for r in reports)
    videos = sorted({str(Path(r["trajectory"]).parent / "rollout.mp4") for r in rows
                     if (Path(r["trajectory"]).parent / "rollout.mp4").is_file()})
    report = {"model": "SmolVLA+SimState", "privileged_model_inputs": True,
        "NOT_plain_VLA_or_autonomous_visual_grounding": True,
        "complete120": True, "trials": 120, "successes": successes, "success_rate": successes/120,
        "statistics": success_statistics(rows), "rows": reports,
        "failure_signatures_overlap": dict(Counter(flag for row in reports for flag in row["failure_signatures"])),
        "progress_snapshot_sha256": sha256_file(progress_path), "trace_evidence": evidence,
        "videos": videos, "all_video_frames_visually_reviewed": False,
        "root_cause_not_established_by_signatures": True, "stable_high_success_established": False,
        "development_seeds_not_final_paper_test": True,
        "next_audit": ["Open actual raw model frames/video for low-approach, wrong-body and safety cases.",
            "Compare causal feedback and before/after motion; do not infer success from a close command.",
            "Collect separate TRAIN on-policy recovery corrections if lack of state/action support is confirmed.",
            "Do not mix these privileged-input scores with the six unprivileged policy rows."]}
    output.mkdir(parents=True)
    atomic_json(output / "analysis.json", report)
    lines = ["# SmolVLA+SimState: actual learned development120", "",
        "Privileged simulator inputs; NOT plain VLA or proof of visual localization.", "",
        f"Completed {successes}/120 ({100*successes/120:.1f}%). Safety aborts count as failures.", "",
        "| Action | Successes/trials | SR | 95% Wilson interval |", "|---|---|---|---|"]
    for task, stats in report["statistics"]["by_task"].items():
        low, high = stats["wilson_95_interval"]
        lines.append(f"| {task} | {stats['successes']}/{stats['trials']} | {100*stats['success_rate']:.1f}% | {100*low:.1f}–{100*high:.1f}% |")
    lines.extend(["", "Overlapping failure signatures (not established causes):",
                  "", json.dumps(report["failure_signatures_overlap"], ensure_ascii=False), "",
                  "Videos/raw input snapshots remain to be visually reviewed. No stable-high-SR claim."])
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.progress, args.output)


if __name__ == "__main__":
    main()
