"""Audit committed numeric labels and feedback without certifying a live source.

All committed rows are included. Simulator grasp flags are audit-only evidence,
never added to model inputs. Suspicious motions are diagnostic candidates, not
automatically failures: a contact-free retreat can correctly move an empty jaw.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.audit_training_readiness import physical_scene_key


def stable_manifest(manifest, live_collection):
    # The live collector legitimately rewrites ONLY these progress fields.
    # Robot/camera/physics/contracts and collection-plan hash remain pinned.
    return {key: value for key, value in manifest.items()
            if not live_collection or key not in {"episodes", "frames", "declared_task_counts", "index_sha256"}}


def episode_support(arrays, task):
    action = np.asarray(arrays["canonical_actions"])
    n = len(action)
    names = ("gripper_closure", "next_gripper_closure", "next_requested_gripper_closure",
             "next_grasped_object", "next_verified_grasp", "joint_positions",
             "next_observed_joint_positions")
    errors = []
    if action.shape != (n, 7) or n < 2 or not np.isfinite(action).all():
        raise ValueError("Invalid canonical actions")
    values = {name: np.asarray(arrays[name]) for name in names}
    for name, value in values.items():
        if len(value) != n or not np.isfinite(value).all():
            raise ValueError("Invalid feedback length/values: " + name)
    closure = values["gripper_closure"].reshape(n)
    next_closure = values["next_gripper_closure"].reshape(n)
    requested = values["next_requested_gripper_closure"].reshape(n)
    expected = (np.clip(action[:, 6], -1, 1) + 1) / 2
    command_error = float(np.max(np.abs(expected - requested)))
    closure_error = float(np.max(np.abs(next_closure[:-1] - closure[1:])))
    joint_error = float(np.max(np.abs(values["next_observed_joint_positions"][:-1]
                                     - values["joint_positions"][1:])))
    if command_error > 1e-6:
        errors.append("requested_gripper_label_does_not_match_canonical_command")
    if closure_error > 1e-6 or joint_error > 1e-6:
        errors.append("adjacent_measured_feedback_time_mismatch")
    if np.any((closure < 0) | (closure > 1)) or np.any((requested < 0) | (requested > 1)):
        errors.append("gripper_feedback_or_command_outside_normalized_range")
    held_after = values["next_grasped_object"].astype(bool).reshape(n)
    verified_after = values["next_verified_grasp"].astype(bool).reshape(n)
    held_before = np.r_[False, held_after[:-1]]
    verified_before = np.r_[False, verified_after[:-1]]
    # A live source's startup observation can itself be held; the audit must
    # not invent that fact. Declared recovery starts here are closed-empty.
    empty_both = ~held_before & ~held_after
    close = action[:, 6] > .5
    open_jaw = action[:, 6] < -.5
    reopen = (closure > .8) & open_jaw & ~held_before
    close_empty_lift = close & empty_both & (action[:, 2] > .005)
    close_empty_xy = close & empty_both & (np.linalg.norm(action[:, :2], axis=1) > .005)
    # Do not treat a transition into a successful grasp as unverified carry.
    unverified_carry = close & ~verified_before & ~verified_after & (action[:, 2] > .005)
    drops = held_before & ~held_after
    held_closure = closure[verified_before]
    empty_closure = closure[(closure > .95) & ~held_before]
    return {
        "frames": n, "errors": errors,
        "requested_gripper_command_max_error": command_error,
        "adjacent_measured_closure_max_error": closure_error,
        "adjacent_measured_joint_max_error": joint_error,
        "first_verified_grasp_frame": int(np.flatnonzero(verified_after)[0]) if verified_after.any() else None,
        "verified_grasp_frames": int(verified_after.sum()),
        "verified_before_measured_closure_range": [float(held_closure.min()), float(held_closure.max())] if len(held_closure) else None,
        "closed_empty_before_measured_closure_range": [float(empty_closure.min()), float(empty_closure.max())] if len(empty_closure) else None,
        "empty_closed_measured_frames": int(((closure > .8) & ~held_before).sum()),
        "reopen_empty_frames": int(reopen.sum()),
        "close_empty_lift_candidate_frames": np.flatnonzero(close_empty_lift).tolist(),
        "close_empty_xy_candidate_frames": np.flatnonzero(close_empty_xy).tolist(),
        "unverified_close_lift_candidate_frames": np.flatnonzero(unverified_carry).tolist(),
        "held_to_unheld_transitions": np.flatnonzero(drops).tolist(),
        "max_downward_command_m": float(max(0, -action[:, 2].min())),
        "fast_diagonal_descent_frames": int(((action[:, 2] < -.020001)
            & (np.linalg.norm(action[:, :2], axis=1) > .005)).sum()),
        "first_action_xy_m": float(np.linalg.norm(action[0, :2])),
        "first_action_z_m": float(action[0, 2]),
        "requested_vs_measured_closure_max_gap": float(np.max(np.abs(requested - next_closure))),
        "non_touch_never_verified": task != "touch" and not bool(verified_after.any()),
    }


def audit(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve diagnostics; choose a fresh output")
    index_bytes = (source / "index.json").read_bytes()
    rows = json.loads(index_bytes)
    if len({row["file"] for row in rows}) != len(rows):
        raise ValueError("Duplicate committed payload")
    manifest_bytes = (source / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    state_path = source / "collection_state.json"
    initial_state = json.loads(state_path.read_text()) if state_path.exists() else None
    live_collection = initial_state is not None and initial_state["status"] == "in_progress"
    script_hash = sha256_file(Path(__file__))
    output.mkdir(parents=True)
    (output / "index_snapshot.json").write_bytes(index_bytes)
    (output / "manifest_snapshot.json").write_bytes(manifest_bytes)
    results = []
    groups = defaultdict(set)
    targets = defaultdict(lambda: defaultdict(set))
    closure_ranges = defaultdict(list)
    sums = defaultdict(Counter)
    for number, row in enumerate(rows):
        task = row["task"]["name"]
        with np.load(source / row["file"], allow_pickle=False) as arrays:
            result = episode_support(arrays, task)
        result.update(file=row["file"], task=task, prompt=row["task"]["prompt"])
        results.append(result)
        group = physical_scene_key(row["task"])
        groups[task].add(group)
        targets[task][row["task"]["pick_name"]].add(group)
        if result["verified_before_measured_closure_range"]:
            closure_ranges[row["task"]["pick_name"]].append(result["verified_before_measured_closure_range"])
        count = sums[task]
        count["episodes"] += 1
        for field in ("frames", "verified_grasp_frames", "empty_closed_measured_frames",
                      "reopen_empty_frames", "fast_diagonal_descent_frames"):
            count[field] += result[field]
        for field in ("close_empty_lift_candidate_frames", "close_empty_xy_candidate_frames",
                      "unverified_close_lift_candidate_frames", "held_to_unheld_transitions"):
            count[field + "_count"] += len(result[field])
            count["episodes_with_" + field] += bool(result[field])
        count["non_touch_never_verified_episodes"] += result["non_touch_never_verified"]
        count["contract_error_episodes"] += bool(result["errors"])
        if (number + 1) % 100 == 0:
            print("numeric feedback rows", number + 1, "/", len(rows), flush=True)
    current_rows = json.loads((source / "index.json").read_bytes())
    current_manifest = json.loads((source / "dataset_manifest.json").read_bytes())
    if (current_rows[:len(rows)] != rows
            or stable_manifest(current_manifest, live_collection) != stable_manifest(manifest, live_collection)
            or sha256_file(Path(__file__)) != script_hash):
        raise RuntimeError("Committed prefix or manifest changed during audit")
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    report = {"scope": "ALL committed numeric feedback snapshot; NOT full data/physics/camera qualification",
        "source": str(source), "source_index_snapshot_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "source_manifest_sha256": manifest_hash, "snapshot_episodes": len(rows),
        "script_sha256": script_hash, "initial_collection_state": initial_state,
        "live_progress_fields_only_allowed_to_change": live_collection,
        "state_at_finish": state, "training_authorized": False, "weights_loaded": False,
        "policy_inputs_changed": False, "final_test_used": False,
        "contract_checks_passed": not any(row["errors"] for row in results),
        "candidate_motions_are_NOT_automatically_invalid": True,
        "no_drop_transitions_does_NOT_certify_drop_recovery": True,
        "verified_hold_measured_closure_range_per_object": {name: [min(value[0] for value in values), max(value[1] for value in values)]
            for name, values in closure_ranges.items()},
        "by_task": {task: {**dict(value), "physical_layouts": len(groups[task]),
            "layouts_per_requested_target": {name: len(ids) for name, ids in targets[task].items()}}
            for task, value in sums.items()}, "rows": results}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not audit(args.source, args.output)["contract_checks_passed"]:
        raise RuntimeError("Numeric feedback contracts failed; inspect preserved report")


if __name__ == "__main__":
    main()
