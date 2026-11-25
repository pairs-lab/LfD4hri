"""Count observed failure/reopen coverage, not just successful episode totals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

from VLA.common.provenance import sha256_file


def summarize(source):
    index = json.loads((source / "index.json").read_text())
    counts = Counter()
    by_task = {}
    for row in index:
        with np.load(source / row["file"]) as z:
            action, closure = z["canonical_actions"], z["gripper_closure"]
            current_correct_grasp = np.r_[False, z["next_grasped_object"].astype(bool)[:-1]]
            empty = (closure > .5) & ~current_correct_grasp
            row_counts = {"episodes": 1, "frames": len(action), "closed_empty_start_episodes": int(empty[0]),
                "closed_empty_observation_frames": int(empty.sum()),
                "closed_empty_labels_reopening": int((empty & (action[:, 6] < -.2)).sum()),
                "closed_empty_labels_remaining_closed_and_lifting": int((empty & (action[:, 6] > .2) & (action[:, 2] > .005)).sum())}
            counts.update(row_counts)
            by_task.setdefault(row["task"]["name"], Counter()).update(row_counts)
    return {"source": str(source.resolve()), "source_index_sha256": sha256_file(source / "index.json"),
        "counts": dict(counts), "by_task": {k: dict(v) for k, v in by_task.items()},
        "closed_empty_uniform_frame_fraction": counts["closed_empty_observation_frames"] / counts["frames"],
        "initial_grasp_assumed_false_scope": "These collections explicitly reject any grasp during startup",
        "failed_state_distribution_sufficiency": "NOT ESTABLISHED; free-space closure differs from failed near-object grasps"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reports = [summarize(source) for source in args.source]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scope": "Demonstration coverage diagnostic, NOT learned performance",
        "datasets": reports, "final_test_used": False, "simulator_grasp_labels_not_policy_inputs": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
