"""Count independent layout/instruction support, not raw frames or copies.

Heldout/development instructions are audited only; never converted into
training rows. Missing combinations need not imply failed generalization.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from config import PICK_TARGETS, PLACE_TARGETS
from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file


def summary(index):
    groups, target_groups, tuple_groups = defaultdict(set), defaultdict(set), defaultdict(set)
    frames, copies = Counter(), Counter()
    for row in index:
        task = row["task"]
        group = physical_scene_key(task)
        groups[task["name"]].add(group)
        target_groups[(task["name"],task["pick_name"])].add(group)
        tuple_groups[(task["name"],task["pick_name"],task.get("place_name"))].add(group)
        frames[task["name"]] += row["steps"]
        copies[task["name"]] += 1
    by_task = {}
    for task in ("touch","pick","move","pick_place"):
        counts = [len(target_groups[(task,target)]) for target in PICK_TARGETS]
        combos = [(target,goal) for target in PICK_TARGETS for goal in PLACE_TARGETS if target != goal] if task in {"move","pick_place"} else [(target,None) for target in PICK_TARGETS]
        covered = [len(tuple_groups[(task,target,goal)]) for target,goal in combos]
        by_task[task] = {"episodes_including_copies": copies[task], "transitions": frames[task], "unique_physical_layouts": len(groups[task]),
            "target_layout_support_min_median_max": np.quantile(counts,[0,.5,1]).tolist(),
            "instruction_tuple_vocabulary": len(combos), "covered_tuples": sum(x > 0 for x in covered),
            "tuple_layout_support_min_median_max": np.quantile(covered,[0,.5,1]).tolist(),
            "target_layout_counts": {target: len(target_groups[(task,target)]) for target in PICK_TARGETS}}
    return by_task, tuple_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--development-progress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Retain coverage snapshots")
    paths = [args.source / "index.json", args.validation / "index.json", args.development_progress, Path(__file__)]
    hashes = {str(p.resolve()): sha256_file(p) for p in paths}
    train, tuples = summary(json.loads(paths[0].read_text()))
    validation, _ = summary(json.loads(paths[1].read_text()))
    progress = json.loads(args.development_progress.read_text())
    checked = []
    for row in progress["rows"]:
        result = Path(row["result"])
        manifest = json.loads(result.with_name("manifest.json").read_text())
        task = next(t for t in manifest["tasks"] if t["name"] == row["task"])
        count = len(tuples[(task["name"],task["pick_name"],task.get("place_name"))])
        checked.append({"seed": row["seed"], "task": row["task"], "prompt": task["prompt"], "TRAIN_layouts_for_exact_instruction_tuple": count})
    if any(sha256_file(Path(p)) != digest for p,digest in hashes.items()): raise ValueError("Inputs changed")
    report = {"scope": "TRAIN layout/vocabulary support, NOT universal sufficiency or causal attribution", "source_hashes": hashes,
        "train": train, "validation": validation, "development_instructions": checked,
        "development_tuples_without_TRAIN_example": sum(row["TRAIN_layouts_for_exact_instruction_tuple"] == 0 for row in checked),
        "policy_inputs_or_dataset_changed": False, "final_test_used": False,
        "vocabulary_combinations_are_potential_instructions_NOT_required_memorized_scenes": True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"train": train, "development_tuples_without_TRAIN_example": report["development_tuples_without_TRAIN_example"]},indent=2))


if __name__ == "__main__":
    main()
