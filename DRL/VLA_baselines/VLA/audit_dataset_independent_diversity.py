"""Independent layout/type coverage and intentional action-rotation diversity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = json.loads((args.source / "index.json").read_text())
    results = {}
    nonzero_rotation, frames = 0, 0
    for name in ("touch", "pick", "move", "pick_place"):
        selected = [r for r in rows if r["task"]["name"] == name]
        picks, goals = defaultdict(set), defaultdict(set)
        for row in selected:
            group = physical_scene_key(row["task"])
            picks[row["task"]["pick_name"]].add(group)
            if row["task"].get("place_name"):
                goals[row["task"]["place_name"]].add(group)
            with np.load(args.source / row["file"]) as data:
                action = data["canonical_actions"]
                frames += len(action)
                nonzero_rotation += int((np.abs(action[:, 3:6]) > 1e-7).any(axis=1).sum())
        results[name] = {"episodes": len(selected), "independent_layouts": len({physical_scene_key(r["task"]) for r in selected}),
            "independent_layouts_by_picked_type": {k: len(v) for k, v in picks.items()},
            "independent_layouts_by_destination_type": {k: len(v) for k, v in goals.items()},
            "independent_labelled_pick_destination_pairs": len({(physical_scene_key(r["task"]), r["task"]["pick_name"], r["task"].get("place_name")) for r in selected})}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source / "index.json"),
        "tasks": results, "frames": frames, "nonzero_rotation_action_frames": nonzero_rotation,
        "scope": "Data sufficiency diagnostic, NOT an overfitting proof or performance estimate", "final_test_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
