"""Check target visibility when ordinary staged demos first require XY choice.

Audit-only simulator masks; never model input, sampler changes or trial filtering.
Counts are projected visibility, not token recognition/localization accuracy.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file


def summary(rows):
    return {"episodes": len(rows),
        "first_xy_frame_quantiles": np.quantile([r["first_xy_frame"] for r in rows], [0, .5, .9, 1]).tolist(),
        "handoff_front_fully_occluded": sum(r["handoff_pixels"]["front"] == 0 for r in rows),
        "handoff_wrist_fully_occluded": sum(r["handoff_pixels"]["wrist"] == 0 for r in rows),
        "handoff_both_views_fully_occluded": sum(max(r["handoff_pixels"].values()) == 0 for r in rows),
        "handoff_best_view_below64_pixels": sum(max(r["handoff_pixels"].values()) < 64 for r in rows),
        "initial_best_view_at_least64_but_handoff_below64": sum(max(r["initial_pixels"].values()) >= 64 and max(r["handoff_pixels"].values()) < 64 for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve earlier diagnostics")
    paths = [source / "index.json", source / "dataset_manifest.json", Path(__file__).resolve()]
    bound = {str(p): sha256_file(p) for p in paths}
    index = json.loads((source / "index.json").read_text())
    rows, no_xy = [], []
    for item in index:
        if item.get("recovery_start_actions_not_training_labels"):
            continue
        path = source / item["file"]
        bound[str(path)] = sha256_file(path)
        with np.load(path, allow_pickle=False) as archive:
            actions = archive["canonical_actions"]
            frames = np.flatnonzero(np.linalg.norm(actions[:, :2], axis=1) > .003)
            if not len(frames):
                no_xy.append(item["file"])
                continue
            first = int(frames[0])
            counts = {}
            for camera in ("front", "wrist"):
                mask = archive[camera+"_pick_mask"]
                if mask.dtype != bool or mask.ndim != 3 or len(mask) != len(actions):
                    raise ValueError("Invalid audit mask shape/dtype")
                counts[camera] = [int(mask[0].sum()), int(mask[first].sum())]
            rows.append({"file": item["file"], "task": item["task"]["name"], "pick_name": item["task"]["pick_name"],
                "first_xy_frame": first, "paired_copy_not_independent_layout": bool(item.get("paired_scene_source")),
                "initial_pixels": {c: v[0] for c, v in counts.items()},
                "handoff_pixels": {c: v[1] for c, v in counts.items()}})
        if len(rows) % 200 == 0:
            print(f"handoff visibility {len(rows)}", flush=True)
    if not rows or any(sha256_file(Path(p)) != digest for p, digest in bound.items()):
        raise ValueError("No measured rows or bound source changed")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    report = {"scope": "Ordinary teacher initial-to-first-XY visibility, NOT neural recognition or qualified performance",
        "source": str(source), "source_hashes": bound, "overall": summary(rows),
        "by_task": {task: summary(items) for task, items in grouped.items()}, "no_xy_motion_episodes": no_xy,
        "pixel64_is_diagnostic_NOT_new_gate": True, "camera_or_recipe_changed": False,
        "masks_are_audit_only": True, "paired_copies_counted_not_independent_scenes": True, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "source_hashes", "no_xy_motion_episodes"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
