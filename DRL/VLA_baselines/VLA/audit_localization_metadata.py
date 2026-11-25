"""Spawn-layout/text coverage diagnostics; no pixels or success labels examined.

Grid occupancy is NOT a data-sufficiency or model-localization certificate.
Recovery copies and alternate instructions are not independent physical scenes.
This never changes a split, scene, observation, demonstration, or fit recipe.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from config import BOUNDS, PICK_TARGETS
from VLA.audit_training_readiness import physical_scene_key
from VLA.common.provenance import sha256_file
from VLA.common.spatial import FIXED_DESTINATION


def spatial_cell(position, bins=5):
    xy = np.asarray(position, dtype=float)[:2]
    bounds = np.asarray(BOUNDS, dtype=float)[:2]
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Invalid spawn coordinates")
    if ((xy < bounds[:, 0] - 1e-7) | (xy > bounds[:, 1] + 1e-7)).any():
        raise ValueError("Target spawn outside original workspace")
    scaled = (xy - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return tuple(np.minimum((scaled * bins).astype(int), bins - 1).tolist())


def summarize(rows):
    if not rows:
        raise ValueError("Empty source")
    groups, tasks = defaultdict(set), defaultdict(list)
    for row in rows:
        task = row["task"]
        group = physical_scene_key(task)
        if row.get("scene_group", group) != group:
            raise ValueError("Stored physical layout identity differs")
        spatial_cell(task["pick_position"])
        groups[group].add(task["pick_name"])
        tasks[task["name"]].append((row, group))
    results = {}
    for name, members in sorted(tasks.items()):
        target_groups, pair_groups = defaultdict(dict), defaultdict(set)
        same_color_scenes, close_neighbor_scenes = set(), set()
        recovery = Counter()
        for row, group in members:
            task = row["task"]
            target_groups[task["pick_name"]][group] = task["pick_position"]
            destination = task.get("place_name")
            if destination:
                pair_groups[(task["pick_name"], destination)].add(group)
            others = list(task["scene_objects"])
            if destination and destination not in FIXED_DESTINATION:
                others.append({"name": destination, "position": task["place_position"]})
            color = task["pick_name"].split(" ", 1)[0]
            if any(item["name"].split(" ", 1)[0] == color for item in others):
                same_color_scenes.add((group, task["pick_name"]))
            origin = np.asarray(task["pick_position"], dtype=float)[:2]
            if any(np.linalg.norm(np.asarray(item["position"], dtype=float)[:2] - origin) < .06
                   for item in others):
                close_neighbor_scenes.add((group, task["pick_name"]))
            if row.get("recovery_start_actions_not_training_labels"):
                recovery[row.get("teacher_policy", "unspecified")] += 1
        coverage = {}
        for target, positions in sorted(target_groups.items()):
            xy = np.asarray(list(positions.values()), dtype=float)[:, :2]
            cells = Counter(spatial_cell(position) for position in positions.values())
            coverage[target] = {
                "independent_layouts": len(positions),
                "occupied_5x5_spawn_cells": len(cells),
                "spawn_xy_min_m": xy.min(axis=0).tolist(),
                "spawn_xy_max_m": xy.max(axis=0).tolist(),
                "cell_counts": {f"{x},{y}": count for (x, y), count in sorted(cells.items())},
            }
        results[name] = {
            "episodes": len(members),
            "physical_layouts": len({group for _, group in members}),
            "missing_requested_object_types": sorted(set(PICK_TARGETS) - set(target_groups)),
            "target_spawn_coverage": coverage,
            "labelled_layouts_with_same_color_distractor": len(same_color_scenes),
            "labelled_layouts_with_neighbor_center_under_6cm": len(close_neighbor_scenes),
            "destination_pairs": len(pair_groups),
            "destination_pairs_only_one_physical_layout": sum(len(values) == 1 for values in pair_groups.values()),
            "recovery_episodes_by_teacher": dict(recovery),
        }
    return {
        "episodes": len(rows), "physical_layouts": len(groups),
        "layouts_with_multiple_requested_objects": sum(len(values) > 1 for values in groups.values()),
        "tasks": results, "data_sufficiency_established": False,
        "scope": "spawn metadata only; NOT image visibility, physics qualification, or learned SR",
        "final_test_used": False,
        "caveats": [
            "Spawn positions differ from settled/grasp-time positions for movable objects.",
            "No benchmark seeds or model outcomes are used to select data or splits.",
            "Five-by-five occupancy does not imply continuous spatial generalization.",
            "A six-centimeter center separation is a diagnostic, not a collision or grasp-failure label.",
            "Same-color distractors do not certify semantic visual disambiguation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = args.source / "index.json"
    digest = sha256_file(index)
    report = summarize(json.loads(index.read_text()))
    if sha256_file(index) != digest:
        raise RuntimeError("Source index changed during audit")
    report.update(source=str(args.source.resolve()), source_index_sha256=digest,
                  audit_script_sha256=sha256_file(Path(__file__)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
