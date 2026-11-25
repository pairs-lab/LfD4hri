"""Measure where paired text interventions actually change action supervision.

CPU diagnostic only: does not alter the running sampler, data or policy inputs.
Same-image t=0 with identical early actions is not immediate localization
supervision. Later divergent labels may still supply useful grounding signal.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.smolvla.balanced_sampling import frame_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-probability", type=float, default=.1)
    parser.add_argument("--action-horizon", type=int, default=5)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or args.action_horizon < 1:
        raise ValueError("Fresh report and positive horizon required")
    paths = [source / "index.json", source / "dataset_manifest.json", Path(__file__).resolve(), Path(__file__).parent / "smolvla/balanced_sampling.py"]
    bound = {str(p): sha256_file(p) for p in paths}
    index = json.loads((source / "index.json").read_text())
    weights = frame_weights(index, args.initial_probability, target_balanced=True, grounding_probability=0.)
    groups, offsets = defaultdict(list), {}
    start = 0
    for item in index:
        offsets[item["file"]] = start
        start += item["steps"]
        if item.get("paired_scene_source") and item.get("recovery_start_kind") != "mid_height_contact_free_closed_empty_v1":
            key = (item["canonical_source"], item["paired_scene_source"], item["task"]["name"], json.dumps(item.get("recovery_start_actions_not_training_labels", [])))
            groups[key].append(item)
    if start != len(weights):
        raise ValueError("Sampler offsets differ from index")
    results = []
    for items in groups.values():
        if len(items) != 2 or items[0]["task"]["prompt"] == items[1]["task"]["prompt"]:
            raise ValueError("Actual text pair is incomplete or duplicated")
        files = [source / row["file"] for row in items]
        for path in files:
            bound[str(path)] = sha256_file(path)
        data = []
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                data.append({key: archive[key] for key in ("canonical_actions", "joint_positions", "gripper_closure")})
        n = min(len(d["canonical_actions"]) for d in data)
        differences = np.max(np.abs(data[0]["canonical_actions"][:n] - data[1]["canonical_actions"][:n]), axis=1)
        exact_differences = np.flatnonzero(differences > 0)
        substantive = np.flatnonzero(differences > .001)
        if not len(substantive):
            raise ValueError("Pair lacks substantive overlapping-frame label differences")
        first = int(substantive[0])
        first_exact = int(exact_differences[0])
        # Horizon5 can carry a later differing label at an earlier sampled anchor.
        # These are label contrasts, not proof of visual/text recognition.
        signal = np.asarray([np.any(differences[i:min(n, i+args.action_horizon)] > .001) for i in range(n)])
        mass = sum(float(weights[offsets[row["file"]]:offsets[row["file"]]+n][signal].sum()) for row in items)
        initial_mass = sum(float(weights[offsets[row["file"]]]) for row in items)
        results.append({"task": items[0]["task"]["name"], "kind": items[0].get("recovery_start_kind", "ordinary_target_pair"),
            "files": [r["file"] for r in items], "first_exact_action_difference_frame": first_exact,
            "first_substantive_action_difference_frame": first, "first_label_difference": float(differences[0]),
            "initial_horizon_has_substantive_contrast": bool(signal[0]),
            "first_contrast_joint_error_rad": float(np.max(np.abs(data[0]["joint_positions"][first] - data[1]["joint_positions"][first]))),
            "first_contrast_aperture_error": abs(float(data[0]["gripper_closure"][first] - data[1]["gripper_closure"][first])),
            "paired_initial_anchor_probability_mass": initial_mass, "paired_contrast_horizon_anchor_probability_mass": mass})
    if any(sha256_file(Path(p)) != digest for p, digest in bound.items()):
        raise ValueError("Bound data or sampler changed during diagnostic")
    report = {"scope": "Where text pairs supply distinct action-chunk labels, NOT localization or success evidence",
        "source": str(source), "source_hashes": bound, "action_horizon": args.action_horizon,
        "pairs": len(results), "pairs_with_first_label_contrast": sum(r["first_label_difference"] > .001 for r in results),
        "pairs_with_initial_horizon_contrast": sum(r["initial_horizon_has_substantive_contrast"] for r in results),
        "first_substantive_contrast_histogram": dict(Counter(r["first_substantive_action_difference_frame"] for r in results)),
        "paired_initial_anchor_probability_mass": sum(r["paired_initial_anchor_probability_mass"] for r in results),
        "paired_contrast_horizon_anchor_probability_mass": sum(r["paired_contrast_horizon_anchor_probability_mass"] for r in results),
        "training_recipe_changed": False, "policy_inputs_changed": False,
        "contrast_mass_is_NOT_total_grounding_signal": True, "rows": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"source_hashes", "rows"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
