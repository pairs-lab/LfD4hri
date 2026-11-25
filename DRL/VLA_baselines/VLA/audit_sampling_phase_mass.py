"""Immutable diagnostic of the deployed sampler's acquisition/feedback mass.

Simulator masks and grasp flags are audit-only, never policy inputs. This
does not modify sampling, authorize training, or establish sufficient data.
Before the workflow materializes TRAIN, merged-source results are projections,
not the actual qualified training distribution.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint, verify_source_payload_fingerprint
from VLA.smolvla.balanced_sampling import frame_weights
from VLA.audit_training_readiness import physical_scene_key


def phase_masks(arrays):
    action = np.asarray(arrays["canonical_actions"])
    n = len(action)
    closure = np.asarray(arrays["gripper_closure"])
    held_after = np.asarray(arrays["next_grasped_object"])
    verified_after = np.asarray(arrays["next_verified_grasp"])
    if (n < 2 or action.shape != (n, 7) or closure.shape != (n,)
            or held_after.shape != (n,) or verified_after.shape != (n,)
            or held_after.dtype != bool or verified_after.dtype != bool
            or not np.isfinite(action).all() or not np.isfinite(closure).all()
            or np.any((closure < 0) | (closure > 1))):
        raise ValueError("Invalid actual action/feedback arrays")
    # Dataset ordinary/closed-empty reset contracts do not start held. Require
    # that separately at source audit; this diagnostic is not for held starts.
    held_before = np.r_[False, held_after[:-1]]
    close = action[:, 6] > .5
    target_unheld_both = ~held_before & ~held_after
    reset = np.zeros(n, bool)
    reset[0] = True
    return {
        "reset": reset,
        "target_unheld_xy_alignment_over_3mm": ~held_before & (np.linalg.norm(action[:, :2], axis=1) > .003),
        "target_unheld_vertical_descent_over_3mm": ~held_before & (action[:, 2] < -.003),
        "target_unheld_pure_vertical_over_3mm": ~held_before & (np.abs(action[:, 2]) > .003)
            & (np.linalg.norm(action[:, :2], axis=1) <= .001),
        "close_acquires_correct_object": close & ~held_before & held_after,
        "correct_object_held_before": held_before,
        "closed_target_unheld_feedback": (closure > .8) & ~held_before,
        "open_while_closed_target_unheld": (closure > .8) & ~held_before & (action[:, 6] < -.5),
        "target_unheld_close_lift_over_5mm": close & target_unheld_both & (action[:, 2] > .005),
        "correct_held_to_unheld": held_before & ~held_after,
    }


def weighted_metrics(masks, weights):
    weights = np.asarray(weights)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid sampler probabilities")
    result = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.shape != weights.shape or mask.dtype != bool:
            raise ValueError("Mask and sampler probabilities are not aligned")
        result[name] = {"frames": int(mask.sum()), "probability_mass": float(weights[mask].sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-probability", type=float, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads((source/"index.json").read_text())
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    if (not rows or manifest.get("episodes") != len(rows)
            or manifest.get("index_sha256") != sha256_file(source/"index.json")
            or len({r["file"] for r in rows}) != len(rows)):
        raise ValueError("Requires complete immutable source")
    state = source/"collection_state.json"
    if state.exists() and not json.loads(state.read_text())["status"].startswith("complete"):
        raise ValueError("Do not project a moving collection prefix")
    bound = {str(p): sha256_file(p) for p in (source/"index.json", source/"dataset_manifest.json",
        Path(__file__), Path(__file__).parent/"smolvla/balanced_sampling.py")}
    fingerprint = source_payload_fingerprint(source, rows)
    weights = frame_weights(rows, args.initial_probability, target_balanced=True, grounding_probability=0.)
    totals, by_task, groups = {}, {}, set()
    offset = 0
    for number, row in enumerate(rows):
        with np.load(source/row["file"], allow_pickle=False) as arrays:
            masks = phase_masks(arrays)
        n = len(masks["reset"])
        if row["steps"] != n:
            raise ValueError("Source index and actual numeric transitions differ")
        metrics = weighted_metrics(masks, weights[offset:offset+n])
        task = row["task"]["name"]
        if task not in by_task:
            by_task[task] = {"episodes": 0, "frames": 0, "phase_metrics": {}}
        by_task[task]["episodes"] += 1
        by_task[task]["frames"] += n
        for name, value in metrics.items():
            for destination in (totals, by_task[task]["phase_metrics"]):
                target = destination.setdefault(name, {"frames": 0, "probability_mass": 0., "episodes_with_phase": 0})
                target["frames"] += value["frames"]
                target["probability_mass"] += value["probability_mass"]
                target["episodes_with_phase"] += bool(value["frames"])
        groups.add(physical_scene_key(row["task"]))
        offset += n
        if (number+1) % 200 == 0:
            print(f"sampler numeric audit {number+1}/{len(rows)}", flush=True)
    if offset != len(weights):
        raise ValueError("Source offsets differ from sampler")
    verify_source_payload_fingerprint(fingerprint, source, rows)
    if any(sha256_file(Path(p)) != h for p, h in bound.items()):
        raise RuntimeError("Source or diagnostic code changed")
    report = {"scope": "Exact target-balanced sampler projection on this source; NOT training qualification or learned SR",
        "source": str(source), "bound_files": bound, "source_payload_fingerprint": fingerprint,
        "episodes": len(rows), "frames": offset, "physical_layouts": len(groups),
        "initial_probability": args.initial_probability, "target_balanced": True,
        "paired_grounding_probability": 0., "sampling_with_replacement": True,
        "probability_sum": float(weights.sum()), "phase_metrics": totals, "by_task": by_task,
        "phase_metrics_overlap_not_a_partition": True,
        "combined_source_is_NOT_qualified_TRAIN": True,
        "held_start_assumption": "ordinary or physically closed-empty resets; verify at separate source audit",
        "target_unheld_is_NOT_proof_of_empty_jaws": True,
        "actual_distractor_hold_ids_not_stored_in_source": True,
        "training_recipe_changed": False, "policy_inputs_changed": False,
        "data_sufficiency_established": False, "final_test_used": False}
    output.mkdir(parents=True)
    (output/"audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_payload_fingerprint", "by_task")}, indent=2))


if __name__ == "__main__":
    main()
