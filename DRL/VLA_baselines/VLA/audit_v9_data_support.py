"""V9 TRAIN quality/quantity support, with measured-feedback semantics.

Coverage counters are diagnostics, not a declaration that data is sufficient.
In particular a close-before-lift counter cannot alone diagnose failed grasps.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from config import PICK_TARGETS, PLACE_TARGETS
from VLA.audit_training_readiness import physical_scene_key, verify_readiness_source
from VLA.common.provenance import sha256_file


def feedback_counts(arrays):
    actions = arrays["canonical_actions"]
    closure = arrays["gripper_closure"]
    correct_before = np.r_[False, arrays["next_grasped_object"][:-1].astype(bool)]
    uncertain = (closure > .5) & ~correct_before
    verified = arrays["next_verified_grasp"].astype(bool)
    correct = arrays["next_grasped_object"].astype(bool)
    return {"frames": len(actions), "initial_closed_without_correct_target": int(uncertain[0]),
        "closed_without_correct_target_frames": int(uncertain.sum()),
        "reopen_labels_from_those_states": int((uncertain & (actions[:, 6] < -.2)).sum()),
        "closed_upward_labels_from_those_states": int((uncertain & (actions[:, 6] > .2) & (actions[:, 2] > .005)).sum()),
        "final_correct_target_held": int(correct[-1]), "final_verified_bilateral_target_grasp": int(verified[-1]),
        "correct_target_loss_transitions": int((correct[:-1] & ~correct[1:]).sum()),
        "max_measured_next_vs_requested_closure_gap": float(np.max(np.abs(arrays["next_gripper_closure"] - arrays["next_requested_gripper_closure"])))}


def audit(source, source_audit, output):
    source, source_audit, output = map(Path, (source, source_audit, output))
    source = source.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    index = json.loads((source / "index.json").read_text())
    proof = json.loads(source_audit.read_text())
    if (not proof.get("passed") or proof.get("failures") or Path(proof["dataset"]).resolve() != source
            or proof["index_sha256"] != sha256_file(source / "index.json")
            or proof["manifest_sha256"] != sha256_file(source / "dataset_manifest.json")
            or not verify_readiness_source(source, manifest, proof)):
        raise ValueError("Exact, successful V9 source audit with measured aperture required")
    if manifest.get("split") not in (None, "train"):
        raise ValueError("Do not inspect outer validation labels for training support decisions")
    tasks = defaultdict(list)
    scenes = defaultdict(set)
    frame_metrics = defaultdict(Counter)
    maximum_gap = defaultdict(float)
    prefix_episodes = Counter()
    prefix_by_task = Counter()
    for row in index:
        task = row["task"]
        group = physical_scene_key(task)
        tasks[task["name"]].append((row, group))
        scenes[group].add(task["prompt"])
        prefix = row.get("recovery_start_actions_not_training_labels", [])
        if prefix:
            prefix_episodes[row.get("teacher_policy", "unknown")] += 1
            prefix_by_task[task["name"]] += 1
        with np.load(source / row["file"], allow_pickle=False) as arrays:
            metrics = feedback_counts(arrays)
        maximum_gap[task["name"]] = max(maximum_gap[task["name"]], metrics.pop("max_measured_next_vs_requested_closure_gap"))
        frame_metrics[task["name"]].update(metrics)
    support = {}
    possible_pairs = {(p, d) for p in PICK_TARGETS for d in PLACE_TARGETS if p != d}
    for name, rows in tasks.items():
        pick_groups, destination_groups, pair_groups = defaultdict(set), defaultdict(set), defaultdict(set)
        initial_q = set()
        for row, group in rows:
            task = row["task"]
            pick_groups[task["pick_name"]].add(group)
            if task.get("place_name"):
                destination_groups[task["place_name"]].add(group)
                pair_groups[(task["pick_name"], task["place_name"])].add(group)
            with np.load(source / row["file"], allow_pickle=False) as arrays:
                initial_q.add(arrays["joint_positions"][0].tobytes())
        pairs = set(pair_groups)
        support[name] = {"episodes": len(rows), "frames": frame_metrics[name]["frames"],
            "independent_physical_layouts": len({group for _, group in rows}),
            "pick_independent_layout_counts": {p: len(groups) for p, groups in pick_groups.items()},
            "destination_independent_layout_counts": {d: len(groups) for d, groups in destination_groups.items()},
            "missing_pick_names": sorted(set(PICK_TARGETS) - set(pick_groups)),
            "distinct_initial_measured_joint_states": len(initial_q),
            "documented_recovery_prefix_episodes": prefix_by_task[name],
            "feedback_diagnostics": dict(frame_metrics[name]), "max_measured_next_vs_requested_closure_gap": maximum_gap[name]}
        if name in ("move", "pick_place"):
            support[name].update(possible_nonself_object_destination_pairs=len(possible_pairs),
                covered_pairs=len(pairs), unrepresented_pairs=len(possible_pairs - pairs),
                pairs_with_only_one_independent_layout=sum(len(groups) == 1 for groups in pair_groups.values()),
                minimum_independent_layouts_per_represented_pair=min(map(len, pair_groups.values()), default=0))
    report = {"scope": "V9 training-data support diagnostic; NOT a data-sufficiency gate or learned success",
        "source": str(source), "source_index_sha256": proof["index_sha256"], "source_manifest_sha256": proof["manifest_sha256"],
        "source_audit_sha256": sha256_file(source_audit), "source_payload_fingerprint": proof["source_payload_fingerprint"],
        "episodes": len(index), "transitions": sum(row["steps"] for row in index),
        "synchronized_rgb_views": 2, "rgb_images": 2 * sum(row["steps"] for row in index),
        "independent_physical_layouts": len(scenes),
        "layouts_with_multiple_text_instructions": sum(len(prompts) > 1 for prompts in scenes.values()),
        "documented_recovery_prefix_episodes": sum(prefix_episodes.values()), "recovery_prefix_teachers": dict(prefix_episodes),
        "state_contract": manifest["gripper_state_contract"], "tasks": support,
        "final_test_used": False, "validation_used": False, "data_sufficiency_established": False,
        "caveats": ["Two RGB views and correlated neighboring frames are not independent demonstrations.",
            "Closed without correct target can be ordinary acquisition or wrong-object holding, NOT proof of an empty failed grasp.",
            "Simulator target/verified-grasp flags are audit labels only, never policy inputs; startup protocols reject existing grasps.",
            "This checks measured aperture, not a new force/tactile feedback modality.",
            "Legacy pick_place terminates held at the goal, not released-and-settled placement.",
            "Missing compositional pairs/recovery support cannot be repaired by longer optimization alone."]}
    # Reverify all audited source payloads and references after reading them.
    if (sha256_file(source / "index.json") != proof["index_sha256"]
            or sha256_file(source / "dataset_manifest.json") != proof["manifest_sha256"]
            or not verify_readiness_source(source, manifest, proof)):
        raise RuntimeError("Audited source changed during support analysis")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return {k: v for k, v in report.items() if k not in ("source_payload_fingerprint", "tasks")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.source_audit, args.output), indent=2))


if __name__ == "__main__":
    main()
