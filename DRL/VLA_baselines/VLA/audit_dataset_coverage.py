"""Dataset quality/coverage evidence; no automatic declaration of sufficiency."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)), "max": float(values.max())}


def scene_key(task):
    objects = [(task["pick_name"], task["pick_position"])]
    objects += [(item["name"], item["position"]) for item in task["scene_objects"]]
    return hashlib.sha256(json.dumps(sorted(objects), separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((args.source / "index.json").read_text())
    manifest = json.loads((args.source / "dataset_manifest.json").read_text())
    tasks = defaultdict(lambda: {"episodes": 0, "frames": 0, "lengths": [],
        "prompts": Counter(), "pick_counts": Counter(), "pair_counts": Counter(),
        "workspace_bins_per_pick": defaultdict(set), "initial_joints": Counter(),
        "low_motion_frames": 0, "translation_saturated_frames": 0,
        "front_invisible_frames": 0, "wrist_invisible_frames": 0,
        "both_invisible_frames": 0, "final_grasped_episodes": 0,
        "final_closed_command_episodes": 0, "grasp_loss_transitions": 0})
    failures = []
    scenes = defaultdict(set)
    init_positions = []
    for number, item in enumerate(index):
        task = item["task"]
        current = tasks[task["name"]]
        scenes[scene_key(task)].add(task["prompt"])
        current["episodes"] += 1
        current["lengths"].append(item["steps"])
        current["pick_counts"][task["pick_name"]] += 1
        current["prompts"][task["prompt"]] += 1
        if task.get("place_name") is not None:
            current["pair_counts"][task["pick_name"] + " -> " + task["place_name"]] += 1
        position = np.asarray(task["pick_position"], dtype=np.float64)
        init_positions.append(position)
        # A spatial support diagnostic, not a hard-coded workspace assertion.
        current["workspace_bins_per_pick"][task["pick_name"]].add(tuple(np.floor(position[:2] / .1).astype(int)))
        with np.load(args.source / item["file"]) as episode:
            action = episode["canonical_actions"]
            q = episode["joint_positions"]
            closure = episode["gripper_closure"]
            next_closure = episode["next_gripper_closure"]
            grasped = episode["next_grasped_object"].astype(bool)
            front = episode["front_pick_mask"].reshape(len(action), -1).any(axis=1)
            wrist = episode["wrist_pick_mask"].reshape(len(action), -1).any(axis=1)
            current["frames"] += len(action)
            for values in (action, q, closure, next_closure):
                if not np.isfinite(values).all():
                    failures.append(f"nonfinite data: {item['file']}")
            if str(episode["prompt"].item()) != task["prompt"]:
                failures.append(f"prompt/index mismatch: {item['file']}")
            current["initial_joints"][hashlib.sha256(q[0].tobytes()).hexdigest()] += 1
            # Stationary grasp transitions may be necessary. Flag for review,
            # never silently delete or identify these as harmful no-op labels.
            low_motion = np.linalg.norm(action[:, :3], axis=1) < .001
            unchanged_gripper = np.abs(next_closure - closure) < 1e-6
            current["low_motion_frames"] += int((low_motion & unchanged_gripper).sum())
            # The scripted teacher caps each axis at 4.5cm; the shared
            # controller safety cap is 5cm. Do not confuse these with 2cm
            # reward distances or call ordinary 2cm motion saturated.
            current["translation_saturated_frames"] += int((np.abs(action[:, :3]).max(axis=1) >= .04499).sum())
            current["front_invisible_frames"] += int((~front).sum())
            current["wrist_invisible_frames"] += int((~wrist).sum())
            current["both_invisible_frames"] += int((~front & ~wrist).sum())
            current["final_grasped_episodes"] += int(grasped[-1])
            current["final_closed_command_episodes"] += int(next_closure[-1] > .5)
            current["grasp_loss_transitions"] += int((grasped[:-1] & ~grasped[1:]).sum())
        if number % 400 == 0:
            print(f"Audited {number}/{len(index)} trajectories", flush=True)
    output_tasks = {}
    for name, values in tasks.items():
        pairs = values.pop("pair_counts")
        prompts = values.pop("prompts")
        init = values.pop("initial_joints")
        bins = values.pop("workspace_bins_per_pick")
        lengths = values.pop("lengths")
        output_tasks[name] = {**values,
            "duration_seconds": distribution(np.asarray(lengths) / manifest["control_hz"]),
            "distinct_prompts": len(prompts), "episodes_per_prompt": distribution(list(prompts.values())),
            "distinct_pick_destination_pairs": len(pairs),
            "episodes_per_pair": distribution(list(pairs.values())) if pairs else None,
            "pair_counts": dict(pairs), "pick_counts": dict(values["pick_counts"]),
            "distinct_initial_joint_states": len(init),
            "workspace_10cm_bins_per_pick": {key: len(value) for key, value in bins.items()},
        }
        for key in ("low_motion_frames", "translation_saturated_frames", "front_invisible_frames",
                    "wrist_invisible_frames", "both_invisible_frames"):
            output_tasks[name][key.replace("_frames", "_fraction")] = values[key] / values["frames"]
    frames = sum(item["steps"] for item in index)
    budgets = {
        "smolvla_connector_current_new_stage": (5000, 16),
        "rdt_h10_current_new_stage": (20000, 4),
        "pi0_v6_planned": (30000, 4), "pi0_fast_v6_planned": (30000, 4),
        "openvla_v6_planned": (10000, 8),
    }
    report = {
        "numerically_passed": not failures, "failures": failures,
        "source": str(args.source.resolve()), "episodes": len(index), "frames": frames,
        "rgb_image_count": frames * 2, "control_hz": manifest["control_hz"],
        "teacher_translation_saturation_threshold_m": .04499,
        "tasks": output_tasks, "distinct_exact_scene_layouts": len(scenes),
        "layouts_with_multiple_task_instructions": sum(len(prompts) > 1 for prompts in scenes.values()),
        "initial_object_xyz_range": {"min": np.min(init_positions, axis=0).tolist(), "max": np.max(init_positions, axis=0).tolist()},
        "sample_budgets": {name: {"optimizer_updates": updates, "effective_samples_per_update": batch,
            "source_frame_presentations": updates * batch, "nominal_frame_passes": updates * batch / frames}
            for name, (updates, batch) in budgets.items()},
        "budget_caveat": "RDT samples four random frames per episode; nominal passes are not uniform epochs. Warm-start stages excluded. Planned budgets are not completed training.",
        "recovery_demonstration_provenance": "No explicit disturbance/recovery protocol documented in this source manifest",
        "gripper_sensor_semantics": "Last commanded closure/open; not measured physical aperture or grasp sensing",
        "data_sufficiency": "NOT ESTABLISHED by counts, numerical passes, expert success or training loss",
        "sources": [
            "https://github.com/openvla/openvla#vla-performance-troubleshooting",
            "https://github.com/Physical-Intelligence/openpi/blob/main/examples/ur5/README.md",
            "https://github.com/thu-ml/RoboticsDiffusionTransformer#fine-tuning-on-your-own-dataset",
            "https://arxiv.org/html/2506.01844v1#S4.SS3",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
