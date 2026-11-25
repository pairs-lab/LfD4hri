"""Teacher action replay: task text must not control grasp physics.

Execute IDENTICAL recorded actions in identical canonical scenes while only
changing requested target text/scoring. Never feeds teacher state to a VLA.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment
from VLA.common.observation_profile import validate_profile, profile_hash
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((args.source / "index.json").read_text())
    canonical_source = Path(json.loads((args.source / "audit.json").read_text())["source"])
    canonical = {r["file"]: r for r in json.loads((canonical_source / "index.json").read_text())}
    profile = validate_profile(json.loads(args.profile.read_text()))
    results = []
    # V9 must not turn the requested task identity into a gripper filter.
    # Keep the historical two-mode diagnostic only for legacy controllers.
    modes = (("all_dynamic_scene_objects",) if profile.get("gripper_controller") == "body_contact_v9"
             else ("selected_target_legacy", "all_dynamic_scene_objects"))
    for mode in modes:
        geometry = {**profile, "grasp_candidate_scope": mode}
        with environment_class_for_profile(geometry)(geometry, image_size=224, render_observations=False,
                                     required_success_steps=1) as env:
            for name in ("touch", "pick", "move", "pick_place"):
                row = next(r for r in index if r["task"]["name"] == name and r["variant"] == 0)
                initial = TaskSpec.from_dict(canonical[row["paired_scene_source"]]["task"])
                first = TaskSpec.from_dict(row["task"])
                other_row = next(r for r in index if r["paired_scene_source"] == row["paired_scene_source"] and r["variant"] == 1)
                second = TaskSpec.from_dict(other_row["task"])
                with np.load(args.source / row["file"], allow_pickle=False) as data:
                    actions = data["canonical_actions"]
                branches = []
                for task in (first, second):
                    wrapper = CanonicalSceneResetEnvironment(env, initial,
                        row.get("recovery_start_actions_not_training_labels", []))
                    obs = wrapper.reset(task)
                    candidate_names = sorted(n for n, body in env.object_ids.items() if body in env.gripper.object_ids)
                    # Render once: sensor equality is tested independently of
                    # zero placeholder images used by fast physics replay.
                    reset_rgb = np.concatenate([env.render(c).reshape(-1) for c in ("front", "wrist")])
                    frames, grasped = [], []
                    for action in actions:
                        obs, _, _, info = env.step(action)
                        worlds = [p.getBasePositionAndOrientation(body) for n, body in sorted(env.object_ids.items())]
                        frames.append(np.concatenate([obs.joint_positions,
                            np.concatenate([np.r_[position, quaternion] for position, quaternion in worlds])]))
                        grasped.append(next((n for n, body in env.object_ids.items() if body == env.gripper.grasped_object), None))
                    branches.append({"states": np.asarray(frames), "grasped": grasped, "reset_rgb": reset_rgb,
                                     "candidate_names": candidate_names, "semantic_success": bool(info["semantic_raw_success"]),
                                     "prompt": task.prompt})
                error = float(np.max(np.abs(branches[0]["states"] - branches[1]["states"])))
                result = {"mode": mode, "task": name, "source_file": row["file"],
                          "frames_per_branch": len(actions), "physical_state_max_abs_error": error,
                          "reset_rgb_max_pixel_error": int(np.max(np.abs(branches[0]["reset_rgb"].astype(int) - branches[1]["reset_rgb"].astype(int)))),
                          "grasp_sequence_equal": branches[0]["grasped"] == branches[1]["grasped"],
                          "branches": [{k: v for k, v in branch.items() if k not in ("states", "reset_rgb")}
                                       for branch in branches]}
                results.append(result)
                print(f"{mode} {name}: physics error {error}, grasp equality {result['grasp_sequence_equal']}", flush=True)
    new = [r for r in results if r["mode"] == "all_dynamic_scene_objects"]
    report = {"passed": all(r["physical_state_max_abs_error"] == 0 and r["reset_rgb_max_pixel_error"] == 0
                             and r["grasp_sequence_equal"] for r in new),
              "profile_sha256": profile_hash(profile), "source": str(args.source.resolve()),
              "source_index_sha256": sha256_file(args.source / "index.json"),
              "profile_source": str(args.profile.resolve()), "results": results,
              "scope": "Four training-scene counterfactual physics probes, NOT learned-policy evaluation or full teacher coverage",
              "production_dataset_changed": False, "final_test_used": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
