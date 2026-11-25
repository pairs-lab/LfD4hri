"""Small teacher-only goal-language probe in identical physical scenes.

No changes to the running train split or benchmark; exported labels are
candidate diagnostic data, never runtime oracle actions for a learned model.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_counterfactual_collection import CanonicalSceneResetEnvironment, alternate_destination
from VLA.collect_demos import collect_episode
from VLA.common.observation_profile import validate_profile
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts-per-task", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.layouts_per_task < 1:
        raise ValueError("Positive layouts-per-task required")
    args.output.mkdir(parents=True)
    index = json.loads((args.source / "index.json").read_text())
    rows, exported, cells = [], [], []
    profile = validate_profile(json.loads(args.profile.read_text()))
    teacher = ScriptedPolicy
    if profile.get("gripper_controller") == "body_contact_v9":
        from VLA.collision_planned_teacher import CollisionPlannedTeacher
        teacher = CollisionPlannedTeacher
    with environment_class_for_profile(profile)(profile, image_size=384, required_success_steps=1) as env:
        for name in ("move", "pick_place"):
            originals = [row for row in index if row["task"]["name"] == name][:args.layouts_per_task]
            for scene_number, original in enumerate(originals):
                first = TaskSpec.from_dict(original["task"])
                second = alternate_destination(first)
                wrapper = CanonicalSceneResetEnvironment(env, first)
                branches = []
                for variant, task in enumerate((first, second)):
                    arrays, metadata = collect_episode(wrapper, task, policy_factory=teacher)
                    filename = f"{name}_scene{scene_number:03d}_goal{variant}.npz"
                    np.savez_compressed(args.output / filename, **arrays)
                    exported.append({**metadata, "file": filename, "paired_scene_source": original["file"], "variant": variant,
                                     "intervention": "destination_only", "candidate_only": True})
                    branches.append(arrays)
                    print(f"{filename}: {task.prompt} teacher success={metadata['success']}", flush=True)
                common = min(len(b["canonical_actions"]) for b in branches)
                different = np.flatnonzero(np.any(np.abs(branches[0]["canonical_actions"][:common] - branches[1]["canonical_actions"][:common]) > 1e-6, axis=1))
                divergence = int(different[0]) if len(different) else None
                # Observations up through the first DIFFERENT action are still
                # physically identical. This late common observation is the
                # meaningful conditioning test, not an identical approach at t0.
                frame = divergence if divergence is not None else 0
                errors = {}
                for key in ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure"):
                    a, b = (branch[key][:frame + 1].astype(np.float64) for branch in branches)
                    errors[key] = float(np.max(np.abs(a - b)))
                row = {"task": name, "source_episode": original["file"], "prompts": [first.prompt, second.prompt],
                       "destinations": [first.place_name, second.place_name],
                       "teacher_success": all(r["success"] for r in exported[-2:]),
                       "first_different_action_frame": divergence, "common_observation_max_errors": errors,
                       "different_action_max_error": float(np.max(np.abs(branches[0]["canonical_actions"][:common] - branches[1]["canonical_actions"][:common])))}
                rows.append(row)
                for phase, f in (("reset", 0), ("goal_divergence", frame)):
                    cell = Image.new("RGB", (896, 264), "black")
                    for variant, branch in enumerate(branches):
                        for camera, key in enumerate(("front_rgb", "wrist_rgb")):
                            cell.paste(Image.fromarray(branch[key][f]).resize((224, 224)), (variant * 448 + camera * 224, 40))
                        ImageDraw.Draw(cell).text((variant * 448 + 4, 4), row["prompts"][variant], fill="white")
                        ImageDraw.Draw(cell).text((variant * 448 + 4, 20), f"{phase} frame {f}", fill="white")
                    cells.append(cell)
    sheet = Image.new("RGB", (896, 264 * len(cells)), "black")
    for i, cell in enumerate(cells):
        sheet.paste(cell, (0, i * 264))
    sheet.save(args.output / "destination_counterfactual_contact_sheet.png")
    report = {"passed": all(r["teacher_success"] and r["first_different_action_frame"] is not None
                             and all(v == 0 for v in r["common_observation_max_errors"].values()) for r in rows),
              "source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source / "index.json"),
              "observation_profile_sha256": env.observation_profile_sha256, "rows": rows,
              "scope": "Small destination-only teacher/label probe; not VLA performance or promoted training data",
              "running_training_dataset_changed": False, "final_test_used": False}
    (args.output / "index.json").write_text(json.dumps(exported, indent=2) + "\n")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise RuntimeError("Destination counterfactual source qualification failed")


if __name__ == "__main__":
    main()
