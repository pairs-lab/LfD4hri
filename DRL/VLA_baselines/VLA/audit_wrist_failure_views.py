"""Rigid hand-eye candidates on actual failure states; no runtime camera change."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from VLA.audit_grasp_recovery import FailedPrefixReset
from VLA.audit_finetune_data import resized_mask_metrics
from VLA.common.observation_profile import camera_geometry, render_geometry
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def corner_count(view, projection):
    vp = np.asarray(projection).reshape(4, 4, order="F") @ np.asarray(view).reshape(4, 4, order="F")
    points = np.array([[x, y, .004, 1] for x in (-.45, .45) for y in (-.85, -.15)])
    clip = points @ vp.T
    ndc = clip[:, :3] / clip[:, 3:]
    return int(((clip[:, 3] > 0) & (np.abs(ndc).max(axis=1) <= 1)).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-wider-fov", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    metadata = json.loads((args.correction / "metadata.json").read_text())
    row = next(r for r in json.loads((args.source / "index.json").read_text())
               if r["file"] == metadata["source_failed_episode"])
    records = []
    with audit_environment(args.source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.25, use_semantic_success=True) as env:
        current = env.observation_profile["cameras"]["wrist"]
        candidates = {"current_parallel65": copy.deepcopy(current)}
        for name, eye, target in (("parallel35", [0, .035, .045], [0, .035, .395]),
                                  ("tilt65_to20", [0, .065, .045], [0, .020, .240]),
                                  ("tilt35_to0", [0, .035, .045], [0, 0, .240])):
            candidates[name] = {**copy.deepcopy(current), "eye_in_tool": eye, "target_in_tool": target}
        if args.include_wider_fov:
            for fov in (90, 95, 105):
                candidates[f"tilt65_f{fov}"] = {**copy.deepcopy(candidates["tilt65_to20"]), "fov_degrees": fov}
        sheet = Image.new("RGB", (224 * len(candidates), 744), "black")
        for phase_index, phase in enumerate(("home", "empty_closed", "held")):
            if phase == "home":
                reset_source_episode(env, args.source, row)
            elif phase == "empty_closed":
                FailedPrefixReset(env, args.source, row, metadata["recovery_start_actions_not_training_labels"]).reset(TaskSpec.from_dict(row["task"]))
            else:
                with np.load(args.correction / "correction.npz") as data:
                    for action in data["canonical_actions"]:
                        env.step(action)
            for column, (name, cfg) in enumerate(candidates.items()):
                rgb, seg, view, projection = render_geometry(env, camera_geometry(env, cfg, "wrist"), cfg["fov_degrees"], 384)
                bodies = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
                target = resized_mask_metrics(bodies == env.pick_id)
                robot = np.isin(bodies, [env.robot_id, env.gripper.body])
                metrics = {"phase": phase, "candidate": name, "rigid_camera_config": cfg,
                    "target_pixels_224": target[0], "target_bounds_224": list(target[1:]),
                    "robot_occupancy_fraction": float(robot.mean()),
                    "physical_table_corner_projection_count": corner_count(view, projection)}
                records.append(metrics)
                Image.fromarray(rgb).save(args.output / f"{phase}_{name}.png")
                cell = Image.new("RGB", (224, 248), "black")
                cell.paste(Image.fromarray(rgb).resize((224, 224)), (0, 24))
                ImageDraw.Draw(cell).text((2, 3), f"{phase} {name} | px={target[0]}", fill="white")
                sheet.paste(cell, (column * 224, phase_index * 248))
    sheet.save(args.output / "candidate_contact_sheet.png")
    (args.output / "audit.json").write_text(json.dumps({"scope": "Single TRAIN-scene rigid-camera stress comparison, NOT qualified new camera profile",
        "records": records, "active_camera_changed": False, "camera_tracks_selected_object": False,
        "final_test_used": False, "requires_full_workspace_and_all_task_grid_before_promotion": True}, indent=2) + "\n")


if __name__ == "__main__":
    main()
