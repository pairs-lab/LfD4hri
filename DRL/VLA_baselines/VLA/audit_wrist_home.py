"""Train-scene-only audit of rigid hand-eye mounts and reachable raised homes.

Separates image-boundary loss from robot occlusion using a fixed camera and
robot-removed render (NO physics step while hidden). Overscan retains focal
length and measures FOV loss. Simulator labels never enter policy inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from config import PICK_TARGETS
from VLA.audit_camera_geometry import POSITIONS
from VLA.common.observation_profile import (ProfiledManipulationEnv, baseline_profile,
    camera_geometry, profile_hash, render_geometry)
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.tasks import TASKS, TaskSpec
from VLA.common.training_selection import select_probe_rows
from VLA.common.provenance import sha256_file
from VLA.common.environment_factory import environment_class_for_profile


TABLE_CORNERS = [(-.45, -.85, .004), (.45, -.85, .004),
                 (.45, -.15, .004), (-.45, -.15, .004)]


def ids(seg):
    return np.where(seg >= 0, seg & ((1 << 24) - 1), -1)


def projected_points(points, view, projection, size):
    matrix = np.asarray(projection).reshape(4, 4, order="F") @ np.asarray(view).reshape(4, 4, order="F")
    clip = matrix @ np.c_[points, np.ones(len(points))].T
    ndc = clip[:3] / clip[3:4]
    px = np.c_[(ndc[0] + 1) * size / 2, (1 - ndc[1]) * size / 2]
    visible = (clip[3] > 0) & (np.abs(ndc[:2]) <= 1).all(axis=0) & (np.abs(ndc[2]) <= 1)
    return px.tolist(), visible.tolist()


def mask_stats(mask):
    yy, xx = np.where(mask)
    return {"pixels": int(mask.sum()), "bbox": [int(np.ptp(xx) + 1), int(np.ptp(yy) + 1)] if len(xx) else [0, 0],
            "margin": int(min(xx.min(), yy.min(), mask.shape[1]-1-xx.max(), mask.shape[0]-1-yy.max())) if len(xx) else None}


def measure(env, camera, size=224, remove_robot=True):
    cfg = env.observation_profile["cameras"][camera]
    geometry = camera_geometry(env, cfg, camera)
    rgb, seg, view, projection = render_geometry(env, geometry, cfg["fov_degrees"], size)
    target_mask = ids(seg) == env.pick_id
    row = {**mask_stats(target_mask)}
    robot = (ids(seg) == env.robot_id) | (ids(seg) == env.gripper.body)
    row["robot_border_margin"] = mask_stats(robot)["margin"]
    _, corner_inside = projected_points(TABLE_CORNERS, view, projection, size)
    row["table_corners_inside_fov"] = sum(corner_inside)
    row["all_scene_objects_pixels"] = {name: int((ids(seg) == body).sum()) for name, body in env.object_ids.items()}
    if camera == "wrist":
        tcp = np.asarray(env.gripper.grasp_center()) - geometry[0]
        axis = geometry[1] - geometry[0]
        row["tcp_axis_angle_degrees"] = float(np.degrees(np.arccos(np.clip(np.dot(tcp, axis) / np.linalg.norm(tcp) / np.linalg.norm(axis), -1, 1))))
    if not remove_robot:
        return row, rgb
    states = [(body, p.getBasePositionAndOrientation(body))
              for body in (env.robot_id, env.gripper.body)]
    saved_state = p.saveState(physicsClientId=env.client_id)
    try:
        for body, (position, quat) in states:
            p.resetBasePositionAndOrientation(body, np.asarray(position) + (20., 20., 20.), quat)
        overscan_fov = np.degrees(2*np.arctan(2*np.tan(np.radians(cfg["fov_degrees"])/2)))
        _, overscan_seg, _, _ = render_geometry(env, geometry, overscan_fov, size*2)
        wide_mask = ids(overscan_seg) == env.pick_id
        centre = wide_mask[size//2:size//2+size, size//2:size//2+size]
        # Same focal pixels. Pixel rounding at crop border can differ by one.
        whole, cropped, visible = int(wide_mask.sum()), int(centre.sum()), int(target_mask.sum())
        row.update(robot_removed_overscan_pixels=whole, robot_removed_native_fov_pixels=cropped,
                   fov_retention=cropped/whole if whole else None,
                   robot_occlusion_retention=min(1., visible/cropped) if cropped else None)
    finally:
        p.restoreState(saved_state, physicsClientId=env.client_id)
        p.removeState(saved_state, physicsClientId=env.client_id)
    restored, _, _, _ = render_geometry(env, geometry, cfg["fov_degrees"], size)
    if not np.array_equal(restored, rgb):
        raise RuntimeError("Occlusion audit changed original camera image after restore")
    return row, rgb


def solve_home(profile, eye_height):
    """IK only in audit; deploy a fixed saved q, never task-dependent IK reset."""
    with environment_class_for_profile(profile)(profile, image_size=224, render_observations=False) as env:
        env.reset(TASKS["pick"])
        state = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True)
        rotation = np.asarray(p.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        cfg = profile["cameras"]["wrist"]
        direction = rotation @ (np.asarray(cfg["target_in_tool"]) - cfg["eye_in_tool"])
        # Choose a fixed eye XY so its optical axis intersects table centre.
        eye = np.array([0., -.50, eye_height]) - direction * (eye_height/(-direction[2]))
        eye[2] = eye_height
        tool = eye - rotation @ cfg["eye_in_tool"]
        for _ in range(15):
            q = p.calculateInverseKinematics(env.robot_id, env.robot_spec.mount_link, tool, state[5],
                                             maxNumIterations=250, residualThreshold=1e-7)
            for j, val in zip(env.robot_spec.arm_joint_ids, q):
                p.resetJointState(env.robot_id, j, val)
        actual = p.getLinkState(env.robot_id, env.robot_spec.mount_link, computeForwardKinematics=True)
        residual = float(np.linalg.norm(np.asarray(actual[4])-tool))
        if residual > .003:
            raise ValueError(f"Unreachable camera height {eye_height}: IK residual {residual}")
        result = copy.deepcopy(profile)
        result["home_joints"] = list(q)
        result["requested_eye_height_m"] = eye_height
        result["ik_position_residual_m"] = residual
        return result


def candidates():
    yield baseline_profile()
    mounts = {
        "axis_side_x65": ((.065, 0., .045), (.065, 0., .395)),
        "axis_side_y65": ((0., .065, .045), (0., .065, .395)),
        "axis_diagonal": ((.07, .05, .04), (.07, .05, .39)),
        "tcp_toein_x65": ((.065, 0., .045), (0., 0., .18)),
        "coaxial_blocked_control": ((0., 0., -.01), (0., 0., .34)),
    }
    for name, (eye, target) in mounts.items():
        for height in (.55, .65):
            profile = baseline_profile()
            profile["name"] = f"{name}_h{int(height*100)}"
            profile["cameras"]["wrist"].update(eye_in_tool=eye, target_in_tool=target)
            try:
                yield solve_home(profile, height)
            except ValueError as exc:
                print(f"Rejected {profile['name']}: {exc}", flush=True)


def source_tasks(source, per_task, include_fixed_destinations=False):
    index = json.loads((source / "index.json").read_text())
    return [TaskSpec.from_dict(item["task"]) for name in TASKS
            for item in select_probe_rows([x for x in index if x["task"]["name"] == name],
                                           per_task, include_fixed_destinations)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--home-height", type=float,
                        help="Audit a new fixed reachable home with camera eye at this world height")
    parser.add_argument("--wrist-fov", type=float, help="Audit fixed intrinsics; never dynamic task-dependent zoom")
    parser.add_argument("--front-fov", type=float)
    parser.add_argument("--front-target-z", type=float)
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--include-fixed-destinations", action="store_true",
                        help="Include all nine named regions, not just early physical bowl/block goals")
    parser.add_argument("--full-grid", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    profiles = [json.loads(args.profile.read_text())] if args.profile else list(candidates())
    if args.home_height is not None:
        if not args.profile or not .4 <= args.home_height <= .75:
            raise ValueError("home-height requires a base profile and height in [0.4,0.75]m")
        profiles = [solve_home(profiles[0], args.home_height)]
        profiles[0]["name"] = profiles[0]["name"].rsplit("_h", 1)[0] + f"_h{round(args.home_height*100)}"
    if args.wrist_fov is not None:
        if not args.profile or not 60 <= args.wrist_fov <= 110:
            raise ValueError("wrist-fov requires a profile and FOV in [60,110]degrees")
        profiles[0]["cameras"]["wrist"]["fov_degrees"] = args.wrist_fov
        profiles[0]["name"] += f"_f{round(args.wrist_fov)}"
    if args.front_fov is not None or args.front_target_z is not None:
        if not args.profile:
            raise ValueError("Front camera changes require an explicit base profile")
        if args.front_fov is not None:
            profiles[0]["cameras"]["front"]["fov_degrees"] = args.front_fov
        if args.front_target_z is not None:
            target = list(profiles[0]["cameras"]["front"]["target_world"])
            target[2] = args.front_target_z
            profiles[0]["cameras"]["front"]["target_world"] = target
        profiles[0]["name"] += f"_front{profiles[0]['cameras']['front']['fov_degrees']:g}z{profiles[0]['cameras']['front']['target_world'][2]:g}"
    tasks = source_tasks(args.source, args.per_task, args.include_fixed_destinations) if args.source else list(TASKS.values())
    rows, summaries = [], []
    for profile in profiles:
        name = profile["name"]
        folder = args.output / name
        folder.mkdir()
        (folder / "profile.json").write_text(json.dumps(profile, indent=2)+"\n")
        panels = []
        successes = []
        with environment_class_for_profile(profile)(profile, image_size=224, required_success_steps=1) as env:
            for task_index, task in enumerate(tasks):
                observation = env.reset(task)
                if profile.get("gripper_controller") == "body_contact_v9":
                    from VLA.collision_planned_teacher import CollisionPlannedTeacher
                    policy = CollisionPlannedTeacher(env)
                else:
                    policy = ScriptedPolicy(env)
                policy.reset(task)
                done = False
                snapshots = set()
                worst = None
                while True:
                    phase = "final" if done else policy.phase
                    frame_row = {"profile": name, "task": task.name, "target": task.pick_name,
                                 "destination": task.place_name,
                                 "task_index": task_index, "step": env.step_count, "phase": phase}
                    images = []
                    for camera in ("front", "wrist"):
                        row, rgb = measure(env, camera)
                        frame_row[camera] = row
                        images.append(rgb)
                    rows.append(frame_row)
                    if env.step_count == 0 and "overview" in profile["cameras"]:
                        frame_row["overview"], overview_rgb = measure(env, "overview", remove_robot=False)
                        Image.fromarray(overview_rgb).save(folder / f"task{task_index:03d}_overview_reset.png")
                    retention = frame_row["wrist"].get("robot_occlusion_retention")
                    if retention is not None:
                        score = retention * (frame_row["wrist"].get("fov_retention") or 0.)
                        if worst is None or score < worst[0]:
                            worst = (score, images[1].copy(), env.step_count)
                    if phase not in snapshots or env.step_count == 0:
                        cell = Image.new("RGB", (448, 250), "black")
                        for k, rgb in enumerate(images):
                            cell.paste(Image.fromarray(rgb), (224*k, 26))
                        ImageDraw.Draw(cell).text((3, 4), f"{name} {task.name} {phase} {task.pick_name}", fill="white")
                        cell.save(folder / f"task{task_index:03d}_{phase}.png")
                        panels.append(cell)
                        snapshots.add(phase)
                    if done:
                        break
                    action = policy.predict(observation)
                    observation, _, done, info = env.step(action)
                successes.append(bool(info["success"]))
                if worst is not None:
                    Image.fromarray(worst[1]).save(folder / f"task{task_index:03d}_worst_wrist_step{worst[2]:03d}.png")
            if args.full_grid:
                from dataclasses import replace
                for position_name, xy in POSITIONS.items():
                    for pick_name in PICK_TARGETS:
                        task = replace(TASKS["touch"], pick_name=pick_name,
                                       pick_position=(*xy, .03 if pick_name.endswith((" block", " egg")) else 0.))
                        env.reset(task)
                        row = {"profile": name, "task": "reset_grid", "target": pick_name,
                               "position": position_name, "step": 0, "phase": "grid"}
                        cell = Image.new("RGB", (448, 246), "black")
                        for k, camera in enumerate(("front", "wrist")):
                            row[camera], rgb = measure(env, camera)
                            cell.paste(Image.fromarray(rgb), (224*k, 22))
                        if "overview" in profile["cameras"]:
                            row["overview"], _ = measure(env, "overview", remove_robot=False)
                        ImageDraw.Draw(cell).text((3, 3), f"{position_name} {pick_name} | front / wrist", fill="white")
                        cell.save(folder / f"grid_{position_name}_{pick_name.replace(' ', '_')}.png")
                        rows.append(row)
        sheet = Image.new("RGB", (448*4, 250*((len(panels)+3)//4)), "black")
        for k, cell in enumerate(panels):
            sheet.paste(cell, (448*(k%4), 250*(k//4)))
        sheet.save(folder / "trajectory_224_contact_sheet.png")
        selected = [r for r in rows if r["profile"] == name]
        resets = [r for r in selected if r["step"] == 0]
        close = [r for r in selected if r["phase"] in ("descend", "close", "lift", "lower", "hold", "final")]
        summary = {"profile": name, "profile_sha256": profile_hash(profile), "teacher_success": sum(successes),
                   "teacher_trials": len(successes), "frames_audited": len(selected),
                   "min_reset_table_corners": min(r["wrist"]["table_corners_inside_fov"] for r in resets),
                   "min_reset_wrist_pixels": min(r["wrist"]["pixels"] for r in resets),
                   "min_reset_front_robot_border_margin": min((r["front"]["robot_border_margin"] for r in resets if r["front"]["robot_border_margin"] is not None), default=None),
                   "min_reset_front_pixels": min(r["front"]["pixels"] for r in resets),
                   "min_reset_overview_robot_border_margin": min((r["overview"]["robot_border_margin"] for r in resets if "overview" in r and r["overview"]["robot_border_margin"] is not None), default=None),
                   "min_reset_wrist_fov_retention": min((r["wrist"]["fov_retention"] for r in resets if r["wrist"]["fov_retention"] is not None), default=None),
                   "reset_scene_objects_under_49_pixels_wrist": sum(sum(v < 49 for v in r["wrist"]["all_scene_objects_pixels"].values()) for r in resets),
                   "all_frames_both_views_under_49_pixels": sum(max(r[c]["pixels"] for c in ("front", "wrist")) < 49 for r in selected),
                   "near_grasp_wrist_absent_frames": sum(r["wrist"]["pixels"] == 0 for r in close),
                   "all_frames_both_views_absent": sum(max(r[c]["pixels"] for c in ("front", "wrist")) == 0 for r in selected),
                   "min_near_grasp_fov_retention": min((r["wrist"]["fov_retention"] for r in close if r["wrist"]["fov_retention"] is not None), default=None),
                   "min_near_grasp_robot_occlusion_retention": min((r["wrist"]["robot_occlusion_retention"] for r in close if r["wrist"]["robot_occlusion_retention"] is not None), default=None)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        (args.output / "audit.json").write_text(json.dumps({"summary": summaries, "rows": rows,
            "source_index_sha256": sha256_file(args.source / "index.json") if args.source else None,
            "includes_all_named_regions": bool(args.include_fixed_destinations),
            "final_test_used": False, "production_defaults_changed": False,
            "caveat": "Teacher success validates physics, not learned VLA success. Robot-removed masks retain clutter occlusion; no sim tick during removal. 224px native renders are geometric audits, not per-model preprocessing parity."}, indent=2)+"\n")


if __name__ == "__main__":
    main()
