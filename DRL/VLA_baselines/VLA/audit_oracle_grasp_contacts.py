"""Measure actual pad geometry and contacts at oracle grasp-assist activation.

No controller/asset/source data change. Success is reported separately from
verified bilateral contact and central-body pinch, never treated as equivalent.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import pybullet as p
import trimesh

from VLA.audit_gripper_geometry import transform_points
from VLA.common.provenance import sha256_file
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def meshes(asset=None):
    asset = asset or Path(__file__).resolve().parents[1] / "asset/robotiq_2f_85"
    xml = ET.parse(asset / "robotiq_2f_85.urdf").getroot()
    result = {}
    for side, link_id in (("right", 4), ("left", 9)):
        link = xml.find(f"link[@name='robotiq_2f_85_{side}_pad']")
        for kind in ("visual", "collision"):
            descriptor = link.find(f"{kind}/geometry/mesh")
            loaded = trimesh.load(asset / descriptor.get("filename"), force="mesh", process=False)
            result[link_id, kind] = np.asarray(loaded.vertices) * np.array([float(x) for x in descriptor.get("scale").split()])
    return result


def measure(env, object_id, vertices):
    lo, hi = map(np.asarray, p.getAABB(object_id))
    centre = (lo + hi) / 2
    contacts = p.getContactPoints(bodyA=env.gripper.body, bodyB=object_id)
    by_link = {link: [c for c in contacts if c[3] == link] for link in env.gripper.PAD_LINKS}
    centres = {}
    ranges = {}
    for kind in ("visual", "collision"):
        points = []
        for link in env.gripper.PAD_LINKS:
            state = p.getLinkState(env.gripper.body, link, computeForwardKinematics=True)
            points.append(transform_points(vertices[link, kind], state[4], state[5]))
        centres[kind] = np.mean([v.mean(axis=0) for v in points], axis=0)
        ranges[kind] = [[float(v[:, 2].min()), float(v[:, 2].max())] for v in points]
    height = float(hi[2] - lo[2])
    return {"chosen_is_requested_object": object_id == env.pick_id,
        "object_aabb_min_m": lo.tolist(), "object_aabb_max_m": hi.tolist(),
        "pad_COM_center_m": np.mean([p.getLinkState(env.gripper.body, link, computeForwardKinematics=True)[0]
                                     for link in env.gripper.PAD_LINKS], axis=0).tolist(),
        "controller_grasp_reference_m": env.gripper.grasp_center().tolist(),
        "pad_mesh_centres_m": {k: v.tolist() for k, v in centres.items()},
        "pad_mesh_z_ranges_m": ranges,
        "collision_pad_centre_above_object_centre_m": float(centres["collision"][2] - centre[2]),
        "collision_pad_centre_relative_body_height": float((centres["collision"][2] - lo[2]) / height),
        "contact_count_all_gripper_links": len(contacts),
        "contact_counts_by_pad_link": {str(k): len(v) for k, v in by_link.items()},
        "both_pad_links_in_contact": all(bool(v) for v in by_link.values()),
        "contact_link_ids": sorted({c[3] for c in contacts}),
        "contact_object_relative_heights": [float((c[6][2] - lo[2]) / height) for c in contacts],
        "actual_motor_angle_rad": float(p.getJointState(env.gripper.body, 1)[0]),
        "commanded_motor_angle_rad": env.gripper.target_angle,
        "assist_constraint_present_before_capture": env.gripper.grasp_constraint is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objects", default="red apple,purple block,yellow banana,white egg,red gelatin_box")
    parser.add_argument("--calibrated-prototype", action="store_true", help="Separate audit-only TCP/contact/teacher experiment; not deployed")
    parser.add_argument("--freeze-contact-joints", action="store_true")
    parser.add_argument("--collision-planner", action="store_true")
    parser.add_argument("--body-v9", action="store_true", help="Versioned corrected visible pads/contact/feedback controller")
    parser.add_argument("--profile", type=Path, help="Explicit V9 deployment profile rather than a historical audit override")
    parser.add_argument("--tasks", default="pick", help="Comma-separated primitives or all")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = json.loads((args.source / "index.json").read_text())
    from config import PICK_TARGETS
    names = list(PICK_TARGETS) if args.objects == "all" else args.objects.split(",")
    tasks = ("touch", "pick", "move", "pick_place") if args.tasks == "all" else args.tasks.split(",")
    selected = [next(r for r in rows if r["task"]["name"] == task and r["task"]["pick_name"] == name
                     and not r.get("recovery_start_actions_not_training_labels")) for task in tasks for name in names]
    if args.body_v9:
        from VLA.common.body_environment import BodyGraspEnvironment, DERIVED_ASSETS
        if args.profile:
            profile = json.loads(args.profile.read_text())
        else:
            profile = json.loads((args.source / "dataset_manifest.json").read_text())["observation_profile"]
            profile = {**profile, "name": "body_contact_v9_audit", "gripper_controller": "body_contact_v9",
                       "gripper_state_contract": "encoder_fk_aperture_closure_v1"}
        environment = BodyGraspEnvironment(profile, image_size=384, required_success_steps=1,
            success_threshold_scale=1.25, use_semantic_success=True)
        vertices = meshes(DERIVED_ASSETS / "robotiq_2f_85")
    else:
        environment = audit_environment(args.source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.25, use_semantic_success=True)
        vertices = meshes()
    results = []
    with environment as env:
        for number, row in enumerate(selected):
            reset_source_episode(env, args.source, row)
            if args.body_v9:
                from VLA.collision_planned_teacher import CollisionPlannedTeacher
                policy = CollisionPlannedTeacher(env)
            elif args.calibrated_prototype:
                from VLA.grasp_contact_prototype import install_contact_prototype, MidbodyContactTeacher
                install_contact_prototype(env, vertices, args.freeze_contact_joints)
                if args.collision_planner:
                    from VLA.collision_planned_teacher import CollisionPlannedTeacher
                    policy = CollisionPlannedTeacher(env)
                else:
                    policy = MidbodyContactTeacher(env)
            else:
                policy = ScriptedPolicy(env)
            task = TaskSpec.from_dict(row["task"])
            policy.reset(task)
            original_create = env.gripper._create_assisted_grasp
            events = []
            folder = args.output / task.name / task.pick_name.replace(" ", "_")
            folder.mkdir(parents=True)
            def capture(object_id):
                event = measure(env, object_id, vertices)
                event.update(step=env.step_count, phase=policy.phase)
                events.append(event)
                for camera in ("front", "wrist"):
                    Image.fromarray(env.render(camera)).save(folder / (camera + "_at_assist.png"))
                # This camera is audit-only and NEVER a VLA input.
                centre = (np.asarray(event["object_aabb_min_m"]) + np.asarray(event["object_aabb_max_m"])) / 2
                view = p.computeViewMatrix((centre + [-.18, -.25, .13]).tolist(), centre.tolist(), [0, 0, 1])
                projection = p.computeProjectionMatrixFOV(42, 1, .01, 3)
                rgba = p.getCameraImage(512, 512, viewMatrix=view, projectionMatrix=projection, renderer=p.ER_TINY_RENDERER)[2]
                Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(512, 512, 4)[..., :3]).save(folder / "AUDIT_ONLY_closeup_at_assist.png")
                original_create(object_id)
            env.gripper._create_assisted_grasp = capture
            env.render_observations = False
            observation = env.observe()
            for _ in range(task.max_steps):
                observation, _, done, info = env.step(policy.predict(observation))
                if done:
                    break
            held = measure(env, env.pick_id, vertices)
            env.render_observations = True
            for camera in ("front", "wrist"):
                Image.fromarray(env.render(camera)).save(folder / (camera + "_terminal.png"))
            result = {"task": task.name, "target": task.pick_name, "source_episode": row["file"], "semantic_success": info["success"],
                "safety": {"violation": info.get("safety_violation"),
                    "diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
                    "min_clearance_m": getattr(env, "safety_min_clearance", None),
                    "max_mimic_error_rad": getattr(env, "safety_max_mimic_error", None)},
                "teacher_grasp_plan": getattr(policy, "plan", None),
                "assist_creation_events": events, "terminal_geometry": held}
            results.append(result)
            (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({"task": task.name, "target": task.pick_name, "success": info["success"],
                "safety_violation": info.get("safety_violation"),
                "both_pads_at_activation": events[0]["both_pad_links_in_contact"] if events else None,
                "centre_height_fraction_at_activation": events[0]["collision_pad_centre_relative_body_height"] if events else None}), flush=True)
    report = {"scope": "Actual oracle contact/geometry audit, NOT learned VLA performance", "results": results,
        "source_index_sha256": sha256_file(args.source / "index.json"),
        "controller_changed_in_this_audit_process_only": args.calibrated_prototype,
        "freeze_contact_joints_audit_experiment": args.freeze_contact_joints,
        "body_v9_controller": args.body_v9,
        "original_assets_changed": False, "derived_visual_pad_scale_fixed": args.body_v9, "final_test_used": False,
        "training_geometry_qualified": False,
        "oracle_success_is_not_bilateral_body_grasp_certification": True}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
