"""Recreate the NEXT uncommitted scene from a stopped TRAIN candidate RNG.

Diagnostic actions/images only, NOT demonstration labels or learned SR.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image
from VLA.collect_expanded_v9_candidate import FrozenRewardReset, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.dataset_replay_context import audit_environment
from VLA.neighbour_clearance_teacher import NeighbourClearanceTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("neighbour_clearance", "adaptive_neighbour", "open_verified", "staged_yaw_clear_touch"), default="neighbour_clearance")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads((args.source/"collection_plan.json").read_text())
    state = json.loads((args.source/"collection_state.json").read_text())
    rows = json.loads((args.source/"index.json").read_text())
    if len(rows) != state["completed_episodes"] or sha256_file(args.source/"index.json") != state["index_sha256"]:
        raise ValueError("Committed snapshot differs")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    seen = {x["file"] for x in rows}
    jobs = [(task, i) for task in select_tasks("all") for i in range(plan["episodes_per_task"])
            if f"{task.name}_{i:05d}.npz" not in seen]
    task, i = jobs[0]
    pick, place = stratified_names(task.name, i)
    sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
    args.output.mkdir(parents=True)
    record = {"scope": "Preserved uncommitted TRAIN exception reproduction, NEVER training labels",
              "parent_source": str(args.source.resolve()), "task": sampled.to_dict(),
              "index_sha256": state["index_sha256"], "state_sha256": sha256_file(args.source/"collection_state.json"),
              "scene_rng_before": state["rng_state"], "scene_rng_after": rng.bit_generator.state,
              "frames": [], "exception": None, "learned_policy_called": False}
    with audit_environment(args.source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        proxy = FrozenRewardReset(env, plan["frozen_reward_distances"])
        observation = proxy.reset(sampled)
        record["reset_gripper_geometry"] = {"tcp": env.gripper.grasp_center().tolist(),
            "table_top": p.getAABB(env.table_id)[1][2],
            "whole_gripper_lowest_aabb_z": min(p.getAABB(env.gripper.body, link)[0][2]
                                               for link in range(-1, p.getNumJoints(env.gripper.body))),
            "pad_lowest_vertex_z": min(v[:, 2].min() for v in env.gripper.world_pad_vertices())}
        closest = [c for body in env.ground_bodies for c in p.getClosestPoints(env.gripper.body, body, distance=2.)]
        contact = min(closest, key=lambda c: c[8])
        record["reset_gripper_geometry"].update(exact_ground_clearance=contact[8],
            nearest_ground_normal=contact[7], geometry_floor_tcp_z=float(env.gripper.grasp_center()[2]-contact[8]+.0035))
        for camera in ("front", "wrist"):
            Image.fromarray(getattr(observation, camera+"_rgb")).save(args.output/f"reset_{camera}.png")
        factory = NeighbourClearanceTeacher
        if args.teacher == "adaptive_neighbour":
            from VLA.adaptive_clearance_teacher import AdaptiveClearanceTeacher
            factory = AdaptiveClearanceTeacher
        if args.teacher == "open_verified":
            from VLA.obstacle_aware_touch_teacher import OpenVerifiedTouchTeacher
            factory = OpenVerifiedTouchTeacher
        if args.teacher == "staged_yaw_clear_touch":
            from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher
            factory = YawClearTouchTeacher
        policy = factory(proxy)
        try:
            policy.reset(sampled)
        except Exception as exc:
            record["exception"] = {"type": type(exc).__name__, "message": str(exc), "step": "reset"}
            record["reset_geometry"] = {name: p.getAABB(body) for name, body in env.object_ids.items()}
            record["expert_plan"] = getattr(policy, "plan", None)
            record["collision_mesh_local"] = np.asarray(p.getMeshData(
                env.pick_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)[1]).tolist()
            record["target_pose"] = p.getBasePositionAndOrientation(env.pick_id)
            (args.output/"reproduction.json").write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps({"task": sampled.prompt, "exception": record["exception"]}, indent=2))
            return
        for step in range(sampled.max_steps):
            lo, hi = p.getAABB(env.pick_id)
            frame = {"step": step, "phase": policy.phase, "target_aabb": [lo, hi],
                     "tcp": env.gripper.grasp_center().tolist(), "held": env.gripper.grasped_object,
                     "bilateral": env.gripper._bilateral(env.pick_id),
                     "table_top": p.getAABB(env.table_id)[1][2], "action": None}
            try:
                action = policy.predict(observation)
                frame["action"] = action.tolist()
            except Exception as exc:
                frame["exception"] = str(exc)
                record["frames"].append(frame)
                record["exception"] = {"type": type(exc).__name__, "message": str(exc), "step": step}
                record["touch_proposals"] = getattr(policy, "touch_proposals", None)
                record["exception_geometry"] = {name: p.getAABB(body) for name, body in env.object_ids.items()}
                record["gripper_link_boxes"] = [p.getAABB(env.gripper.body, link) for link in
                                                range(-1, p.getNumJoints(env.gripper.body))]
                record["gripper_pad_vertices"] = [v.tolist() for v in env.gripper.world_pad_vertices()]
                record["surface_world"] = (np.asarray(p.getMatrixFromQuaternion(
                    p.getBasePositionAndOrientation(env.pick_id)[1])).reshape(3, 3)
                    @policy.touch_surface_local+p.getBasePositionAndOrientation(env.pick_id)[0]).tolist()
                for camera in ("front", "wrist"):
                    Image.fromarray(getattr(observation, camera+"_rgb")).save(args.output/f"exception_{camera}.png")
                break
            record["frames"].append(frame)
            observation, _, done, info = env.step(action)
            if done:
                record["terminal_info"] = info
                break
    (args.output/"reproduction.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps({"task": sampled.prompt, "exception": record["exception"], "last_frames": record["frames"][-4:]}, indent=2))


if __name__ == "__main__":
    main()
