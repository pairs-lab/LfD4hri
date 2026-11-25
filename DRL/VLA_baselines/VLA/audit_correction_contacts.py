"""Replay a preserved correction's labels and inspect physical contacts; no model."""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    item = next(row for row in json.loads((args.source / "index.json").read_text()) if row["file"] == args.episode)
    path = args.source / args.episode
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as arrays:
        actions = arrays["canonical_actions"].copy()
    rows = []
    with audit_environment(args.source, image_size=384, render_observations=False,
                           required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
        reset_source_episode(env, args.source, item)
        names = {body: name for name, body in env.object_ids.items()}
        names[env.table_id] = "table"
        for t, action in enumerate(actions):
            _, _, done, info = env.step(action)
            if action[6] > 0 or t in (0, 10, len(actions)-1):
                contacts = p.getContactPoints(bodyA=env.gripper.body)
                tcp = env.gripper.grasp_center()
                target = p.getBasePositionAndOrientation(env.pick_id)
                aabb = p.getAABB(env.pick_id)
                rows.append({"frame": t, "command": action.tolist(), "tcp_m": np.asarray(tcp).tolist(),
                    "target_position_m": list(target[0]), "target_aabb": aabb,
                    "jaw_gap_m": env.gripper.width(), "measured_closure": env.normalized_gripper_closure(),
                    "bilateral": env.gripper._bilateral(env.pick_id), "held_body": names.get(env.gripper.grasped_object),
                    "contacts": [{"other": names.get(c[2], str(c[2])), "gripper_link": c[3], "other_link": c[4],
                        "on_gripper_world_m": c[5], "on_other_world_m": c[6], "normal_force_N": c[9], "distance_m": c[8]}
                        for c in contacts]})
            if done:
                break
    if sha256_file(path) != digest:
        raise RuntimeError("Preserved correction payload changed")
    report = {"scope": "preserved expert-label physical contact replay, not learned results",
              "source": str(args.source.resolve()), "file": args.episode, "payload_sha256": digest,
              "terminal_success": info["success"], "safety_violation": info.get("safety_violation"), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
