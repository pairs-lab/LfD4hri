"""Compare canonical/live-FK/isolated-FK on one immutable source trajectory.

Diagnostic only: no runtime, dataset, success or training-gate changes.
Isolated FK tests whether querying targets by resetting the live robot perturbs
contact/constraint solver state. It is not an oracle or learned fallback.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pybullet as p
from VLA.common.provenance import sha256_file
from VLA.common.native_action_fields import native_gripper_closure
from VLA.dataset_replay_context import audit_environment, reset_source_episode


class IsolatedFK:
    def __init__(self, env):
        self.env = env
        self.client = p.connect(p.DIRECT)
        self.robot = p.loadURDF(str(env.robot_spec.urdf), useFixedBase=True, physicsClientId=self.client)

    def close(self):
        p.disconnect(self.client)

    def action(self, targets, closure):
        state = p.getLinkState(self.env.robot_id, self.env.robot_spec.mount_link,
                              computeForwardKinematics=True, physicsClientId=self.env.client_id)
        for joint, value in zip(self.env.robot_spec.arm_joint_ids, targets, strict=True):
            p.resetJointState(self.robot, joint, float(value), physicsClientId=self.client)
        target = p.getLinkState(self.robot, self.env.robot_spec.mount_link,
                               computeForwardKinematics=True, physicsClientId=self.client)
        translation = np.asarray(target[4], float)-np.asarray(state[4], float)
        inverse = p.invertTransform((0, 0, 0), state[5])[1]
        relative = p.multiplyTransforms((0, 0, 0), inverse, (0, 0, 0), target[5])[1]
        angles = np.asarray(p.getEulerFromQuaternion(relative))
        angles = (angles+np.pi)%(2*np.pi)-np.pi
        return np.r_[np.clip(translation, -.05, .05), np.clip(angles, -.25, .25),
                     2*float(closure)-1].astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-after-success", action="store_true",
                        help="Offline terminal contact inspection only; never continue safety aborts/timeouts")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source = args.source.resolve()
    item = next(row for row in json.loads((source/"index.json").read_text()) if row["file"] == args.episode)
    path = source/item["file"]
    digest = sha256_file(path)
    reports = []
    with np.load(path, allow_pickle=False) as arrays:
        closures = native_gripper_closure(arrays)
        for mode in ("canonical", "live_fk", "isolated_fk"):
            frames = []
            with audit_environment(source, image_size=384, render_observations=False,
                                   required_success_steps=1, use_semantic_success=False) as env:
                reset_source_episode(env, source, item)
                fk = IsolatedFK(env) if mode == "isolated_fk" else None
                try:
                    for t, reference in enumerate(arrays["canonical_actions"]):
                        target = arrays["next_joint_targets"][t]
                        closure = float(closures[t])
                        action = (reference if mode == "canonical" else
                                  fk.action(target, closure) if fk else env.joint_targets_to_canonical(target, closure))
                        before = env.observe()
                        position = p.getBasePositionAndOrientation(env.pick_id)[0]
                        _, _, done, info = env.step(action)
                        closest = p.getClosestPoints(env.gripper.body, env.pick_id, distance=.02)
                        contacts = p.getContactPoints(env.gripper.body, env.pick_id)
                        frames.append({"frame": t, "action_error": (action-reference).tolist(),
                            "state_joint_max_error": float(np.max(np.abs(before.joint_positions-arrays["joint_positions"][t]))),
                            "tcp_before": before.ee_position.tolist(), "object_before": list(position),
                            "touching_after": info["touching_object"], "strict_after": info["strict_raw_success"],
                            "semantic_after": info["semantic_raw_success"], "ee_object_after": info["ee_object_distance"],
                            "object_after": list(p.getBasePositionAndOrientation(env.pick_id)[0]),
                            "closest_gripper_object_surface_after_m": min((c[8] for c in closest), default=None),
                            "contact_normal_forces_after_n": [c[9] for c in contacts],
                            "safety_violation": info.get("safety_violation")})
                        if t in (0, 15, 30, len(arrays["canonical_actions"])-1) or done:
                            for camera in ("front", "wrist"):
                                Image.fromarray(env.render(camera)).save(args.output/f"{mode}_{t:03d}_{camera}.png")
                        if done and (not args.continue_after_success or info.get("safety_violation")
                                     or env.step_count >= env.task.max_steps):
                            break
                finally:
                    if fk:
                        fk.close()
            reports.append({"mode": mode, "strict_success": any(r["strict_after"] for r in frames),
                "semantic_success": any(r["semantic_after"] for r in frames), "frames": frames})
            print(json.dumps({k:v for k,v in reports[-1].items() if k != "frames"}), flush=True)
    if sha256_file(path) != digest:
        raise RuntimeError("Source payload changed")
    (args.output/"diagnosis.json").write_text(json.dumps({"scope": "Single TRAIN-source native-control diagnosis; NOT corrected data or qualification",
        "source": str(source), "episode": args.episode, "payload_sha256": digest, "task": item["task"],
        "continued_after_first_hit_for_terminal_inspection": args.continue_after_success,
        "runtime_or_dataset_changed": False, "reports": reports}, indent=2)+"\n")


if __name__ == "__main__":
    main()
