"""Offline diagnosis of a committed TRAIN candidate, never a policy fallback.

Replay the exact recorded reset and retain contact/pose/action evidence. A
teacher run is a separate experiment, not a replacement of the failed payload.
"""
import argparse
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

from VLA.collect_demos import collect_episode
from VLA.common.physics_safety import surface_clearance
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def physical_state(env):
    position, quaternion = p.getBasePositionAndOrientation(env.pick_id)
    contacts = p.getContactPoints(bodyA=env.gripper.body)
    return {
        "tcp": np.asarray(env.gripper.grasp_center()).tolist(),
        "object_position": list(position), "object_quaternion": list(quaternion),
        "measured_closure": float(env.normalized_gripper_closure()),
        "surface_clearance_m": float(surface_clearance(env)),
        "pad_gap_m": float(env.gripper.width()),
        "held": env.gripper.grasped_object,
        "contacts": [{"body": c[2], "link": c[3], "gap_m": c[8],
                       "force_n": c[9], "normal": list(c[7])} for c in contacts],
    }


class TraceProxy:
    def __init__(self, env, source, row):
        self.env, self.source, self.row = env, source, row
        self.trace = []
        self.teacher = None
        self.recorder = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        return reset_source_episode(self.env, self.source, self.row)

    def step(self, action):
        if self.recorder:
            self.recorder.add(self.env.observe(), "OFFLINE demonstration audit | " + self.row["task"]["prompt"])
        before = physical_state(self.env)
        phase = getattr(self.teacher, "phase", None)
        result = self.env.step(action)
        self.trace.append({"frame": len(self.trace), "phase": phase,
            "action": np.asarray(action).tolist(), "before": before,
            "after": physical_state(self.env), "success": bool(result[3]["success"]),
            "safety_violation": result[3].get("safety_violation"),
            "safety_diagnostics_before_rollback": result[3].get("safety_diagnostics_before_rollback")})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", help="Separate teacher candidate as module:Class; omit for saved-action replay")
    parser.add_argument("--original-collection-reset", action="store_true",
                        help="Recompute the original declared closed-hover reset, then require exact saved sensors")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    plan = json.loads((source / "collection_plan.json").read_text())
    if plan.get("final_test_used") is not False or "TRAIN" not in plan.get("scope", ""):
        raise ValueError("This instrument is restricted to declared TRAIN collection")
    row = next(r for r in json.loads((source / "index.json").read_text()) if r["file"] == args.episode)
    protected = [source / "index.json", source / "collection_state.json",
                 source / "dataset_manifest.json", source / args.episode, Path(__file__)]
    protected += [Path(p) for p in plan["collection_sources_sha256"]]
    hashes = {str(p): sha256_file(p) for p in protected}
    args.output.mkdir(parents=True)
    with np.load(source / args.episode, allow_pickle=False) as a:
        reference = {k: a[k] for k in a.files}
    with audit_environment(source, image_size=384, required_success_steps=1,
                           use_semantic_success=False) as env:
        proxy = TraceProxy(env, source, row)
        if args.video:
            proxy.recorder = VideoRecorder(args.output / "rollout.mp4", fps=10,
                                            overview_render=lambda: env.render("overview"))
        if args.original_collection_reset:
            from VLA.collect_expanded_v9_candidate import FrozenRewardReset
            from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset
            from VLA.common.success_contract_snapshot import recorded_distance_thresholds
            base = FrozenRewardReset(env, {row["task"]["name"]: recorded_distance_thresholds(row)})
            resetter = ClosedHoverRecoveryReset(base) if row.get("recovery_start_actions_not_training_labels") else base
            proxy.reset = resetter.reset
        task = TaskSpec.from_dict(row["task"])
        observation = proxy.reset(task)
        reset_errors = {k: float(np.max(np.abs(reference[k][0].astype(np.float64)-np.asarray(value, np.float64)))) for k, value in {
            "front_rgb": observation.front_rgb, "wrist_rgb": observation.wrist_rgb,
            "joint_positions": observation.joint_positions,
            "gripper_closure": env.normalized_gripper_closure()}.items()}
        reset_exact = all(v == 0 for v in reset_errors.values())
        reset_policy_inputs_exact = (all(reset_errors[k] == 0 for k in ("front_rgb", "wrist_rgb", "joint_positions"))
            and np.float32(reference["gripper_closure"][0]) == np.float32(env.normalized_gripper_closure()))
        if not reset_policy_inputs_exact:
            (args.output / "reset_failure.json").write_text(json.dumps(reset_errors, indent=2)+"\n")
            print(json.dumps(reset_errors), flush=True)
            raise ValueError("Recorded initial RGB/joints/encoder did not reproduce exactly")
        if args.teacher:
            module, name = args.teacher.split(":")
            cls = getattr(importlib.import_module(module), name)
            hashes[str(Path(importlib.import_module(module).__file__).resolve())] = sha256_file(
                Path(importlib.import_module(module).__file__))
            def factory(wrapper):
                wrapper.teacher = cls(wrapper)
                return wrapper.teacher
            arrays, meta = collect_episode(proxy, task, policy_factory=factory)
            plan_result = proxy.teacher.plan
            replan_events = getattr(proxy.teacher, "replan_events", [])
        else:
            for action in reference["canonical_actions"]:
                _, _, done, info = proxy.step(action)
                if done:
                    break
            arrays, plan_result = reference, None
            replan_events = []
            meta = {"success": bool(info["success"]), "steps": len(proxy.trace),
                    "safety_violation": info.get("safety_violation")}
        for camera in ("front", "wrist"):
            Image.fromarray(env.render(camera)).save(args.output / f"terminal_{camera}.png")
        if proxy.recorder:
            proxy.recorder.add(env.observe(), "OFFLINE ONLY | success=" + str(meta["success"]))
            proxy.recorder.close()
    from VLA.audit_preserved_touch_failure import preserved_reset_metadata
    meta.update(preserved_reset_metadata(row))
    meta.update(file="candidate.npz", candidate_only=True)
    np.savez_compressed(args.output / "candidate.npz", **arrays)
    (args.output / "index.json").write_text(json.dumps([meta], indent=2)+"\n")
    manifest = copy.deepcopy(json.loads((source / "dataset_manifest.json").read_text()))
    for key in ("collection_plan_sha256", "training_qualification", "split"):
        manifest.pop(key, None)
    manifest.update(episodes=1, frames=meta["steps"], declared_task_counts={row["task"]["name"]: 1},
                    training_certified=False, index_sha256=sha256_file(args.output / "index.json"))
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    frames = sorted(set([0, max(0, len(arrays["front_rgb"])-10),
                         max(0, len(arrays["front_rgb"])-8), len(arrays["front_rgb"])-1]))
    sheet = Image.new("RGB", (768, 408 * len(frames)), "white")
    draw = ImageDraw.Draw(sheet)
    for number, frame in enumerate(frames):
        for col, camera in enumerate(("front", "wrist")):
            image = Image.fromarray(arrays[camera + "_rgb"][frame])
            image.save(args.output / f"frame{frame:03d}_{camera}.png")
            sheet.paste(image, (col * 384, number * 408 + 24))
            draw.text((col * 384, number * 408), f"{camera} pre-action frame {frame}", fill="black")
    sheet.save(args.output / "contact_sheet.png")
    report = {"scope": "OFFLINE TRAIN exact contact diagnosis NOT VLA SR",
        "source": str(source), "episode": args.episode, "task": row["task"],
        "source_index_sha256": hashes[str(source / "index.json")],
        "source_state_sha256": hashes[str(source / "collection_state.json")],
        "source_payload_sha256": hashes[str(source / args.episode)],
        "reset_exact": reset_exact, "reset_policy_inputs_exact": bool(reset_policy_inputs_exact),
        "reset_errors": reset_errors, "teacher": args.teacher, "teacher_plan": plan_result,
        "replan_events": replan_events,
        "passed": bool(meta["success"] and not meta.get("safety_violation") and
                       (row["task"]["name"] == "touch" or arrays["next_verified_grasp"][-1])),
        "metadata": meta, "trace": proxy.trace, "protected_hashes": hashes,
        "runtime_changed": False, "training_performed": False, "final_test_used": False}
    report.update(pybullet_binary_sha256=sha256_file(Path(p.__file__)),
                  collection_pybullet_binary_matches=sha256_file(Path(p.__file__)) == plan["pybullet_binary_sha256"])
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(sha256_file(Path(p)) != h for p, h in hashes.items()):
        raise RuntimeError("Source/runtime changed during diagnostic")
    print(json.dumps({"reset_exact": reset_exact, "metadata": meta, "teacher_plan": plan_result}), flush=True)


if __name__ == "__main__":
    main()
