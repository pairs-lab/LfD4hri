"""Exact stopped uncommitted TRAIN scene: offline repair/video/replay data."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.collect_demos import collect_episode
from VLA.collect_expanded_v9_candidate import FrozenRewardReset, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.dataset_replay_context import audit_environment
from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset
from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher


class TracedScene(FrozenRewardReset):
    def __init__(self, env, thresholds, recorder, reproduction_folder, check_normal):
        super().__init__(env, thresholds)
        self.recorder, self.teacher = recorder, None
        self.trace, self.reset_checks = [], {}
        self.reproduction_folder, self.check_normal = reproduction_folder, check_normal

    def reset(self, task):
        observation = super().reset(task)
        if self.check_normal:
            for camera in ("front", "wrist"):
                old = np.asarray(Image.open(self.reproduction_folder/f"reset_{camera}.png"))
                actual = getattr(observation, camera+"_rgb")
                self.reset_checks[camera] = int(np.max(np.abs(actual.astype(int)-old.astype(int))))
            if any(self.reset_checks.values()):
                raise ValueError("Reset pixels differ from exact original exception")
        return observation

    def step(self, action):
        self.recorder.add(self.env.observe(), "OFFLINE TEACHER REPAIR | "+self.env.task.prompt)
        _, reward, done, info = result = self.env.step(action)
        self.trace.append({"frame": len(self.trace), "phase": self.teacher.phase,
            "action": np.asarray(action).tolist(), "pad_index": self.teacher.touch_pad_index,
            "centre": self.env.gripper.grasp_center().tolist(),
            "closure": self.env.normalized_gripper_closure(),
            "success": info["success"], "safety_violation": info.get("safety_violation")})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reproduction", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--closed-hover", action="store_true", help="Separate recovery diagnostic, not original normal reset")
    parser.add_argument("--normal-repair", type=Path, help="Required canonical normal repair for recovery replay")
    parser.add_argument("--teacher", choices=("yaw_clear", "central_section_clearance"), default="yaw_clear")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    plan = json.loads((source/"collection_plan.json").read_text())
    state = json.loads((source/"collection_state.json").read_text())
    reproduction = json.loads(args.reproduction.read_text())
    if args.closed_hover:
        if args.normal_repair is None:
            raise ValueError("Recovery needs an immutable repaired normal export")
        normal = json.loads((args.normal_repair/"results.json").read_text())
        if (normal.get("passed") is not True or normal.get("reset_exact") is not True
                or Path(normal["source"]).resolve() != source or normal["episode"] != state["file"]
                or normal["source_state_sha256"] != sha256_file(source/"collection_state.json")
                or normal["teacher"] != args.teacher):
            raise ValueError("Recovery canonical normal repair differs from the stopped scene")
    if (not manifest.get("candidate_only") or manifest.get("final_test_used")
            or state["status"] != "stopped_on_collection_exception"
            or reproduction["state_sha256"] != sha256_file(source/"collection_state.json")
            or reproduction["index_sha256"] != sha256_file(source/"index.json")
            or not reproduction["exception"]
            or reproduction["exception"]["message"] != state["message"]):
        raise ValueError("Exact preserved TRAIN exception evidence required")
    task_name, number = state["file"].removesuffix(".npz").rsplit("_", 1)
    if task_name not in ("touch", "pick", "move", "pick_place") or not number.isdigit():
        raise ValueError("Normal scene exception only; no guessed recovery reset")
    base = next(task for task in select_tasks("all") if task.name == task_name)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    pick, place = stratified_names(task_name, int(number))
    task = randomize_rl_scene(base, rng, forced_pick_name=pick, forced_place_name=place)
    if (json.loads(json.dumps(task.to_dict())) != state["task"]
            or json.loads(json.dumps(task.to_dict())) != reproduction["task"]
            or rng.bit_generator.state != state["rng_after_sampling"]):
        raise ValueError("Failed scene/RNG differs; no reroll")
    protected = [source/"index.json", source/"dataset_manifest.json", source/"collection_state.json",
        args.reproduction, args.reproduction.parent/"reset_front.png", args.reproduction.parent/"reset_wrist.png",
        Path(__file__), Path(__file__).with_name("yaw_clear_touch_teacher.py"),
        Path(__file__).with_name("obstacle_aware_touch_teacher.py")]
    if args.teacher == "central_section_clearance":
        protected.extend([Path(__file__).with_name("central_section_clearance_teacher.py"),
                          Path(__file__).with_name("ray_refined_clearance_teacher.py")])
    hashes = {str(path.resolve()): sha256_file(path) for path in protected}
    if args.closed_hover:
        for name in ("index.json", "dataset_manifest.json", "results.json", state["file"]):
            path = (args.normal_repair/name).resolve()
            hashes[str(path)] = sha256_file(path)
    verify_environment_runtime_fingerprint(manifest["environment_runtime_fingerprint"], manifest["observation_profile"])
    output.mkdir(parents=True)
    with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
        recorder = VideoRecorder(output/"rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
        proxy = TracedScene(env, plan["frozen_reward_distances"], recorder, args.reproduction.parent, not args.closed_hover)
        resetter = ClosedHoverRecoveryReset(proxy) if args.closed_hover else proxy
        def factory(wrapper):
            teacher_type = YawClearTouchTeacher
            if args.teacher == "central_section_clearance":
                from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
                teacher_type = CentralSectionClearanceTeacher
            proxy.teacher = teacher_type(wrapper)
            return proxy.teacher
        try:
            arrays, meta = collect_episode(resetter, task, policy_factory=factory)
            recorder.add(env.observe(), "TEACHER ONLY | success="+str(meta["success"]))
        finally:
            recorder.close()
    filename = state["file"].removesuffix(".npz")+("_closedhover" if args.closed_hover else "")+".npz"
    meta.update(file=filename, candidate_only=True)
    if args.closed_hover:
        meta.update(variant=0, canonical_source=str(args.normal_repair.resolve()), paired_scene_source=state["file"],
            canonical_source_index_sha256=sha256_file(args.normal_repair/"index.json"),
            paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
            recovery_start_actions_not_training_labels=[a.tolist() for a in resetter.startup_actions],
            recovery_start_kind=resetter.startup_evidence["kind"], recovery_start_evidence=resetter.startup_evidence)
    np.savez_compressed(output/filename, **arrays)
    (output/"index.json").write_text(json.dumps([meta], indent=2)+"\n")
    new_manifest = copy.deepcopy(manifest)
    new_manifest.update(episodes=1, frames=meta["steps"], declared_task_counts={task_name: 1}, training_certified=False,
        index_sha256=sha256_file(output/"index.json"))
    new_manifest.pop("collection_plan_sha256", None)
    (output/"dataset_manifest.json").write_text(json.dumps(new_manifest, indent=2)+"\n")
    for frame in sorted(set([0, 17, min(40, meta["steps"]-1), meta["steps"]-1])):
        for camera in ("front", "wrist"):
            Image.fromarray(arrays[camera+"_rgb"][frame]).save(output/f"frame{frame:03d}_{camera}.png")
    report = {"scope": "Same uncommitted TRAIN scene teacher repair_NOT_neural_performance",
        "source": str(source), "episode": filename, "task": json.loads(json.dumps(task.to_dict())),
        "source_index_sha256": sha256_file(source/"index.json"),
        "source_state_sha256": sha256_file(source/"collection_state.json"),
        "reproduction_sha256": sha256_file(args.reproduction), "protected_hashes": hashes,
        "metadata": meta, "reset_exact": bool(not args.closed_hover and not any(proxy.reset_checks.values())),
        "reset_checks": proxy.reset_checks, "teacher": args.teacher, "trace": proxy.trace,
        "touch_proposals": getattr(proxy.teacher, "touch_proposals", None),
        "expert_plan": proxy.teacher.plan,
        "passed": bool(meta["success"] and not meta.get("safety_violation")),
        "runtime_changed": False, "training_performed": False, "final_test_used": False}
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    if any(sha256_file(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("Preserved source/evidence/code changed")
    print(json.dumps({"passed": report["passed"], "steps": meta["steps"],
        "safety": meta.get("safety_violation"), "reset_exact": report["reset_exact"]}))


if __name__ == "__main__":
    main()
