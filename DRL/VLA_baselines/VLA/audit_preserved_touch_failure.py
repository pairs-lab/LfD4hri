"""Reproduce/repair the exact preserved TRAIN teacher failure, with video."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image

from VLA.collect_demos import collect_episode
from VLA.collect_preserved_teacher_ablation import PreservedReset
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher
from VLA.obstacle_aware_touch_teacher import ObstacleAwareTouchTeacher, OpenVerifiedTouchTeacher


def preserved_reset_metadata(item):
    # Reset-only provenance must survive export: without these fields a replay
    # would incorrectly start a closed-hover correction from the normal home.
    fields = ("variant", "canonical_source", "canonical_source_index_sha256", "paired_scene_source",
              "paired_reset_strategy", "recovery_start_actions_not_training_labels",
              "recovery_start_kind", "recovery_start_evidence", "scene_group",
              "paired_source_dataset", "paired_collection_source")
    return {key: copy.deepcopy(item[key]) for key in fields if key in item}


class TracedReset(PreservedReset):
    def __init__(self, env, source, item, recorder):
        super().__init__(env, source, item)
        self.recorder, self.trace, self.teacher = recorder, [], None

    def step(self, action):
        self.recorder.add(self.env.observe(), "OFFLINE TEACHER DIAGNOSTIC | "+self.item["task"]["prompt"])
        result = self.env.step(action)
        self.trace.append({"frame": len(self.trace), "phase": self.teacher.phase,
            "action": np.asarray(action).tolist(), "pad_index": getattr(self.teacher, "touch_pad_index", None),
            "centre": self.env.gripper.grasp_center().tolist(),
            "target_gap_m": min((c[8] for c in p.getClosestPoints(self.env.gripper.body, self.env.pick_id, .025)), default=None),
            "contacts": [{"object": name, "gripper_link": c[3], "gap_m": c[8], "force_n": c[9]}
                         for name, body in self.env.object_ids.items()
                         for c in p.getContactPoints(self.env.gripper.body, body)],
            "success": result[3]["success"], "safety_violation": result[3].get("safety_violation")})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--teacher", choices=("original", "obstacle_aware", "open_verified"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    if not manifest.get("candidate_only") or manifest.get("final_test_used"):
        raise ValueError("Preserved development TRAIN only, never final test")
    item = next(r for r in json.loads((source/"index.json").read_text()) if r["file"] == args.episode)
    if item["task"]["name"] != "touch" or item["success"]:
        raise ValueError("Select the preserved failed touch episode")
    protected = (source/"index.json", source/"dataset_manifest.json", source/"collection_state.json",
                 source/item["file"], Path(__file__), Path(__file__).with_name("obstacle_aware_touch_teacher.py"),
                 Path(__file__).with_name("staged_acquisition_teacher.py"))
    hashes = {str(path): sha256_file(path) for path in protected}
    verify_environment_runtime_fingerprint(manifest["environment_runtime_fingerprint"], manifest["observation_profile"])
    output.mkdir(parents=True)
    teacher = {"original": StagedAcquisitionTeacher, "obstacle_aware": ObstacleAwareTouchTeacher,
               "open_verified": OpenVerifiedTouchTeacher}[args.teacher]
    with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
        recorder = VideoRecorder(output/"rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
        proxy = TracedReset(env, source, item, recorder)
        def factory(wrapper):
            policy = teacher(wrapper)
            proxy.teacher = policy
            return policy
        try:
            arrays, meta = collect_episode(proxy, TaskSpec.from_dict(item["task"]), policy_factory=factory)
            recorder.add(env.observe(), "TEACHER ONLY | success="+str(meta["success"]))
        finally:
            recorder.close()
    meta.update(preserved_reset_metadata(item))
    meta.update(file=item["file"], candidate_only=True, exact_reset_checks=proxy.reset_checks)
    np.savez_compressed(output/item["file"], **arrays)
    for frame in sorted(set([0, 17, min(40, len(arrays["front_rgb"])-1), len(arrays["front_rgb"])-1])):
        for camera in ("front", "wrist"):
            Image.fromarray(arrays[camera+"_rgb"][frame]).save(output/f"frame{frame:03d}_{camera}.png")
    with np.load(source/item["file"], allow_pickle=False) as old:
        actions_identical = np.array_equal(arrays["canonical_actions"], old["canonical_actions"])
    report = {"scope": "Exact failed TRAIN teacher diagnosis_NOT_neural_success",
        "source": str(source), "episode": item["file"], "source_index_sha256": hashes[str(source/"index.json")],
        "source_state_sha256": hashes[str(source/"collection_state.json")],
        "source_payload_sha256": hashes[str(source/item["file"])], "protected_hashes": hashes,
        "teacher": args.teacher, "metadata": meta, "trace": proxy.trace,
        "original_actions_identical": actions_identical, "touch_proposals": proxy.teacher.touch_proposals if args.teacher != "original" else None,
        "task": item["task"], "reset_exact": all(v == 0 for v in proxy.reset_checks.values()),
        "passed": bool(meta["success"] and not meta.get("safety_violation")),
        "runtime_changed": False, "training_performed": False, "final_test_used": False}
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    (output/"index.json").write_text(json.dumps([meta], indent=2)+"\n")
    manifest = copy.deepcopy(manifest)
    manifest.update(episodes=1, frames=meta["steps"], declared_task_counts={"touch": 1}, training_certified=False)
    manifest.pop("collection_plan_sha256", None)
    manifest["index_sha256"] = sha256_file(output/"index.json")
    (output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("Preserved source or diagnostic implementation changed")
    print(json.dumps({"passed": report["passed"], "success": meta["success"], "steps": meta["steps"],
                      "safety": meta.get("safety_violation"), "original_actions_identical": actions_identical,
                      "touch_proposals": report["touch_proposals"]}))


if __name__ == "__main__":
    main()
