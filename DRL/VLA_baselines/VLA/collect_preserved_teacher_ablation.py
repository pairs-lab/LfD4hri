"""Recollect ONE preserved TRAIN scene with a separate teacher, never filter it."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode


class PreservedReset:
    def __init__(self, env, source, item):
        self.env, self.source, self.item = env, source, item
        self.reset_checks = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        if task != TaskSpec.from_dict(self.item["task"]):
            raise ValueError("Ablation must preserve every body, pose and instruction")
        observation = reset_source_episode(self.env, self.source, self.item)
        with np.load(self.source/self.item["file"], allow_pickle=False) as a:
            errors = {"joints": float(np.max(np.abs(observation.joint_positions-a["joint_positions"][0]))),
                      "closure": abs(self.env.normalized_gripper_closure()-float(a["gripper_closure"][0])),
                      "front": int(np.max(np.abs(observation.front_rgb.astype(int)-a["front_rgb"][0].astype(int)))),
                      "wrist": int(np.max(np.abs(observation.wrist_rgb.astype(int)-a["wrist_rgb"][0].astype(int))))}
        if errors["joints"] > 1e-6 or errors["closure"] > 1e-6 or errors["front"] or errors["wrist"]:
            raise ValueError("Preserved scene reset differs: " + str(errors))
        self.reset_checks = errors
        return observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = json.loads((source/"dataset_manifest.json").read_text())
    if not parent.get("candidate_only") or parent.get("final_test_used"):
        raise ValueError("Use a preserved development TRAIN candidate, never final TEST")
    item = next(x for x in json.loads((source/"index.json").read_text()) if x["file"] == args.episode)
    from VLA.neighbour_clearance_teacher import NeighbourClearanceTeacher
    protected = [source/"index.json", source/"dataset_manifest.json", source/item["file"],
                 Path(__file__), Path(__file__).with_name("neighbour_clearance_teacher.py"),
                 Path(__file__).with_name("clearance_correction_teacher.py"),
                 Path(__file__).with_name("collision_planned_teacher.py")]
    hashes = {str(path): sha256_file(path) for path in protected}
    verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
    output.mkdir(parents=True)
    (output/"ablation_plan.json").write_text(json.dumps({"scope": "Same preserved TRAIN scene, teacher-only ablation",
        "source": str(source), "episode": item["file"], "protected_hashes": hashes,
        "pybullet_binary_sha256": sha256_file(Path(p.__file__)), "scene_or_runtime_changed": False,
        "learned_policy_called": False}, indent=2)+"\n")
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        proxy = PreservedReset(env, source, item)
        arrays, meta = collect_episode(proxy, TaskSpec.from_dict(item["task"]), policy_factory=NeighbourClearanceTeacher)
    meta.update(file=item["file"], candidate_only=True, preserved_scene_source=str(source),
                preserved_scene_payload_sha256=hashes[str(source/item["file"])], exact_reset_checks=proxy.reset_checks)
    np.savez_compressed(output/item["file"], **arrays)
    (output/"index.json").write_text(json.dumps([meta], indent=2)+"\n")
    manifest = copy.deepcopy(parent)
    manifest.update(episodes=1, frames=meta["steps"], episodes_per_task=None,
                    declared_task_counts={meta["task"]["name"]: 1}, training_certified=False,
                    purpose="Preserved failed-scene teacher ablation, NOT qualified training or new independent layout")
    for key in ("index_sha256", "collection_plan_sha256"):
        manifest.pop(key, None)
    (output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("Preserved ablation input/code changed")
    verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
    print(json.dumps(meta, indent=2))
    if not meta["success"] or meta.get("safety_violation"):
        raise RuntimeError("Ablation failed; preserve and diagnose, do not filter")


if __name__ == "__main__":
    main()
