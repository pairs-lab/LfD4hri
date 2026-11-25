"""Preserve and recollect the NEXT deterministic scene after an exception.

No source mutation, scene skipping, validation examples or learned controls.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pybullet as p
from VLA.adaptive_clearance_teacher import AdaptiveClearanceTeacher
from VLA.collect_demos import collect_episode
from VLA.collect_expanded_v9_candidate import FrozenRewardReset, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import select_tasks, randomize_rl_scene
from VLA.dataset_replay_context import audit_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", choices=("adaptive_neighbour", "ray_refined", "geometry_clearance"), default="adaptive_neighbour")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    state = json.loads((source/"collection_state.json").read_text())
    plan = json.loads((source/"collection_plan.json").read_text())
    parent = json.loads((source/"dataset_manifest.json").read_text())
    reproduction = json.loads(args.reproduction.read_text())
    if reproduction["state_sha256"] != sha256_file(source/"collection_state.json") or not reproduction["exception"]:
        raise ValueError("Uncommitted exception reproduction changed")
    rows = json.loads((source/"index.json").read_text())
    if len(rows) != state["completed_episodes"] or sha256_file(source/"index.json") != state["index_sha256"]:
        raise ValueError("Source checkpoint differs")
    completed = {row["file"] for row in rows}
    task, i = next((task, i) for task in select_tasks("all") for i in range(plan["episodes_per_task"])
                   if f"{task.name}_{i:05d}.npz" not in completed)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]
    pick, place = stratified_names(task.name, i)
    sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
    if json.loads(json.dumps(sampled.to_dict())) != reproduction["task"]:
        raise ValueError("Recreated scene differs from preserved exception")
    factory = AdaptiveClearanceTeacher
    if args.teacher == "ray_refined":
        from VLA.ray_refined_clearance_teacher import RayRefinedClearanceTeacher
        factory = RayRefinedClearanceTeacher
    if args.teacher == "geometry_clearance":
        from VLA.geometry_clearance_teacher import GeometryClearanceTeacher
        factory = GeometryClearanceTeacher
    protected = [source/"index.json", source/"dataset_manifest.json", source/"collection_state.json",
                 args.reproduction, Path(__file__), Path(__file__).with_name("adaptive_clearance_teacher.py")]
    if args.teacher == "ray_refined":
        protected.append(Path(__file__).with_name("ray_refined_clearance_teacher.py"))
    if args.teacher == "geometry_clearance":
        protected.extend(Path(__file__).with_name(name) for name in ("geometry_clearance_teacher.py", "ray_refined_clearance_teacher.py"))
    hashes = {str(path.resolve()): sha256_file(path) for path in protected}
    output.mkdir(parents=True)
    (output/"ablation_plan.json").write_text(json.dumps({"source": str(source), "task": sampled.to_dict(),
        "hashes": hashes, "runtime_changed": False, "scene_or_object_changed": False,
        "teacher": args.teacher, "training_certified": False}, indent=2)+"\n")
    verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
    with audit_environment(source, image_size=384, seed=plan["seed"], required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        proxy = FrozenRewardReset(env, plan["frozen_reward_distances"])
        arrays, meta = collect_episode(proxy, sampled, policy_factory=factory)
    filename = f"{task.name}_{i:05d}.npz"
    np.savez_compressed(output/filename, **arrays)
    meta.update(file=filename, candidate_only=True, preserved_uncommitted_source=str(source))
    (output/"index.json").write_text(json.dumps([meta], indent=2)+"\n")
    manifest = copy.deepcopy(parent)
    manifest.update(episodes=1, frames=meta["steps"], episodes_per_task=None,
        declared_task_counts={task.name: 1}, training_certified=False,
        purpose="Preserved uncommitted TRAIN scene ablation, NOT complete or qualified data")
    for key in ("index_sha256", "collection_plan_sha256"):
        manifest.pop(key, None)
    (output/"dataset_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("Ablation source/code changed")
    print(json.dumps({"success": meta["success"], "steps": meta["steps"], "safety": meta["safety_violation"]}))
    if not meta["success"] or meta["safety_violation"]:
        raise RuntimeError("Preserved ablation failed; do not skip it")


if __name__ == "__main__":
    main()
