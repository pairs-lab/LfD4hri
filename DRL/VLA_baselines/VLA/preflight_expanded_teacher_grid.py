"""Predeclared all-target/four-task physics preflight while collection runs.

Replays the SAME seeded scene declaration, using first19 scenes per primitive
(one per original target) and its scheduled closed-hover variants. Small RGB
is diagnostic only, NEVER finetuning data or a native-camera certificate.
No failed scene is skipped or silently retried. Four independent simulators.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from config import PICK_TARGETS
from VLA.collect_demos import collect_episode
from VLA.collect_expanded_v9_candidate import FrozenRewardReset, stratified_names
from VLA.common.provenance import sha256_file
from VLA.common.tasks import randomize_rl_scene, select_tasks, TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.obstacle_aware_touch_teacher import OpenVerifiedTouchTeacher
from VLA.offline_closed_hover_recovery import ClosedHoverRecoveryReset


def declared_grid(plan):
    if plan["expert_teacher"] not in {"staged_open_obstacle_touch", "staged_yaw_clear_touch", "staged_central_section_clearance", "staged_pose_revalidated_acquisition"} or plan.get("final_test_used") is not False:
        raise ValueError("Use the exact new TRAIN teacher declaration")
    if plan["episodes_per_task"] < len(PICK_TARGETS):
        raise ValueError("Need a declared scene for every original target")
    rng, tasks = np.random.default_rng(plan["seed"]), {}
    for task in select_tasks("all"):
        selected = []
        for number in range(plan["episodes_per_task"]):
            pick, place = stratified_names(task.name, number)
            sampled = randomize_rl_scene(task, rng, forced_pick_name=pick, forced_place_name=place)
            if number >= len(PICK_TARGETS):
                continue
            ordinary = {"file": f"{task.name}_{number:05d}.npz", "task": sampled.to_dict(), "closed_hover": False}
            selected.append(ordinary)
            every = plan.get("closed_hover_recovery_every", 0)
            if every and (number+1)%every == 0:
                selected.append({**ordinary, "file": f"{task.name}_{number:05d}_closedhover.npz", "closed_hover": True})
        tasks[task.name] = selected
    return tasks


def worker(tokens):
    name, jobs, plan, folder, protected = tokens
    folder = Path(folder)
    folder.mkdir()
    rows = []
    teacher = OpenVerifiedTouchTeacher
    if plan["expert_teacher"] == "staged_yaw_clear_touch":
        from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher
        teacher = YawClearTouchTeacher
    if plan["expert_teacher"] == "staged_central_section_clearance":
        from VLA.central_section_clearance_teacher import CentralSectionClearanceTeacher
        teacher = CentralSectionClearanceTeacher
    if plan["expert_teacher"] == "staged_pose_revalidated_acquisition":
        from VLA.pose_revalidated_acquisition_teacher import PoseRevalidatedAcquisitionTeacher
        teacher = PoseRevalidatedAcquisitionTeacher
    with audit_environment(plan["contract_source"], image_size=32, required_success_steps=1,
                           use_semantic_success=False) as env:
        frozen = FrozenRewardReset(env, plan["frozen_reward_distances"])
        for job in jobs:
            if any(sha256_file(path) != digest for path, digest in protected.items()):
                raise RuntimeError("Declared collection/teacher changed during physics preflight")
            proxy = ClosedHoverRecoveryReset(frozen) if job["closed_hover"] else frozen
            arrays, meta = collect_episode(proxy, TaskSpec.from_dict(job["task"]), policy_factory=teacher)
            meta.update(file=job["file"], closed_hover=job["closed_hover"], diagnostic_rgb_size=32,
                        diagnostic_is_training_data=False)
            np.savez_compressed(folder/job["file"], **arrays)
            rows.append(meta)
            (folder/"progress.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(name, len(rows), "/", len(jobs), job["file"], meta["success"], meta.get("safety_violation"), flush=True)
            if not meta["success"] or meta.get("safety_violation"):
                return {"task": name, "passed": False, "planned": len(jobs), "completed": len(rows),
                        "failed_file": job["file"], "rows": rows}
    return {"task": name, "passed": True, "planned": len(jobs), "completed": len(rows), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan_path = args.collection/"collection_plan.json"
    plan = json.loads(plan_path.read_text())
    groups = declared_grid(plan)
    protected = {str(plan_path.resolve()): sha256_file(plan_path),
                 str(Path(__file__).resolve()): sha256_file(Path(__file__)), **plan["collection_sources_sha256"]}
    if any(sha256_file(path) != digest for path, digest in protected.items()):
        raise ValueError("Preflight differs from the actual frozen collector implementation")
    args.output.mkdir(parents=True)
    declaration = {"scope": "ALL-target four-task TRAIN physics preflight_NOT_VLA_performance_or_training_data",
        "source_collection": str(args.collection.resolve()), "protected_hashes": protected,
        "selection": "first19 declared independent scenes per task, plus every scheduled recovery; no outcome selection",
        "planned_trials": sum(len(jobs) for jobs in groups.values()), "workers": 4,
        "diagnostic_rgb_size": 32, "runtime_camera_pose_controller_success_changed": False,
        "final_test_used": False, "training_performed": False, "jobs": groups}
    (args.output/"plan.json").write_text(json.dumps(declaration, indent=2)+"\n")
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, (name, jobs, plan, str(args.output.resolve()/name), protected))
                   for name, jobs in groups.items()]
        for future in as_completed(futures):
            results.append(future.result())
            (args.output/"progress.json").write_text(json.dumps(results, indent=2)+"\n")
    passed = all(r["passed"] for r in results)
    (args.output/"results.json").write_text(json.dumps({**declaration, "passed": passed, "results": results}, indent=2)+"\n")
    if not passed:
        raise RuntimeError("Exact declared preflight failures preserved; diagnose, never filter or promote")


if __name__ == "__main__":
    main()
