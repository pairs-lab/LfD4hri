"""Frozen TRAIN-only all-object/four-task teacher preflight, NOT qualification.

Every predeclared case is evaluated, including failures. No neural outcomes,
validation labels, rerolls, easier replacement scenes, or training are used.
"""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import traceback

import numpy as np
from config import PICK_TARGETS
from VLA.collect_demos import collect_episode
from VLA.collect_paired_grounding_candidate import FrozenRetargetReset
from VLA.collect_touch_replay_correction import ForceVerifiedTouchCollection
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.geometry_clearance_teacher import GeometryClearanceTeacher
from VLA.robust_touch_teacher import GentleForceTouchTeacher

TASKS = ("touch", "pick", "move", "pick_place")


def select_cases(index, declaration, source):
    source = str(Path(source).resolve())
    allowed = {r["file"]:r for r in declaration if r["source"] == source}
    selected = []
    for task in TASKS:
        seen = set()
        for row in index:
            name = row["task"]["pick_name"]
            if (row["file"] not in allowed or row["task"]["name"] != task or name in seen
                    or row.get("paired_scene_source") or row.get("recovery_start_actions_not_training_labels")):
                continue
            selected.append((row, allowed[row["file"]]))
            seen.add(name)
        if seen != set(PICK_TARGETS):
            raise ValueError(f"Missing ordinary TRAIN objects for {task}: {set(PICK_TARGETS)-seen}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((source/"index.json").read_text())
    declaration = json.loads(args.train_episodes.read_text())
    selected = select_cases(index, declaration, source)
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    runtime = environment_runtime_fingerprint(manifest["observation_profile"])
    binding = {"source":str(source), "source_index_sha256":sha256_file(source/"index.json"),
        "source_manifest_sha256":sha256_file(source/"dataset_manifest.json"),
        "train_declaration_sha256":sha256_file(args.train_episodes),
        "collector_sha256":sha256_file(Path(__file__)),
        "touch_teacher_sha256":sha256_file(Path(__file__).with_name("robust_touch_teacher.py")),
        "touch_quality_wrapper_sha256":sha256_file(Path(__file__).with_name("collect_touch_replay_correction.py")),
        "environment_runtime_fingerprint":runtime}
    for row, permitted in selected:
        if sha256_file(source/row["file"]) != permitted["sha256"]:
            raise ValueError("Frozen TRAIN payload changed before preflight")
    args.output.mkdir(parents=True)
    plan = {**binding, "scope":"all-object four-task reused-TRAIN-scene expert preflight ONLY",
        "validation_used":False, "final_test_used":False, "training_authorized":False,
        "selection":"first ordinary TRAIN row per object/task in original source order",
        "planned_cases":[{"file":r["file"], "task":r["task"]["name"],
                          "pick_name":r["task"]["pick_name"],"sha256":d["sha256"]} for r,d in selected]}
    (args.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    reports, collected = [], []
    for number, (row, permitted) in enumerate(selected):
        result = {"original_file":row["file"], "task":row["task"]["name"],
            "pick_name":row["task"]["pick_name"], "original_payload_sha256":permitted["sha256"]}
        try:
            with audit_environment(source,image_size=384,required_success_steps=1,use_semantic_success=False) as env:
                touch = row["task"]["name"] == "touch"
                proxy = (ForceVerifiedTouchCollection if touch else FrozenRetargetReset)(env,source,row)
                arrays, meta = collect_episode(proxy,TaskSpec.from_dict(row["task"]),
                    policy_factory=GentleForceTouchTeacher if touch else GeometryClearanceTeacher)
                if touch:
                    meta.update(expert_terminal_max_normal_force_n=proxy.last_contact_force,
                        expert_quality_failure=proxy.last_quality_failure,
                        expert_required_positive_contact_ticks=2,expert_terminal_force_quality_cap_n=5.)
                with np.load(source/row["file"],allow_pickle=False) as original:
                    reset_exact = all(np.array_equal(arrays[key][0],original[key][0]) for key in
                        ("front_rgb","wrist_rgb","joint_positions","gripper_closure"))
                if not reset_exact:
                    raise RuntimeError("Reset pixels/feedback changed")
                filename = f"{number:03d}_{row['task']['name']}.npz"
                np.savez_compressed(args.output/filename,**arrays)
                # Ordinary task is unchanged: direct TaskSpec reset reproduces
                # original spawn order. Do not invent a counterfactual variant.
                meta.update(file=filename,candidate_only=True,correction_origin={
                    "source":str(source),"file":row["file"],"sha256":permitted["sha256"]})
                collected.append(meta)
                result.update(file=filename,success=meta["success"],steps=meta["steps"],
                    safety_violation=meta["safety_violation"],reset_exact=reset_exact,
                    terminal_force_n=meta.get("expert_terminal_max_normal_force_n"),
                    quality_failure=meta.get("expert_quality_failure"))
        except Exception:
            result.update(success=False,collection_exception=traceback.format_exc())
        reports.append(result)
        (args.output/"index.json").write_text(json.dumps(collected,indent=2)+"\n")
        (args.output/"progress.json").write_text(json.dumps({"completed":len(reports),
            "planned":len(selected),"reports":reports,"training_authorized":False},indent=2)+"\n")
        print(json.dumps(result),flush=True)
    if (sha256_file(source/"index.json") != binding["source_index_sha256"]
            or sha256_file(source/"dataset_manifest.json") != binding["source_manifest_sha256"]
            or environment_runtime_fingerprint(manifest["observation_profile"]) != runtime):
        raise RuntimeError("Frozen parent/runtime changed")
    candidate = copy.deepcopy(manifest)
    for key in ("merge_components","training_qualification","split","collection_plan_sha256",
                "comparison_validation","independent_layouts"):
        candidate.pop(key,None)
    candidate.update(candidate_only=True,training_certified=False,stratified_target_vocabulary=False,
        purpose="Frozen TRAIN-only expert preflight; all failures preserved; NOT training qualification",
        episodes=len(collected),frames=sum(r["steps"] for r in collected),episodes_per_task=None,
        declared_task_counts=dict(Counter(r["task"]["name"] for r in collected)),
        index_sha256=sha256_file(args.output/"index.json"),preflight_plan_sha256=sha256_file(args.output/"plan.json"))
    (args.output/"dataset_manifest.json").write_text(json.dumps(candidate,indent=2)+"\n")
    summary = {task:{"planned":sum(r["task"]==task for r in reports),
        "successes":sum(r["task"]==task and r["success"] for r in reports)} for task in TASKS}
    report = {**plan,"summary":summary,"reports":reports,"all_planned_cases_executed":True,
        "passed":all(r["success"] and not r.get("safety_violation") for r in reports)}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"summary":summary}),flush=True)


if __name__ == "__main__":
    main()
