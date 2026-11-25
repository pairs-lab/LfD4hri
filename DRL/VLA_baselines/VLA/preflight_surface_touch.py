"""Test collision-normal touch expert on ALL19 predeclared ordinary TRAIN objects.

All successes/failures remain in the result. This is not data qualification;
the original robot, timing, scoring, cameras, split and safeguards are frozen.
"""
import argparse
import copy
import json
from pathlib import Path
import traceback
import numpy as np
from VLA.preflight_gentle_teacher import select_cases
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher,SurfaceLimitedTouchCollection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--train-episodes",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    index=json.loads((source/"index.json").read_text())
    declaration=json.loads(args.train_episodes.read_text())
    selected=[(r,d) for r,d in select_cases(index,declaration,source) if r["task"]["name"]=="touch"]
    manifest=json.loads((source/"dataset_manifest.json").read_text())
    binding={"source":str(source),"source_index_sha256":sha256_file(source/"index.json"),
        "source_manifest_sha256":sha256_file(source/"dataset_manifest.json"),
        "train_declaration_sha256":sha256_file(args.train_episodes),
        "environment_runtime_fingerprint":environment_runtime_fingerprint(manifest["observation_profile"]),
        "code_sha256":{name:sha256_file(Path(__file__).with_name(name)) for name in
            (Path(__file__).name,"surface_limited_touch_teacher.py","robust_touch_teacher.py",
             "collect_touch_replay_correction.py","preflight_gentle_teacher.py")}}
    for row,permitted in selected:
        if sha256_file(source/row["file"])!=permitted["sha256"]:
            raise ValueError("Frozen TRAIN payload changed")
    args.output.mkdir(parents=True)
    plan={**binding,"scope":"ALL19-object reused-TRAIN-scene collision-normal touch preflight ONLY",
        "selection":"same first ordinary TRAIN rows as original76-case preflight; no failure filtering",
        "training_authorized":False,"validation_used":False,"final_test_used":False,
        "cases":[{"file":r["file"],"pick_name":r["task"]["pick_name"],"sha256":d["sha256"]} for r,d in selected]}
    (args.output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    rows,reports=[],[]
    for number,(row,permitted) in enumerate(selected):
        report={"original_file":row["file"],"pick_name":row["task"]["pick_name"],
                "original_payload_sha256":permitted["sha256"]}
        try:
            with audit_environment(source,image_size=384,required_success_steps=1,use_semantic_success=False) as env:
                proxy=SurfaceLimitedTouchCollection(env,source,row)
                arrays,meta=collect_episode(proxy,TaskSpec.from_dict(row["task"]),policy_factory=SurfaceLimitedTouchTeacher)
                with np.load(source/row["file"],allow_pickle=False) as original:
                    exact=all(np.array_equal(arrays[k][0],original[k][0]) for k in
                              ("front_rgb","wrist_rgb","joint_positions","gripper_closure"))
                if not exact:
                    raise RuntimeError("Original reset pixels/feedback changed")
                filename=f"touch_{number:03d}.npz"
                np.savez_compressed(args.output/filename,**arrays)
                meta.update(file=filename,candidate_only=True,
                    correction_origin={"source":str(source),"file":row["file"],"sha256":permitted["sha256"]},
                    expert_terminal_force_quality_cap_n=5.,expert_required_positive_contact_ticks=2,
                    expert_terminal_max_normal_force_n=proxy.last_contact_force,
                    expert_quality_failure=proxy.last_quality_failure)
                rows.append(meta)
                report.update(file=filename,success=meta["success"],steps=meta["steps"],reset_exact=exact,
                    safety_violation=meta["safety_violation"],terminal_force_n=proxy.last_contact_force,
                    quality_failure=proxy.last_quality_failure)
        except Exception:
            report.update(success=False,collection_exception=traceback.format_exc())
        reports.append(report)
        (args.output/"index.json").write_text(json.dumps(rows,indent=2)+"\n")
        (args.output/"progress.json").write_text(json.dumps({"completed":len(reports),"planned":len(selected),
            "reports":reports,"training_authorized":False},indent=2)+"\n")
        print(json.dumps(report),flush=True)
    if (sha256_file(source/"index.json")!=binding["source_index_sha256"]
            or sha256_file(source/"dataset_manifest.json")!=binding["source_manifest_sha256"]
            or environment_runtime_fingerprint(manifest["observation_profile"])!=binding["environment_runtime_fingerprint"]
            or any(sha256_file(Path(__file__).with_name(name))!=digest for name,digest in binding["code_sha256"].items())):
        raise RuntimeError("Frozen source/runtime or preflight code changed")
    candidate=copy.deepcopy(manifest)
    for key in ("merge_components","training_qualification","split","collection_plan_sha256",
                "comparison_validation","independent_layouts"):
        candidate.pop(key,None)
    candidate.update(candidate_only=True,training_certified=False,stratified_target_vocabulary=False,
        purpose="ALL19 TRAIN touch expert preflight; failures retained; NOT qualified training data",
        episodes=len(rows),frames=sum(r["steps"] for r in rows),episodes_per_task=None,
        declared_task_counts={"touch":len(rows)},index_sha256=sha256_file(args.output/"index.json"))
    (args.output/"dataset_manifest.json").write_text(json.dumps(candidate,indent=2)+"\n")
    final={**plan,"all_planned_cases_executed":True,"reports":reports,
        "successes":sum(r["success"] for r in reports),"planned":len(reports),
        "passed":all(r["success"] and not r.get("safety_violation") for r in reports)}
    (args.output/"report.json").write_text(json.dumps(final,indent=2)+"\n")
    print(json.dumps({"passed":final["passed"],"successes":final["successes"],"planned":final["planned"]}),flush=True)


if __name__=="__main__":
    main()
