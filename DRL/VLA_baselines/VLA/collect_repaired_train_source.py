"""Replace ALL frozen TRAIN touch labels; preserve ALL other TRAIN bytes.

No validation/test labels, scene filtering, scene rerolls or qualification.
The first failed corrective case is persisted and stops collection.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import traceback
import numpy as np
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection,SurfaceLimitedTouchTeacher
from VLA.frozen_reset_rounding import compare_frozen_reset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--touch-preflight",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--completed-prefix",type=Path,help="Preserve every verified completed case from a stopped older collector; no success-subset selection")
    args=parser.parse_args()
    source=args.train.resolve(); validation=args.validation.resolve(); seed=args.touch_preflight.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    original_manifest=json.loads((source/"dataset_manifest.json").read_text())
    original_index=json.loads((source/"index.json").read_text())
    val_index=json.loads((validation/"index.json").read_text())
    if (original_manifest.get("split")!="train" or not original_index
            or {r["scene_group"] for r in original_index}&{r["scene_group"] for r in val_index}):
        raise ValueError("Frozen TRAIN-only source with zero outer-validation overlap required")
    preflight=json.loads((seed/"report.json").read_text())
    if not preflight["passed"] or preflight["planned"]!=19 or preflight["validation_used"]:
        raise ValueError("All19-object TRAIN preflight required; no selected-success subset")
    code={name:sha256_file(Path(__file__).with_name(name)) for name in
        (Path(__file__).name,"semantic_touch_teacher.py","surface_limited_touch_teacher.py",
         "robust_touch_teacher.py","collect_touch_replay_correction.py")}
    if any(code.get(name)!=digest for name,digest in preflight["code_sha256"].items() if name in code):
        raise ValueError("Preflight touch protocol implementation changed")
    seeds={r["correction_origin"]["file"]:r for r in json.loads((seed/"index.json").read_text())}
    prefix_rows=[]
    prefix_binding=None
    if args.completed_prefix:
        previous=args.completed_prefix.resolve()
        previous_plan=json.loads((previous/"repair_plan.json").read_text())
        previous_state=json.loads((previous/"collection_state.json").read_text())
        prefix_rows=json.loads((previous/"index.json").read_text())
        if (previous_state["status"]!="stopped_exception"
                or previous_state["completed_episodes"]!=len(prefix_rows)
                or previous_state["index_sha256"]!=sha256_file(previous/"index.json")
                or previous_plan["train_index_sha256"]!=sha256_file(source/"index.json")
                or previous_plan["outer_validation_index_sha256"]!=sha256_file(validation/"index.json")
                or [r["file"] for r in prefix_rows]!=[r["file"] for r in original_index[:len(prefix_rows)]]
                or any(not r["success"] or r.get("safety_violation") for r in prefix_rows)):
            raise ValueError("Require the ENTIRE verified completed prefix, not a filtered successful subset")
        prefix_binding={"source":str(previous),"index_sha256":sha256_file(previous/"index.json"),
            "repair_plan_sha256":sha256_file(previous/"repair_plan.json"),"state_sha256":sha256_file(previous/"collection_state.json"),
            "episodes":len(prefix_rows),"reason":"previous collector rejected machine-roundoff in FK aperture; failure evidence remains immutable"}
    prefix_by_file={r["file"]:r for r in prefix_rows}
    binding={"train_source":str(source),"train_index_sha256":sha256_file(source/"index.json"),
        "train_manifest_sha256":sha256_file(source/"dataset_manifest.json"),
        "outer_validation":str(validation),"outer_validation_index_sha256":sha256_file(validation/"index.json"),
        "touch_preflight_report_sha256":sha256_file(seed/"report.json"),"code_sha256":code,
        "completed_prefix":prefix_binding,"reset_roundoff_helper_sha256":sha256_file(Path(__file__).with_name("frozen_reset_rounding.py")),
        "environment_runtime_fingerprint":environment_runtime_fingerprint(original_manifest["observation_profile"])}
    args.output.mkdir(parents=True)
    plan={**binding,"scope":"ALL690 frozen TRAIN scenes; replace ALL168 touch protocols, preserve ALL522 non-touch payloads",
        "planned_counts":dict(Counter(r["task"]["name"] for r in original_index)),
        "training_authorized":False,"validation_labels_used":False,"final_test_used":False,
        "protocol":"original strict success AND near-surface gap<=0.2mm; NOT force-verified physical touch",
        "independent_layouts_added":0}
    (args.output/"repair_plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    collected=[]; new_count=seed_count=preserved=0; reused=0
    def commit(state,error=None):
        (args.output/"index.json").write_text(json.dumps(collected,indent=2)+"\n")
        manifest=copy.deepcopy(original_manifest)
        for key in ("split","readiness_report","readiness_report_sha256","training_qualification","merge_components",
                    "independent_layouts","collection_plan_sha256"):
            manifest.pop(key,None)
        manifest.update(candidate_only=True,training_certified=False,stratified_target_vocabulary=False,
            purpose="ALL frozen TRAIN corrective source; no outer validation used; NOT qualified data",
            episodes=len(collected),frames=sum(r["steps"] for r in collected),episodes_per_task=None,
            declared_task_counts=dict(Counter(r["task"]["name"] for r in collected)),
            index_sha256=sha256_file(args.output/"index.json"),repair_plan_sha256=sha256_file(args.output/"repair_plan.json"))
        (args.output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        (args.output/"collection_state.json").write_text(json.dumps({"status":state,"planned_episodes":len(original_index),
            "completed_episodes":len(collected),"new_touch_episodes":new_count,"seeded_touch_episodes":seed_count,
            "unchanged_non_touch_episodes":preserved,"index_sha256":manifest["index_sha256"],
            "reused_verified_prefix_episodes":reused,
            "planned_counts":plan["planned_counts"],"training_authorized":False,"error":error},indent=2)+"\n")
    commit("in_progress")
    try:
        for old in original_index:
            filename=old["file"]
            original_path=source/filename
            digest=sha256_file(original_path)
            if filename in prefix_by_file:
                accepted=prefix_by_file[filename]
                if accepted["task"]!=old["task"] or accepted["scene_group"]!=old["scene_group"]:
                    raise ValueError("Completed prefix scene/task changed")
                previous_path=previous/filename
                if old["task"]["name"]=="touch":
                    with np.load(previous_path,allow_pickle=False) as arrays,np.load(original_path,allow_pickle=False) as original:
                        reset_parity=compare_frozen_reset(arrays,original)
                    if accepted["expert_touch_terminal_semantics"]!="original_strict_AND_surface_gap_le_0.2mm":
                        raise ValueError("Completed prefix touch protocol changed")
                else:
                    if sha256_file(previous_path)!=digest:
                        raise ValueError("Completed prefix non-touch payload differs")
                    preserved+=1
                os.link(previous_path,args.output/filename)
                meta={**accepted,"completed_prefix_origin":{"source":str(previous),"file":filename,"sha256":sha256_file(previous_path)}}
                reused+=1
            elif old["task"]["name"]!="touch":
                os.link(original_path,args.output/filename)
                meta={**old,"repair_payload_origin":{"source":str(source),"file":filename,"sha256":digest},
                      "non_touch_payload_unchanged":True}
                preserved+=1
            else:
                origin=old["split_origin"]
                accepted=seeds.get(origin["file"])
                if accepted is not None:
                    if (accepted["task"]!=old["task"] or accepted["correction_origin"]["sha256"]!=digest
                            or not accepted["success"] or accepted.get("safety_violation")):
                        raise ValueError("Preflight is not the exact same frozen TRAIN case")
                    os.link(seed/accepted["file"],args.output/filename)
                    meta={**old,**accepted,"file":filename,"repair_payload_origin":{
                        "source":str(seed),"file":accepted["file"],"sha256":sha256_file(seed/accepted["file"])}}
                    seed_count+=1
                else:
                    with audit_environment(source,image_size=384,required_success_steps=1,use_semantic_success=False) as env:
                        proxy=NearSurfaceTouchCollection(env,source,old)
                        arrays,fresh=collect_episode(proxy,TaskSpec.from_dict(old["task"]),policy_factory=SurfaceLimitedTouchTeacher)
                        with np.load(original_path,allow_pickle=False) as original:
                            reset_parity=compare_frozen_reset(arrays,original)
                        fresh.update(expert_touch_terminal_semantics="original_strict_AND_surface_gap_le_0.2mm",
                            expert_terminal_surface_gap_m=proxy.last_surface_gap,
                            expert_terminal_max_normal_force_n=proxy.last_contact_force,
                            expert_quality_failure=proxy.last_quality_failure)
                    np.savez_compressed(args.output/filename,**arrays)
                    meta={**old,**fresh,"file":filename,"correction_origin":{
                        "source":str(source),"file":filename,"sha256":digest},
                        "reset_pixels_and_feedback_exact":reset_parity["all_fields_bitwise_exact"],"reset_rounding_parity":reset_parity}
                    new_count+=1
            if sha256_file(original_path)!=digest:
                raise RuntimeError("Frozen original payload changed")
            collected.append(meta)
            if not meta["success"] or meta.get("safety_violation"):
                commit("stopped_preserved_failure",{"file":filename,"metadata":meta})
                raise RuntimeError("Same failed TRAIN scene retained; no skipped/replaced scene")
            if len(collected)%10==0 or meta["task"]["name"]=="touch":
                commit("in_progress")
                print(json.dumps({"completed":len(collected),"planned":len(original_index),
                    "file":filename,"task":meta["task"]["name"],"steps":meta["steps"],"success":meta["success"]}),flush=True)
        if (sha256_file(source/"index.json")!=binding["train_index_sha256"]
                or sha256_file(source/"dataset_manifest.json")!=binding["train_manifest_sha256"]
                or sha256_file(validation/"index.json")!=binding["outer_validation_index_sha256"]
                or environment_runtime_fingerprint(original_manifest["observation_profile"])!=binding["environment_runtime_fingerprint"]
                or sha256_file(Path(__file__).with_name("frozen_reset_rounding.py"))!=binding["reset_roundoff_helper_sha256"]
                or (prefix_binding is not None and any(sha256_file(previous/name)!=prefix_binding[key] for name,key in
                    (("index.json","index_sha256"),("repair_plan.json","repair_plan_sha256"),("collection_state.json","state_sha256"))))
                or any(sha256_file(Path(__file__).with_name(name))!=digest for name,digest in code.items())):
            raise RuntimeError("Frozen source, outer validation, runtime or repair code changed")
        commit("completed_candidate_not_qualified")
    except Exception:
        if json.loads((args.output/"collection_state.json").read_text())["status"]=="in_progress":
            commit("stopped_exception",{"traceback":traceback.format_exc(),"next_file":original_index[len(collected)]["file"] if len(collected)<len(original_index) else None})
        raise


if __name__=="__main__":
    main()
