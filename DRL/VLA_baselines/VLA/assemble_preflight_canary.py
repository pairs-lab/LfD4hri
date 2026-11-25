"""Assemble all76 predeclared scenes, replacing ALL19 touch protocols explicitly.

The failed old touch preflight is retained. No successful-only subset selection,
scene replacement, validation use or data/training qualification is permitted.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from VLA.common.provenance import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-task-preflight",type=Path,required=True)
    parser.add_argument("--near-touch-preflight",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base=args.all_task_preflight.resolve(); touch=args.near_touch_preflight.resolve()
    base_report=json.loads((base/"report.json").read_text())
    touch_report=json.loads((touch/"report.json").read_text())
    if (not base_report["all_planned_cases_executed"] or not touch_report["all_planned_cases_executed"]
            or not touch_report["passed"] or len(touch_report["reports"])!=19
            or base_report["environment_runtime_fingerprint"]!=touch_report["environment_runtime_fingerprint"]):
        raise ValueError("Complete, frozen, all-object preflights required")
    base_index=json.loads((base/"index.json").read_text())
    touch_index=json.loads((touch/"index.json").read_text())
    originals={r["correction_origin"]["file"]:r for r in base_index if r["task"]["name"]=="touch"}
    replacements={r["correction_origin"]["file"]:r for r in touch_index}
    if set(originals)!=set(replacements) or len(originals)!=19:
        raise ValueError("Cannot omit or substitute any original touch scene")
    chosen=[]
    for row in base_index:
        parent=base
        if row["task"]["name"]=="touch":
            parent=touch; replacement=replacements[row["correction_origin"]["file"]]
            if replacement["task"]!=row["task"] or replacement["correction_origin"]!=row["correction_origin"]:
                raise ValueError("Touch protocol changed physical scene/task/origin")
            row=replacement
        if not row["success"] or row.get("safety_violation"):
            raise ValueError("Failed case retained in preflight; cannot assemble training-format canary")
        chosen.append((parent,row))
    if len(chosen)!=76:
        raise ValueError("Exactly76 predeclared cases required")
    args.output.mkdir(parents=True)
    index=[]
    for number,(parent,row) in enumerate(chosen):
        filename=f"{number:03d}_{row['task']['name']}.npz"
        os.link(parent/row["file"],args.output/filename)
        index.append({**row,"file":filename,"preflight_data_origin":{
            "source":str(parent),"file":row["file"],"sha256":sha256_file(parent/row["file"])}})
    (args.output/"index.json").write_text(json.dumps(index,indent=2)+"\n")
    manifest=copy.deepcopy(json.loads((base/"dataset_manifest.json").read_text()))
    manifest.update(candidate_only=True,training_certified=False,
        purpose="76-scene four-task training-format/native-camera canary; ALL19 touch protocols replaced, NOT final data qualification",
        episodes=76,frames=sum(r["steps"] for r in index),declared_task_counts={k:19 for k in ("touch","pick","move","pick_place")},
        index_sha256=sha256_file(args.output/"index.json"),
        preflight_reports=[{"path":str(p/"report.json"),"sha256":sha256_file(p/"report.json")} for p in (base,touch)])
    (args.output/"dataset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"episodes":76,"frames":manifest["frames"],"training_authorized":False}),flush=True)


if __name__=="__main__":
    main()
