"""Actual partial checkpoint schema/normalizer gate; NOT fit qualification."""
import argparse
import json
import os
from pathlib import Path
import shutil

from VLA.common.provenance import sha256_file,build_directory_fingerprint
from VLA.cross_vla120_trial import atomic_json
from VLA.recover_interrupted_cycles import SOURCE,SMOL_SAVED
from VLA.smolvla.training import validate_checkpoint_schema
from VLA.smolvla.audit_checkpoint_predictions import validate_saved_normalization


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve metadata audit evidence")
    cp=args.checkpoint.resolve()
    config=json.loads((cp.parent/"training_state/training_step.json").read_text())
    info=json.loads((SOURCE/"lerobot_dataset/meta/info.json").read_text())
    schema=validate_checkpoint_schema(cp.parents[2],config["step"],info,5,str(SMOL_SAVED),True)
    target=args.output/"UNQUALIFIED_partial_snapshot"
    shutil.copytree(cp,target,copy_function=os.link)
    lineage={**schema,"checkpoint":str(target.resolve()),"source":str(SOURCE/"qualified_train"),
        "source_index_sha256":sha256_file(SOURCE/"qualified_train/index.json"),
        "source_manifest_sha256":sha256_file(SOURCE/"qualified_train/dataset_manifest.json"),
        "frames":116257,"episodes":2107,"dataset_root":str(SOURCE/"lerobot_dataset"),
        "stats_sha256":sha256_file(SOURCE/"lerobot_dataset/meta/stats.json"),
        "dataset_info_sha256":sha256_file(SOURCE/"lerobot_dataset/meta/info.json"),
        "partial_only_NOT_full_fit_or_task_success":True}
    atomic_json(target/"vla_checkpoint_manifest.json",lineage)
    identity=build_directory_fingerprint(target)
    audit=json.loads((SOURCE/"qualified_train_audit/audit.json").read_text())
    normalization=validate_saved_normalization(target.resolve(),SOURCE/"qualified_train",audit,identity)
    atomic_json(args.output/"audit.json",{"passed":True,"scope":"Actual saved partial checkpoint metadata/schema/normalizer gate ONLY",
        "original_checkpoint":str(cp),"saved_microsteps":config["step"],"saved_optimizer_updates":config["step"]//16,
        "normalization":normalization,"unchanged_source_and_original_checkpoint":True,
        "new_lineage_fields_checked":["frames","dataset_root","stats_sha256","dataset_info_sha256"],
        "qualified_for_deployment":False,"full_optimizer_updates_required":5000,"neural_success_established":False})


if __name__=="__main__":
    main()
