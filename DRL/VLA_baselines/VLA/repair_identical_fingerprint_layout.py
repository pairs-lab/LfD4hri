"""Repair ONLY idempotently rewritten metadata mtimes, not changed payloads.

Every checkpoint byte must still match the preserved full-content fingerprint.
Only three finalizer JSON mtimes may differ. Old receipt is backed up outside
the checkpoint; weights/processor/stats files are never written or replaced.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.fingerprint_checkpoint import layout_records,verify_fingerprint

ALLOWED={"fp32_master_contract.json","vla_checkpoint_manifest.json","simstate_contract.json"}


def repair(checkpoint,output):
    if output.exists():raise FileExistsError("Preserve every layout repair receipt")
    old_path=checkpoint/"benchmark_fingerprint.json"
    old=json.loads(old_path.read_text());actual=layout_records(checkpoint)
    indexed={r["path"]:r for r in actual}
    if set(indexed)!={r["path"] for r in old["records"]}:raise ValueError("Files added/removed; not a metadata-mtime repair")
    changed=[]
    for row in old["records"]:
        if indexed[row["path"]]["bytes"]!=row["bytes"] or sha256_file(checkpoint/row["path"])!=row["sha256"]:
            raise ValueError("A payload CHANGED; never overwrite its fingerprint")
        if indexed[row["path"]]["mtime_ns"]!=row["mtime_ns"]:
            if row["path"] not in ALLOWED:raise ValueError("Non-finalizer file mtime changed")
            changed.append(row["path"])
    if not changed:raise ValueError("Nothing requires metadata-mtime repair")
    output.mkdir(parents=True)
    (output/"previous_benchmark_fingerprint.json").write_bytes(old_path.read_bytes())
    with (output/"refresh.log").open("x") as log:
        result=subprocess.run([sys.executable,"-m","VLA.fingerprint_checkpoint",str(checkpoint),"--overwrite"],
            stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode:raise RuntimeError("Layout refresh failed; old receipt preserved")
    new=verify_fingerprint(checkpoint,full=True)
    if new["content_tree_sha256"]!=old["content_tree_sha256"]:raise ValueError("Full contents unexpectedly changed")
    atomic_json(output/"audit.json",{"passed":True,"checkpoint":str(checkpoint),
        "only_metadata_mtimes_changed":changed,"payloads_changed":[],"weights_and_processors_never_written":True,
        "old_full_content_fingerprint_preserved":True,"content_tree_sha256_UNCHANGED":new["content_tree_sha256"]})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();repair(args.checkpoint.resolve(),args.output.resolve())
