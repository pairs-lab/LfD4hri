"""Audit frozen weights then4 learned diagnostic trials after the adapter fit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT,gpu_state
from VLA.recover_interrupted_cycles import SMOL_FINAL,pid_alive


def run(pilot,primary,output):
    output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+6*3600
    while True:
        state=json.loads((pilot/"status.json").read_text())
        if state.get("status")=="failed":raise ValueError("Adapter pilot failed; no substituted evaluation")
        if (pilot/"checkpoint_ready.json").exists():break
        if not pid_alive(state["pid"]):raise ValueError("Pilot owner disappeared without a saved checkpoint receipt")
        atomic_json(output/"progress.json",{"status":"await_saved1000_adapter","pid":os.getpid(),"completed_trials":0,"planned_trials":4})
        if time.monotonic()>deadline:raise TimeoutError("No adapter save within bounded wait")
        time.sleep(15)
    cp=Path(json.loads((pilot/"checkpoint_ready.json").read_text())["checkpoint"])
    for tag in ("projection_smoke", "projection_train"):
        candidate=ROOT/"VLA/smolvla_simstate"/pilot.name/tag/"checkpoints"/("000004" if tag=="projection_smoke" else "004000")/"pretrained_model"
        with (output/(tag+"_full_weights.log")).open("x") as log:
            p=subprocess.run([sys.executable,"-m","VLA.smolvla.audit_projection_adapter_weights","--base",str(SMOL_FINAL),
                "--checkpoint",str(candidate),"--output",str(output/(tag+"_full_weights.json"))],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if p.returncode:raise ValueError("Frozen weights changed; do not claim projection-only adaptation")
    child=json.loads((primary/"active_child.json").read_text());gpu=gpu_state()
    if set(gpu["compute_pids"])-{str(child["pid"])} or gpu["free_mib"]<3500:
        raise ValueError("No qualified diagnostic GPU headroom")
    import torch
    torch.cuda.set_per_process_memory_fraction(3000*1024**2/torch.cuda.get_device_properties(0).total_memory,device=0)
    from VLA.diagnose_simstate_partial import evaluate
    # The same model/raw state18 has a different declared finetuning strategy.
    # No angle or controller correction is silently combined with it.
    evaluate(pilot,output,cp,1000,angle_alias=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot",type=Path,required=True);parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--launch",action="store_true")
    args=parser.parse_args();output=args.output.resolve()
    if args.launch:
        marker=output.with_name(output.name+"_launch.json")
        if marker.exists() or output.exists():raise FileExistsError("Never reroll a completed adapter diagnostic")
        with output.with_name(output.name+"_driver.log").open("x") as log:
            p=subprocess.Popen([sys.executable,"-m","VLA.watch_projection_adapter_pilot","--pilot",str(args.pilot.resolve()),
                "--primary",str(args.primary.resolve()),"--output",str(output)],cwd=ROOT,env={**os.environ,"OMP_NUM_THREADS":"2",
                    "MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","PYTHONUNBUFFERED":"1"},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":p.pid,"planned_diagnostic_trials":4,"frozen_weights_full_audit_required":True});print(p.pid)
    else:
        try:
            run(args.pilot.resolve(),args.primary.resolve(),output)
        except Exception as error:
            progress = json.loads((output/"progress.json").read_text()) if (output/"progress.json").exists() else {}
            atomic_json(output/"progress.json",{**progress,"status":"preserved_error_no_reroll",
                "error":f"{type(error).__name__}: {error}","pid":os.getpid()})
            raise
