"""Bounded1000-update goal-projection pilot beside the owned full-state fit.

This is a separately declared finetuning strategy, NOT the selected final
checkpoint or part of the ordinary120 denominators. Long optimization cannot
start without real2-update/new-column/FP32 gates. Same model, inputs, env.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT,gpu_state
from VLA.recover_interrupted_cycles import SMOL_FINAL,SIM,pid_alive
from VLA.run_privileged_priority_cycle import PROJECTION,qualify_existing_data


def run(output,primary):
    output.mkdir(parents=True,exist_ok=False)
    proof=qualify_existing_data()
    batch=ROOT/"VLA/audits/privileged_priority_r21_20261014/fixed_actual_training_batches.json"
    report=json.loads(batch.read_text())
    if not report.get("passed") or report["wrapper_sha256"]!=sha256_file(ROOT/"VLA/smolvla/simstate_dataset.py"):
        raise ValueError("Actual state18 batch dispatch not qualified")
    primary_child=json.loads((primary/"active_child.json").read_text())
    state=json.loads((primary/"status.json").read_text());gpu=gpu_state()
    if (not pid_alive(state["pid"]) or not pid_alive(primary_child["pid"]) or primary_child["stage"]!="simstate_train"
            or set(gpu["compute_pids"])-{str(primary_child["pid"])} or gpu["free_mib"]<4500):
        raise ValueError("No bounded headroom beside the known trainer; do not overlap unknown owners")
    history=[]
    def save(stage,status="running",error=None):
        atomic_json(output/"status.json",{"pid":os.getpid(),"status":status,"stage":stage,"error":error,
            "history":history,"planned_adapter_updates":1000,"primary_fit_UNCHANGED":str(primary),
            "not_final5000_checkpoint_selection":True,"architecture_unchanged":True,
            "oracle_controller_at_deployment":False,"heartbeat_utc":datetime.now(timezone.utc).isoformat()})
    env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
        "PYTHONUNBUFFERED":"1","TOKENIZERS_PARALLELISM":"false","VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR":"0",
        "VLA_SMOLVLA_TARGET_BALANCED":"1","VLA_SMOLVLA_GROUNDING_PROB":"0","VLA_SMOLVLA_INITIAL_FRAME_PROB":".1",
        "VLA_SMOLVLA_SAMPLING_SOURCE":str(ROOT/"VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train"),
        "VLA_SIMSTATE_TRAIN_PROJECTION":str(PROJECTION),"VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB":"4000"}
    def stage(name,command,extra=None,proof_path=None):
        save(name)
        with (output/(name+".log")).open("x") as log:
            p=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**env,**(extra or {})},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/"active_child.json",{"pid":p.pid,"stage":name,"command":list(map(str,command))})
            p.wait(timeout=3600*4)
        history.append({"stage":name,"exit_code":p.returncode})
        if p.returncode:raise RuntimeError(name+" failed; preserve pilot, no unchanged retry")
        if proof_path and json.loads(Path(proof_path).read_text()).get("passed") is not True:raise ValueError("Failed real gate")
    try:
        from VLA.run_simstate_repair_cycle import training_command,finalize_checkpoint
        for label,updates,acc in (("projection_smoke",2,2),("projection_train",1000,4)):
            fit=ROOT/"VLA/smolvla_simstate"/output.name/label
            command=training_command(SMOL_FINAL,PROJECTION,fit,updates,acc)
            command=["VLA.smolvla.train_simstate_projection_adapter" if v=="VLA.smolvla.train_simstate" else v for v in command]
            command=["--policy.optimizer_lr=0.001" if str(v).startswith("--policy.optimizer_lr=") else
                "--policy.scheduler_decay_lr=0.00001" if str(v).startswith("--policy.scheduler_decay_lr=") else
                f"--policy.scheduler_warmup_steps={0 if updates==2 else 50*acc}" if str(v).startswith("--policy.scheduler_warmup_steps=") else v for v in command]
            command += ["--policy.optimizer_weight_decay=0.0"]
            stage(label,command,{"VLA_SIMSTATE_OPTIMIZER_UPDATES":str(updates),"VLA_SMOLVLA_MASTER_AUDIT":str(fit/"master_dtype_audit.json")})
            cp=finalize_checkpoint(fit,updates,acc,PROJECTION)
            stage(label+"_precision",[SIM,"-m","VLA.smolvla.audit_training_precision","--checkpoint",cp,"--output",output/(label+"_precision.json")],
                proof_path=output/(label+"_precision.json"))
            if json.loads((output/(label+"_precision.json")).read_text())["optimizer_steps"]!=[float(updates)]:raise ValueError("Saved Adam count differs")
            if updates==2:
                stage("new_zero_init_columns",[SIM,"-m","VLA.smolvla.audit_simstate_updates","--base",SMOL_FINAL,"--checkpoint",cp,
                    "--output",output/"new_zero_init_columns.json"],proof_path=output/"new_zero_init_columns.json")
            stage(label+"_fingerprint",[SIM,"-m","VLA.fingerprint_checkpoint",cp])
        save("completed1000_projection_pilot_requires_learned_evaluation",status="completed_fit_NOT_success_claim")
        atomic_json(output/"checkpoint_ready.json",{"checkpoint":str(cp),"updates":1000,"privileged_model_inputs":True,
            "all_other_weights_frozen":True,"architecture_changed":False,"learned_success_NOT_established":True})
    except Exception as error:
        save("preserved_error_NO_false_success",status="failed",error=f"{type(error).__name__}: {error}");raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--launch",action="store_true")
    args=parser.parse_args();output=args.output.resolve();primary=args.primary.resolve()
    if args.launch:
        marker=output.with_name(output.name+"_launch.json")
        if output.exists() or marker.exists():raise FileExistsError("No repeated pilot")
        with output.with_name(output.name+"_driver.log").open("x") as log:
            p=subprocess.Popen([sys.executable,"-m","VLA.run_projection_adapter_pilot","--output",str(output),"--primary",str(primary)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":p.pid,"planned_adapter_updates":1000,"new_columns_learning_rate":.001,
            "projection_only_zero_init_strategy":True,"not_primary_checkpoint_selection":True});print(p.pid)
    else:run(output,primary)
