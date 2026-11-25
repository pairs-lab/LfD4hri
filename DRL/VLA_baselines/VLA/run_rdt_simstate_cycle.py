"""Qualified privileged RDT5000 -> saved precision -> learned120 -> audit.

Different input regime from ordinary RDT. One finite cycle, never rerolls
failed episodes or implies adequate learning from numerical qualification.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json
from VLA.recover_interrupted_cycles import pid_alive
from VLA.run_cross_vla120 import ROOT, gpu_state

RDT = ROOT/"VLA/.venvs/rdt/bin/python"
SIM = ROOT/"VLA/third_party/lerobot/.venv/bin/python"
BASE = ROOT/"VLA/rdt/body_v9_recovery_r17_20261013/test1/train/checkpoints/step-005000"
SOURCE = ROOT/"VLA/audits/development40_smolvla_r14_20261013/rdt_qualified_train"
AUDITS = ROOT/"VLA/audits/development40_rdt_r14_20261013"
PROJECTION = ROOT/"VLA/audits/recovery_r17_20261013/train"
QUALIFIED = ROOT/"VLA/audits/rdt_privileged_smoke_r24_20261014/qualified"


def train_command(output):
    return [RDT,"-m","VLA.rdt.train_simstate","--projection",PROJECTION,
        "--simstate-qualification",QUALIFIED/"simstate_qualification.json","train",
        "--source",SOURCE,"--audit-report",AUDITS/"qualified_source_audit/audit.json",
        "--action-contract-report",AUDITS/"qualified_fk.json","--language-embeddings",AUDITS/"qualified_embeddings.pt",
        "--base-checkpoint",BASE,"--vision-checkpoint",ROOT/"VLA/checkpoints/siglip-so400m-patch14-384",
        "--signal-qualification",QUALIFIED/"native_signal_qualification.json","--output",output,
        "--steps","5000","--action-horizon","10","--episode-batch-size","1","--samples-per-episode","4",
        "--gradient-accumulation","4","--learning-rate","0.0001","--weight-decay","0.01",
        "--signal-precision","fp32_state_action_readout_latent_v1","--optimizer","adamw","--num-workers","0",
        "--save-interval","250","--seed","20261010"]


def run(output, primary):
    if output.exists():
        raise FileExistsError("Never duplicate/retry an unchanged finetuning cycle")
    output.mkdir(parents=True)
    # Both guards must match the same actual smoke/config before GPU training.
    from VLA.rdt import training
    from VLA.rdt.train_simstate import require_new_qualification
    from VLA.rdt.signal_qualification import require_signal_qualification
    command = train_command(ROOT/"VLA/rdt_simstate"/output.name/"train")
    saved_argv = sys.argv; sys.argv = [str(RDT), *map(str,command[command.index("train"):])]
    try:
        args = training.parse_args()
    finally:
        sys.argv = saved_argv
    require_new_qualification(args, QUALIFIED/"simstate_qualification.json", PROJECTION)
    require_signal_qualification(args)
    files = [Path(__file__), ROOT/"VLA/rdt/train_simstate.py", ROOT/"VLA/rdt/simstate_server.py",
        ROOT/"VLA/rdt/simstate_adapter.py", ROOT/"VLA/rdt/simstate_inputs.py", ROOT/"VLA/evaluate_rdt_simstate120.py",
        ROOT/"VLA/analyze_rdt_simstate120.py", QUALIFIED/"simstate_qualification.json", QUALIFIED/"native_signal_qualification.json"]
    files += [ROOT/"asset/params/reward.xml", ROOT/"config.py", ROOT/"VLA/simstate_feedback.py",
        ROOT/"VLA/common/environment.py", ROOT/"VLA/common/observation_profile.py", ROOT/"VLA/common/body_environment.py",
        ROOT/"VLA/common/body_gripper.py", ROOT/"VLA/common/physics_safety.py", ROOT/"VLA/common/environment_factory.py",
        ROOT/"VLA/common/spatial.py", ROOT/"VLA/common/tasks.py"]
    frozen = {str(p):sha256_file(p) for p in files}
    history = []; server = None; active = None
    def save(stage,status="running",error=None):
        atomic_json(output/"status.json", {"pid":os.getpid(),"status":status,"stage":stage,"history":history,
            "error":error,"privileged_model_inputs":True,"oracle_controller_at_deployment":False,
            "new_optimizer_updates":5000,"prior_family_optimizer_updates":5000,"planned_trials":120,
            "trials_per_action":30,"frozen_files":frozen,"stable_high_success_NOT_established":True,
            "heartbeat_utc":datetime.now(timezone.utc).isoformat()})
    def check_files():
        if any(sha256_file(Path(p)) != digest for p,digest in frozen.items()):
            raise ValueError("Frozen cycle changed")
    def stage(name,command):
        nonlocal active
        check_files(); save(name)
        with (output/(name+".log")).open("x") as log:
            child = subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            active = child
            atomic_json(output/"active_child.json",{"stage":name,"pid":child.pid,"command":list(map(str,command))})
            deadline = time.monotonic()+36*3600
            while child.poll() is None:
                save(name); time.sleep(10)
                if time.monotonic()>deadline:
                    raise TimeoutError("Bounded stage deadline exceeded; preserve owned child receipt")
        history.append({"stage":name,"exit_code":child.returncode})
        if child.returncode:
            raise RuntimeError(name+" failed; preserve artifacts, no blind retry")
    env = {**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1",
        "TRANSFORMERS_OFFLINE":"1","PYTHONUNBUFFERED":"1","VLA_RDT_SIMSTATE_GPU_MEMORY_LIMIT_MIB":"8500"}
    try:
        child = json.loads((primary/"active_child.json").read_text()); gpu = gpu_state()
        allowed = {str(child["pid"])} if (child["stage"] == "privileged120" and pid_alive(child["pid"])) else set()
        if set(gpu["compute_pids"])-allowed or gpu["free_mib"]<9500:
            raise ValueError("Unknown GPU owner/insufficient bounded training headroom")
        stage("privileged_train",command)
        cp = args.output/"checkpoints/step-005000"
        stage("saved_precision",[RDT,"-m","VLA.rdt.audit_training_precision","--checkpoint",cp,"--updates","5000",
            "--output",output/"saved_precision.json"])
        if json.loads((output/"saved_precision.json").read_text())["optimizer_steps"] != [5000.]:
            raise ValueError("Actual saved counts differ")
        stage("checkpoint_fingerprint",[RDT,"-m","VLA.fingerprint_checkpoint",cp])
        check_files(); save("starting_privileged_server")
        server_command = [RDT,"-m","VLA.rdt.simstate_server","--checkpoint",cp,"--vision",ROOT/"VLA/checkpoints/siglip-so400m-patch14-384",
            "--embeddings",AUDITS/"qualified_embeddings.pt","--port","5734"]
        with (output/"server.log").open("x") as log:
            server = subprocess.Popen(list(map(str,server_command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(output/"server_receipt.json",{"pid":server.pid,"command":list(map(str,server_command))})
        import requests
        deadline = time.monotonic()+900
        while True:
            if server.poll() is not None:
                raise RuntimeError("Privileged server exited; no substitute model")
            try:
                response = requests.get("http://127.0.0.1:5734/metadata",timeout=2)
                if response.ok and Path(response.json()["checkpoint"]["path"]).resolve() == cp.resolve():
                    break
            except (requests.RequestException,KeyError):
                pass
            if time.monotonic()>deadline:
                raise TimeoutError("Owned server did not become ready")
            save("starting_privileged_server"); time.sleep(10)
        stage("privileged120",[SIM,"-m","VLA.evaluate_rdt_simstate120","--checkpoint",cp,
            "--output",output/"development120","--port","5734"])
        stage("failure_analysis",[SIM,"-m","VLA.analyze_rdt_simstate120","--progress",output/"development120/progress.json",
            "--output",output/"failure_audit"])
        save("completed_finite_cycle_video_and_learning_review_required",status="completed")
    except Exception as error:
        save("preserved_error_no_reroll",status="failed",error=f"{type(error).__name__}: {error}"); raise
    finally:
        # Never disconnect a server while an owned neural episode is in flight
        # (e.g. an unexpected bounded-stage timeout). Preserve both receipts.
        if server is not None and server.poll() is None and (active is None or active.poll() is not None):
            server.terminate(); server.wait(timeout=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--primary",type=Path,required=True)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args();output = args.output.resolve()
    if args.launch:
        marker = output.with_name(output.name+"_launch.json")
        if marker.exists() or output.exists():
            raise FileExistsError("Never repeat a registered launch")
        with output.with_name(output.name+"_driver.log").open("x") as log:
            child = subprocess.Popen([str(SIM),"-m","VLA.run_rdt_simstate_cycle","--output",str(output),"--primary",str(args.primary.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":child.pid,"privileged_model_inputs":True,"planned_new_optimizer_updates":5000,"planned_trials":120})
        print(child.pid)
    else:
        run(output,args.primary.resolve())
