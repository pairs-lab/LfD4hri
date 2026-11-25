"""Detached, locked recovery of interrupted owned fits and causal-data audits.

No existing trainer is signalled. Saved counters, not stale log percentages,
are authoritative. Each lane preserves evidence and errors independently.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"VLA/audits/recovery_r17_20261013"
FIRST_RECOVERY = OUTPUT
SIM = ROOT/"VLA/third_party/lerobot/.venv/bin/python"
RDT = ROOT/"VLA/.venvs/rdt/bin/python"
SOURCE = ROOT/"VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013"
SMOL_OLD = ROOT/"VLA/audits/development40_smolvla_r16_phase_share_20261013"
RDT_OLD = ROOT/"VLA/audits/development40_rdt_r16_verified_guard_20261013"
SMOL_TRAIN = ROOT/"VLA/smolvla/body_v9_r16_phase_share_20261013/test1/train"
SMOL_SAVED = SMOL_TRAIN/"checkpoints/016000/pretrained_model"
SMOL_FINAL = SMOL_TRAIN/"checkpoints/080000/pretrained_model"
RDT_SAVED = ROOT/"VLA/rdt/body_v9_staged_r16_verified_guard_20261013/test1/train/checkpoints/step-001000"
RDT_TRAIN = ROOT/"VLA/rdt/body_v9_recovery_r17_20261013/test1/train"
RDT_FINAL = RDT_TRAIN/"checkpoints/step-005000"


def pid_alive(pid):
    if type(pid) is not int or pid < 1: return False
    path = Path(f"/proc/{pid}/stat")
    if not path.exists(): return False
    try: return path.read_text().rsplit(")",1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError): return False


def verify_frozen(state):
    if any(sha256_file(Path(p)) != digest for p,digest in state["frozen_files"].items()):
        raise ValueError("Old qualified recipe/source changed; no unsupported restart")


def launch(adopt=False,handoff=None):
    if OUTPUT.exists(): raise FileExistsError("Recovery evidence is immutable; do not launch duplicates")
    previous = {}
    for name, folder in (("smolvla",SMOL_OLD),("rdt",RDT_OLD),("simstate",ROOT/"VLA/audits/simstate18_full_repair_r1_20261013")):
        state = json.loads((folder/"status.json").read_text())
        if pid_alive(state["pid"]): raise ValueError("Previous owned process is still alive: "+name)
        verify_frozen(state)
        previous[name] = state
    active = subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"], text=True).strip()
    adopted = None
    if handoff:
        state_path=Path(handoff).resolve()/"status.json"
        adopted=json.loads(state_path.read_text())
        if adopted["stage"]!="parallel_saved_fit_recovery_and_CPU_replay" or adopted["status"]!="running":
            raise ValueError("Handoff is only allowed in the known passive lane wait")
        live=adopted["lanes"]["rdt_continue"]
        if not live["alive"] or not pid_alive(live["pid"]): raise ValueError("No live RDT continuation to adopt")
        proc=Path(f"/proc/{adopted['pid']}")
        tokens=(proc/"cmdline").read_bytes().split(b"\0")
        if b"VLA.recover_interrupted_cycles" not in tokens or str(Path(handoff).resolve()).encode() not in tokens or (proc/"cwd").resolve()!=ROOT:
            raise ValueError("Coordinator identity differs; do not signal it")
        children=(proc/"task"/str(adopted["pid"])/"children").read_text().split()
        if children!=[str(live["pid"])]: raise ValueError("Unexpected live child; do not hand off")
        child_cmd=(Path(f"/proc/{live['pid']}")/"cmdline").read_bytes().split(b"\0")
        if b"VLA.rdt.continue_interrupted_training" not in child_cmd or str(RDT_TRAIN).encode() not in child_cmd:
            raise ValueError("Child is not the declared RDT trainer")
        first=json.loads((FIRST_RECOVERY/"status.json").read_text())
        permitted={str(live["pid"]),str(first["lanes"]["smol_resume"]["pid"])}
        if set(active.splitlines())-permitted: raise ValueError("Unexpected GPU compute owner")
    elif adopt:
        adopted = json.loads((FIRST_RECOVERY/"status.json").read_text())
        failed = adopted["lanes"].get("rdt_continue",{})
        if failed.get("exit_code") != 1 or RDT_TRAIN.exists():
            raise ValueError("Only the diagnosed PRE-training horizon-binding failure may be repaired")
        error_log = (FIRST_RECOVERY/"rdt_continue.log").read_text()
        if "RDT current lineage differs from saved checkpoint" not in error_log:
            raise ValueError("Unrecognized failure; do not launch unchanged training")
        permitted = {str(adopted["lanes"]["smol_resume"]["pid"])}
        if set(active.splitlines())-permitted:
            raise ValueError("Unrelated GPU compute owner; do not overlap")
    elif active:
        raise ValueError("GPU has an existing compute owner; inspect before recovery")
    OUTPUT.mkdir(parents=True)
    for name,state in previous.items(): atomic_json(OUTPUT/f"previous_{name}_status.json",state)
    atomic_json(OUTPUT/"plan.json", {"previous": previous, "planned_saved_total_updates_per_fit":5000,
        "saved_start_updates_smolvla":1000,"saved_start_updates_rdt":1000,
        "old_unsaved_log_updates_are_NOT_adopted":True,"signals_sent":0,
        "same_robot_objects_controller_cameras_success":True,"privileged_input_scores_are_separate":True,
        "ordinary120_per_model_budget_preserved":True,"rdt_final":str(RDT_FINAL),"smolvla_final":str(SMOL_FINAL),
        "adopt_existing_live_lanes_without_signals":adopt,"adopted_cycle_snapshot":adopted,
        "handoff_previous_coordinator":str(Path(handoff).resolve()) if handoff else None,
        "adopted_live_rdt_pid":adopted["lanes"]["rdt_continue"]["pid"] if handoff else None,
        "diagnosed_fix": "Apply trainer.main action_horizon override64->10 before exact lineage comparison" if adopt else None})
    if handoff:
        # Signal ONLY the positively identified passive coordinator, NOT its
        # process group or the live RDT/Smol/physics workers. They keep their
        # own inherited file descriptors and receipts.
        os.kill(adopted["pid"],signal.SIGTERM)
        deadline=time.monotonic()+15
        while pid_alive(adopted["pid"]):
            if time.monotonic()>deadline: raise TimeoutError("Old passive coordinator did not stop")
            time.sleep(.1)
    with (OUTPUT/"driver.log").open("x") as stream:
        process = subprocess.Popen([str(SIM),"-m","VLA.recover_interrupted_cycles","--run","--output",str(OUTPUT)]+(["--adopt-existing"] if adopt else []),cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic_json(OUTPUT/"launch.json",{"pid":process.pid,"detached_session":True,"stdout_bound_to_file":True,
        "started_at_utc":datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"launched_pid":process.pid,"output":str(OUTPUT)},indent=2))


def finalize_smol():
    from safetensors import safe_open
    from VLA.smolvla.training import validate_checkpoint_schema
    master = json.loads((SMOL_TRAIN/"master_dtype_audit.json").read_text())
    if not master.get("passed"): raise ValueError("SmolVLA masters failed")
    with safe_open(SMOL_FINAL/"model.safetensors",framework="np") as weights:
        if any(weights.get_slice(n).get_dtype() != "F32" for n in master["trainable_parameter_names"]):
            raise ValueError("Saved learned weights lost FP32")
    with safe_open(SMOL_FINAL.parent/"training_state/optimizer_state.safetensors",framework="np") as moments:
        counts = {float(moments.get_tensor(n)) for n in moments.keys() if n.endswith("/step")}
        if counts != {5000.}: raise ValueError("Restored optimizer did not reach exactly5000 saved updates")
        if any(moments.get_slice(n).get_dtype() != "F32" for n in moments.keys() if "exp_avg" in n):
            raise ValueError("Saved moments lost FP32")
    info = json.loads((SOURCE/"lerobot_dataset/meta/info.json").read_text())
    schema = validate_checkpoint_schema(SMOL_TRAIN,80000,info,5,str(SMOL_SAVED),True)
    atomic_json(SMOL_FINAL/"fp32_master_contract.json",{**master,"saved_weights_and_adam_moments_fp32":True})
    atomic_json(SMOL_FINAL/"vla_checkpoint_manifest.json",{**schema,"source":str(SOURCE/"qualified_train"),
        "source_manifest_sha256":sha256_file(SOURCE/"qualified_train/dataset_manifest.json"),
        "source_index_sha256":sha256_file(SOURCE/"qualified_train/index.json"),
        "steps":80000,"optimizer_updates":5000,"continued_from_checkpoint":str(SMOL_SAVED),
        "restored_saved_optimizer_updates":1000,"new_saved_optimizer_updates":4000,
        "frames":116257,"episodes":2107,"dataset_root":str(SOURCE/"lerobot_dataset"),
        "stats_sha256":sha256_file(SOURCE/"lerobot_dataset/meta/stats.json"),
        "dataset_info_sha256":sha256_file(SOURCE/"lerobot_dataset/meta/info.json"),
        "privileged_model_input":False,"normalization_unchanged":True})


def run(adopt=False):
    lock = (ROOT/"VLA/logs"/(OUTPUT.name+".lock")).open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((OUTPUT/"plan.json").read_text())
    for state in plan["previous"].values(): verify_frozen(state)
    lanes, history = {},[]
    train_projection = (FIRST_RECOVERY if adopt else OUTPUT)/"train"
    env = {**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONUNBUFFERED":"1",
        "HF_HOME":str(ROOT/"VLA/checkpoints/huggingface"),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
        "TOKENIZERS_PARALLELISM":"false","VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR":"1",
        "VLA_SMOLVLA_INITIAL_FRAME_PROB":".1","VLA_SMOLVLA_TARGET_BALANCED":"1","VLA_SMOLVLA_GROUNDING_PROB":"0",
        "VLA_SMOLVLA_SAMPLING_SOURCE":str(SOURCE/"qualified_train"),"VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB":"6400"}
    def save(stage,status="running",error=None):
        atomic_json(OUTPUT/"status.json",{"pid":os.getpid(),"status":status,"stage":stage,"error":error,
            "lanes":{k:{"pid":v['process'].pid,"alive":v['process'].poll() is None,"exit_code":v['process'].poll()} for k,v in lanes.items()},
            "history":history,"heartbeat_utc":datetime.now(timezone.utc).isoformat(),
            "no_existing_job_signalled":True,"learned_success_established":False})
    def start(name,command,extra=None):
        stream=(OUTPUT/(name+".log")).open("x")
        process=subprocess.Popen(list(map(str,command)),cwd=ROOT,env={**env,**(extra or {})},stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT)
        lanes[name]={"process":process,"stream":stream}; save("parallel_saved_fit_recovery_and_CPU_replay")
    def check_report(path,updates=None):
        report=json.loads(Path(path).read_text())
        if report.get("passed") is not True: raise ValueError("Failed actual gate: "+str(path))
        if updates is not None and report.get("optimizer_steps") != [float(updates)]:
            raise ValueError("Saved optimizer count differs")
    def stage(name,command,extra=None,report=None):
        save(name)
        with (OUTPUT/(name+".log")).open("x") as stream:
            result=subprocess.run(list(map(str,command)),cwd=ROOT,env={**env,**(extra or {})},
                stdout=stream,stderr=subprocess.STDOUT,timeout=86400)
        history.append({"stage":name,"exit_code":result.returncode})
        if result.returncode: raise RuntimeError(name+" failed; retained log, no silent retry")
        if report: check_report(report)
    try:
        save("verify_saved_recipe_and_checkpoint_bindings")
        from VLA.common.source_training_gate import require_training_source
        require_training_source(SOURCE/"qualified_train",model="smolvla",optimizer_updates=5000)
        # Saved masters/Adam must be inspected before launching continuation.
        stage("rdt_saved_optimizer_audit",[RDT,"-m","VLA.rdt.continue_interrupted_training","--cycle",RDT_OLD,
            "--checkpoint",RDT_SAVED,"--output",RDT_TRAIN,"--audit-output",OUTPUT/"rdt_saved_optimizer_audit.json","--inspect-only"],
            report=OUTPUT/"rdt_saved_optimizer_audit.json")
        if not adopt:
            start("smol_resume",[SIM,"-m","VLA.smolvla.train_phase_balanced",f"--config_path={SMOL_SAVED/'train_config.json'}",
                "--resume=true","--save_freq=4000"],{"VLA_SMOLVLA_MASTER_AUDIT":str(SMOL_TRAIN/"master_dtype_audit.json")})
        if plan.get("adopted_live_rdt_pid"):
            class ReceiptObservedProcess:
                pid=plan["adopted_live_rdt_pid"]
                def poll(self):
                    if pid_alive(self.pid): return None
                    receipt=RDT_TRAIN/"continuation_summary.json"
                    if not receipt.exists(): return 1
                    data=json.loads(receipt.read_text())
                    return 0 if data.get("cumulative_optimizer_updates")==5000 and data.get("completed_new_updates")==4000 else 1
                @property
                def returncode(self): return self.poll()
            lanes["rdt_continue"]={"process":ReceiptObservedProcess(),"stream":None,
                "completion_receipt_NOT_observed_OS_exit_code":True}
            atomic_json(OUTPUT/"rdt_live_adoption.json",{"pid":plan["adopted_live_rdt_pid"],"trainer_not_restarted":True,
                "no_updates_lost":True,"requires_actual_saved5000_receipt_and_later_precision_gate":True})
        else:
            start("rdt_continue",[RDT,"-m","VLA.rdt.continue_interrupted_training","--cycle",RDT_OLD,
                "--checkpoint",RDT_SAVED,"--output",RDT_TRAIN,"--audit-output",OUTPUT/"rdt_live_continuation_audit.json"])
        if not adopt:
            start("project_train",[SIM,"-m","VLA.resume_simstate_projection","--source",SOURCE/"qualified_train",
                "--source-audit",SOURCE/"qualified_train_audit/audit.json","--previous",ROOT/"VLA/audits/simstate18_full_repair_r1_20261013/train",
                "--output",train_projection,"--split","train","--workers","2"])
        deadline=time.monotonic()+72*3600
        def dependencies_running():
            if any(v['process'].poll() is None for v in lanes.values()): return True
            if adopt:
                previous=json.loads((FIRST_RECOVERY/"status.json").read_text())
                if pid_alive(previous["pid"]): return True
                if previous["status"] != "failed" or "Independent lanes failed: rdt_continue;" not in (previous.get("error") or ""):
                    raise ValueError("Other inherited lane failed; do not adopt incomplete results")
            return False
        while dependencies_running():
            save("parallel_saved_fit_recovery_and_CPU_replay")
            if time.monotonic()>deadline: raise TimeoutError("Finite recovery wait expired; no completion claim")
            time.sleep(15)
        for name,v in lanes.items():
            if v['stream'] is not None: v['stream'].close()
            history.append({"stage":name,"exit_code":v['process'].returncode,
                "completion_receipt_NOT_observed_OS_exit_code":v.get("completion_receipt_NOT_observed_OS_exit_code",False)})
        # Preserve other successful lanes even if one failed. Do not kill them.
        failures=[name for name,v in lanes.items() if v['process'].returncode]
        if failures: raise RuntimeError("Independent lanes failed: "+", ".join(failures)+"; successful lanes preserved")
        if adopt:
            check_report(train_projection/"audit.json")
            check_report(ROOT/"VLA/audits/recovery_fast_projection_pilot_20261014/audit.json")
        finalize_smol()
        for name,command,report in [
            ("smol_precision",[SIM,"-m","VLA.smolvla.audit_training_precision","--checkpoint",SMOL_FINAL,"--output",OUTPUT/"smol_precision.json"],OUTPUT/"smol_precision.json"),
            ("rdt_precision",[RDT,"-m","VLA.rdt.audit_training_precision","--checkpoint",RDT_FINAL,"--updates","5000","--output",OUTPUT/"rdt_precision.json"],OUTPUT/"rdt_precision.json"),
            ("rdt_reload",[RDT,"-m","VLA.rdt.audit_deployment_precision","--checkpoint",RDT_FINAL,"--output",OUTPUT/"rdt_reload.json"],OUTPUT/"rdt_reload.json")]:
            stage(name,command,report=report)
        check_report(OUTPUT/"smol_precision.json",5000)
        for name,cp in (("smol",SMOL_FINAL),("rdt",RDT_FINAL)):
            stage(name+"_fingerprint",[SIM,"-m","VLA.fingerprint_checkpoint",cp])
        stage("smol_actual_input_parity",[SIM,"-m","VLA.smolvla.audit_train_infer_inputs","--checkpoint",SMOL_FINAL,
            "--source",SOURCE/"qualified_train","--audit-report",SOURCE/"qualified_train_audit/audit.json",
            "--episodes-per-task","2","--output",OUTPUT/"smol_actual_input_parity.json"],report=OUTPUT/"smol_actual_input_parity.json")
        stage("project_validation",[SIM,"-m","VLA.resume_simstate_projection","--source",SOURCE/"validation_candidate",
            "--source-audit",SOURCE/"validation_audit/audit.json","--output",OUTPUT/"validation","--split","validation","--workers","2"])
        from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
        validate_holdout(SOURCE/"validation_candidate",SOURCE/"qualified_train")
        stage("full_actual_loader",[SIM,"-m","VLA.smolvla.audit_simstate_loader","--projection",train_projection,
            "--dataset",SOURCE/"lerobot_dataset","--base-model",SMOL_FINAL,"--output",OUTPUT/"full_actual_loader.json"],report=OUTPUT/"full_actual_loader.json")
        from VLA.run_simstate_repair_cycle import training_command,finalize_checkpoint
        # Process-local CLI producer retains all qualified source/model gates;
        # periodic saves reduce lost work without changing learned objectives.
        for label,updates,accumulation in (("simstate_smoke",2,2),("simstate_train",5000,8)):
            fit=ROOT/"VLA/smolvla_simstate"/OUTPUT.name/label
            command=training_command(SMOL_FINAL,train_projection,fit,updates,accumulation)
            command=[f"--save_freq={min(updates,250)*accumulation}" if str(v).startswith("--save_freq=") else v for v in command]
            stage(label,command,{"VLA_SIMSTATE_TRAIN_PROJECTION":str(train_projection),"VLA_SIMSTATE_OPTIMIZER_UPDATES":str(updates),
                "VLA_SMOLVLA_MASTER_AUDIT":str(fit/"master_dtype_audit.json"),"VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB":"10000"})
            cp=finalize_checkpoint(fit,updates,accumulation,train_projection)
            stage(label+"_precision",[SIM,"-m","VLA.smolvla.audit_training_precision","--checkpoint",cp,"--output",OUTPUT/(label+"_precision.json")],
                report=OUTPUT/(label+"_precision.json"))
            check_report(OUTPUT/(label+"_precision.json"),updates)
            stage(label+"_fingerprint",[SIM,"-m","VLA.fingerprint_checkpoint",cp])
            if updates==2:
                stage("new_simstate_columns",[SIM,"-m","VLA.smolvla.audit_simstate_updates","--base",SMOL_FINAL,
                    "--checkpoint",cp,"--output",OUTPUT/"new_simstate_columns.json"],report=OUTPUT/"new_simstate_columns.json")
        stage("simstate120",[SIM,"-m","VLA.evaluate_simstate120","--checkpoint",cp,"--output",OUTPUT/"development120"])
        stage("simstate120_failure_analysis",[SIM,"-m","VLA.analyze_simstate120","--progress",OUTPUT/"development120/progress.json",
            "--output",OUTPUT/"post120_failure_audit"])
        stage("ordinary120_per_model",[SIM,"-m","VLA.run_recovered_cross120","--rdt-checkpoint",RDT_FINAL,
            "--output",ROOT/("VLA/audits/cross_vla120_"+OUTPUT.name)])
        save("completed_measured_cohorts_REQUIRE_success_and_video_review",status="completed")
    except Exception as error:
        save("preserved_error_NO_false_success",status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--launch",action="store_true");mode.add_argument("--run",action="store_true")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--adopt-existing",action="store_true")
    parser.add_argument("--handoff-coordinator",type=Path)
    args=parser.parse_args()
    if args.output: OUTPUT=args.output.resolve()
    launch(args.adopt_existing,args.handoff_coordinator) if args.launch else run(args.adopt_existing)
