"""Deadline-bounded full input gates -> real smoke -> fit -> learned120.

Does not interrupt existing training/evaluation. Own GPU work waits for both
the current RDT and serialized native queue. New67 long fit is never launched
on incomplete extraction, stale input contracts or failed real CUDA/reload.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT,gpu_state
from VLA.scene64_conditioning import NUMERIC_SCHEMA,NUMERIC_NAMES

SIM=ROOT/"VLA/third_party/lerobot/.venv/bin/python"
OV=ROOT/"VLA/.venvs/openvla/bin/python"
PI=ROOT/"VLA/third_party/openpi/.venv/bin/python"
SOURCE=ROOT/"VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013"
PROJECTION=ROOT/"VLA/audits/scene64_recovery_r30_20261014/full_train64"
BASE=ROOT/"VLA/smolvla_simstate/privileged_batchfix_r22_20261014/simstate_train/checkpoints/040000/pretrained_model"
OTHER=ROOT/"VLA/audits/other_vla_privileged_r29_20261014/gpu_queue_rdt_serialized_20261015"


def finalise(fit,updates,accumulation,admission,projection):
    from safetensors import safe_open
    cp=fit/"checkpoints"/f"{updates*accumulation:06d}"/"pretrained_model"
    cfg=json.loads((cp/"config.json").read_text())
    if cfg["max_state_dim"]!=67 or cfg["input_features"]["observation.state"]["shape"]!=[67]:
        raise ValueError("Saved new input was truncated to32")
    masters=json.loads((fit/"master_dtype_audit.json").read_text())
    with safe_open(cp/"model.safetensors",framework="np") as weights:
        if any(weights.get_slice(name).get_dtype()!="F32" for name in masters["trainable_parameter_names"]):
            raise ValueError("Learned native values were downcast")
    with safe_open(cp.parent/"training_state/optimizer_state.safetensors",framework="np") as moments:
        if any(moments.get_slice(name).get_dtype()!="F32" for name in moments.keys() if "exp_avg" in name):
            raise ValueError("Actual saved optimizer moments were downcast")
    atomic_json(cp/"fp32_master_contract.json",{**masters,"saved_weights_and_adam_moments_fp32":True})
    paths=[ROOT/"VLA/scene64_conditioning.py",ROOT/"VLA/scene64_schema.py",ROOT/"VLA/simstate_scene64.py",ROOT/"VLA/simstate_feedback.py"]
    atomic_json(cp/"scene67_contract.json",{"schema":NUMERIC_SCHEMA,"feature_names":NUMERIC_NAMES,
        "oracle_controller_at_deployment":False,"input_bound_files":{str(p):sha256_file(p) for p in paths},
        "projection_audit_sha256":sha256_file(projection/"audit.json"),"recovery_admission_sha256":sha256_file(admission),
        "new_optimizer_updates":updates,"prior_family_updates":20000,"normalization_first18_and_actions_inherited_from_TRAIN":True})
    source=SOURCE/"qualified_train"
    atomic_json(cp/"vla_checkpoint_manifest.json",{"source":str(source),"source_manifest_sha256":sha256_file(source/"dataset_manifest.json"),
        "input_regime":NUMERIC_SCHEMA,"original_source_episodes":2107,"original_source_frames":116257,
        "extra_exact_onpolicy_correction_episodes":38,"extra_correction_frames":1479,
        "correction_admission":str(admission),"original_profile_robot_objects_physics_unchanged":True})
    return cp


def command(warm,fit,updates,accumulation):
    rename=json.dumps({"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"})
    return [SIM,"-m","VLA.smolvla.train_scene67","--policy.type=smolvla",f"--policy.pretrained_path={warm}",
        "--policy.load_vlm_weights=true","--policy.max_state_dim=67","--policy.chunk_size=5","--policy.n_action_steps=1",
        "--policy.freeze_vision_encoder=true","--policy.train_expert_only=false","--policy.train_state_proj=true",
        "--policy.optimizer_lr=0.000025","--policy.scheduler_decay_lr=0.000000625",
        f"--policy.scheduler_warmup_steps={0 if updates==2 else 150*accumulation}",
        f"--policy.scheduler_decay_steps={max(updates,1000)*accumulation}","--policy.push_to_hub=false",
        "--dataset.repo_id=local/scene67_original_with_exact_recovery",f"--dataset.root={SOURCE/'lerobot_dataset'}",
        "--dataset.image_transforms.enable=false",f"--rename_map={rename}",f"--output_dir={fit}",
        f"--job_name=smolvla_scene67_{updates}",f"--steps={updates*accumulation}","--batch_size=2","--num_workers=0",
        "--accelerator.mixed_precision=bf16",f"--accelerator.gradient_accumulation.steps={accumulation}",
        "--wandb.enable=false","--env_eval_freq=0","--eval_steps=0",f"--save_freq={min(updates,250)*accumulation}",
        "--save_checkpoint=true","--seed=20261015"]


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/"VLA/scene64_conditioning.py",ROOT/"VLA/scene64_schema.py",ROOT/"VLA/scene64_dataset.py",
        ROOT/"VLA/smolvla/scene67_dataset.py",ROOT/"VLA/smolvla/train_scene67.py",ROOT/"VLA/smolvla/scene67_adapter.py",
        ROOT/"VLA/smolvla/prepare_scene67_checkpoint.py",ROOT/"VLA/smolvla/audit_scene67_batches.py",
        ROOT/"VLA/audit_scene64_native_inputs.py",ROOT/"VLA/evaluate_scene67_smolvla.py",args.recovery_admission.resolve()]
    frozen={str(p):sha256_file(p) for p in files};history=[];stage_name="waiting_complete_physical_extraction";active=None
    deadline=time.monotonic()+args.wait_hours*3600
    environment={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","PYTHONUNBUFFERED":"1",
        "HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","VLA_SCENE67_PROJECTION":str(PROJECTION),
        "VLA_SCENE67_RECOVERY_ADMISSION":str(args.recovery_admission.resolve()),"VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR":"1",
        "VLA_SMOLVLA_INITIAL_FRAME_PROB":".1","VLA_SMOLVLA_TARGET_BALANCED":"1","VLA_SMOLVLA_GROUNDING_PROB":"0",
        "VLA_SMOLVLA_SAMPLING_SOURCE":str(SOURCE/"qualified_train"),"VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB":"10000"}
    def save(status,error=None):
        atomic_json(output/"status.json",{"pid":os.getpid(),"status":status,"stage":stage_name,"history":history,
            "error":error,"frozen_files":frozen,"updated_at_utc":datetime.now(timezone.utc).isoformat(),
            "deadline_remaining_seconds":max(0,deadline-time.monotonic()),"no_existing_job_interrupted":True,
            "new_model_success_rate_established":False,"all_other_models_scene64_finetuned":False})
    def check():
        if any(sha256_file(path)!=digest for path,digest in frozen.items()):raise ValueError("Frozen new input/source changed")
    def stage(name,argv,extra=None,proof=None):
        nonlocal stage_name,active
        stage_name=name;check();save("running")
        with (output/(name+".log")).open("x") as log:
            active=subprocess.Popen(list(map(str,argv)),cwd=ROOT,env={**environment,**(extra or {})},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/"active_child.json",{"pid":active.pid,"stage":name,"command":list(map(str,argv))})
            while active.poll() is None:
                save("running");time.sleep(15)
                if time.monotonic()>deadline:raise TimeoutError("Owned stage retained; no in-flight episode killed")
        history.append({"stage":name,"exit_code":active.returncode})
        if active.returncode:raise RuntimeError(name+" failed; preserve actual artifacts")
        if proof is not None and not json.loads(proof.read_text()).get("passed"):raise ValueError(name+" gate failed")
    try:
        while not (PROJECTION/"audit.json").exists():
            check();save("running")
            progress=json.loads((PROJECTION/"progress.json").read_text())
            if str(progress["status"]).startswith("failed"):raise ValueError("Full physical extraction failed")
            if time.monotonic()>deadline:raise TimeoutError("Finite full-data deadline")
            time.sleep(30)
        for mode,python in (("numeric",SIM),("language",OV),("openpi",PI)):
            stage("full_native_"+mode,[python,"-m","VLA.audit_scene64_native_inputs","--projection",PROJECTION,
                "--mode",mode,"--output",output/("full_native_"+mode)],extra={"JAX_PLATFORMS":"cpu"} if mode=="openpi" else None,
                proof=output/("full_native_"+mode)/"audit.json")
        batch=output/"actual_scene67_batches.json"
        stage("actual_scene67_batches",[SIM,"-m","VLA.smolvla.audit_scene67_batches","--base-model",BASE,
            "--projection",PROJECTION,"--recovery-admission",args.recovery_admission,"--output",batch],proof=batch)
        stage_name="waiting_registered_RDT_and_other_queue_then_quiet_GPU"
        while True:
            check();pending=False
            for root in (ROOT/"VLA/audits/rdt_privileged_postfit_r25_20261014",OTHER):
                state=json.loads((root/"status.json").read_text());proc=Path(f"/proc/{state['pid']}")
                alive=proc.exists() and (proc/"stat").read_text().rsplit(")",1)[1].split()[0]!="Z"
                pending |= alive or state.get("status") not in {"completed","failed_preserve_all_actual_artifacts","failed"}
            gpu=gpu_state()
            if not pending and not gpu["compute_pids"] and gpu["free_mib"]>=11000:break
            save("running")
            if time.monotonic()>deadline:raise TimeoutError("Finite GPU wait; no owner signalled")
            time.sleep(30)
        warm=output/"architecture_warmstart67"
        stage("prepare_architecture_warmstart",[SIM,"-m","VLA.smolvla.prepare_scene67_checkpoint","--base",BASE,"--output",warm])
        qualification=output/"actual_scene67_smoke_qualification.json"
        for updates,label in ((2,"smoke"),(args.updates,"train")):
            fit=ROOT/"VLA/smolvla_scene67"/output.name/label
            extra={"VLA_SCENE67_UPDATES":str(updates),"VLA_SMOLVLA_MASTER_AUDIT":str(fit/"master_dtype_audit.json")}
            if updates>2:extra["VLA_SCENE67_SMOKE_QUALIFICATION"]=str(qualification)
            stage(label,command(warm,fit,updates,4),extra)
            cp=finalise(fit,updates,4,args.recovery_admission,PROJECTION)
            precision=output/(label+"_precision.json")
            stage(label+"_precision",[SIM,"-m","VLA.smolvla.audit_training_precision","--checkpoint",cp,"--output",precision],proof=precision)
            if json.loads(precision.read_text())["optimizer_steps"]!=[float(updates)]:raise ValueError("Actual saved Adam count differs")
            if updates==2:
                from safetensors import safe_open
                import numpy as np
                with safe_open(cp/"model.safetensors",framework="np") as saved:
                    new=saved.get_tensor("model.state_proj.weight")[:,18:]
                    if not np.isfinite(new).all() or not np.count_nonzero(new):
                        raise ValueError("New49 conditioning inputs received no actual updates")
                stage("actual_native_reload8",[SIM,"-m","VLA.evaluate_scene67_smolvla","--checkpoint",cp,
                    "--output",output/"actual_native_reload8","--smoke-source",SOURCE/"qualified_train"])
                actual=json.loads((output/"actual_native_reload8/progress.json").read_text())
                if actual["status"]!="completed8_integration_NOT_SR" or actual["neural_controls"]!=8:raise ValueError("Actual new67 neural robot gate failed")
                dependencies={**frozen,str(PROJECTION/"audit.json"):sha256_file(PROJECTION/"audit.json"),str(batch):sha256_file(batch),
                    str(precision):sha256_file(precision),str(output/"actual_native_reload8/progress.json"):sha256_file(output/"actual_native_reload8/progress.json")}
                atomic_json(qualification,{"passed":True,"actual_optimizer_updates":2,"actual_neural_controls":8,
                    "exact_trainable_values_reload":True,"task_SR_certified":False,"dependencies":dependencies})
        stage("locked_development120",[SIM,"-m","VLA.evaluate_scene67_smolvla","--checkpoint",cp,"--output",output/"development120"])
        stage_name="completed_scene67_fit_and_development120_requires_final_review";save("completed")
    except Exception as error:save("failed_preserve_all_artifacts",f"{type(error).__name__}: {error}");raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--recovery-admission",type=Path,required=True);p.add_argument("--updates",type=int,default=2500)
    p.add_argument("--wait-hours",type=float,default=60);p.add_argument("--launch",action="store_true");a=p.parse_args()
    if not 0<a.wait_hours<=72 or not 2<a.updates<=5000:raise ValueError("Finite deadline/bounded fit required")
    if not a.launch:return run(a)
    marker=a.output.with_name(a.output.name+"_launch.json")
    if marker.exists() or a.output.exists():raise FileExistsError("No duplicate registered fit")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.with_name(a.output.name+"_launcher.log").open("x") as log:
        child=subprocess.Popen([str(SIM),"-m","VLA.run_scene67_finetune_cycle","--output",str(a.output.resolve()),
            "--recovery-admission",str(a.recovery_admission.resolve()),"--updates",str(a.updates),"--wait-hours",str(a.wait_hours)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic_json(marker,{"pid":child.pid,"deadline_budget_hours":a.wait_hours,"existing_jobs_signalled":False})
    print(child.pid)


if __name__ == "__main__":main()
