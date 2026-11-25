"""One declared 4-task DEVELOPMENT check at the first250-update state18 save.

Runs beside the owned trainer with a bounded inference memory budget. This
NEVER selects/replaces the already declared5000-update final checkpoint,
adds to its120-trial denominator, changes the controller or rerolls outcomes.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT,SEEDS,TASKS,PROFILE_PROTOCOL,gpu_state
from VLA.recover_interrupted_cycles import pid_alive


def evaluate(cycle,output,checkpoint,updates,angle_alias=False):
    from safetensors import safe_open
    from VLA.common.environment_factory import environment_class_for_profile
    from VLA.common.profile_binding import checkpoint_profile
    from VLA.common.observation_profile import profile_hash
    from VLA.common.recorder import VideoRecorder
    from VLA.common.tasks import randomize_rl_scene,select_tasks
    from VLA.benchmark import _interaction_audit,_save_observation_snapshot
    from VLA.simstate_feedback import feedback_state
    from VLA.smolvla.simstate_adapter import SimStateSmolVLA
    from VLA.fingerprint_checkpoint import verify_fingerprint
    # Saved optimizer receipts, not printed training percentages.
    with safe_open(checkpoint.parent/"training_state/optimizer_state.safetensors",framework="np") as tensors:
        steps={float(tensors.get_tensor(k)) for k in tensors.keys() if k.endswith("/step")}
    if steps!={float(updates)}:raise ValueError("Actual saved optimizer counts differ from the declared diagnostic")
    verify_fingerprint(checkpoint,full=True)
    protocol=json.loads(PROFILE_PROTOCOL.read_text());profile=checkpoint_profile(str(checkpoint))
    if profile_hash(profile)!=protocol["observation_profile_sha256"]:raise ValueError("No camera/physics change")
    seed=SEEDS[0];rng=np.random.default_rng(seed)
    tasks=[randomize_rl_scene(t,rng) for t in select_tasks("all")]
    env=environment_class_for_profile(profile)(observation_profile=profile,robot="ur5e",image_size=384,
        control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
        success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
        semantic_success_thresholds=protocol["success"]["semantic_thresholds"])
    suffix="_angle_alias" if angle_alias else ""
    folder=ROOT/"VLA/smolvla_simstate"/cycle.name/f"diagnostic{updates}{suffix}_seed20268001"
    folder.mkdir(parents=True,exist_ok=False)
    rows=[];traces={}
    if angle_alias:
        from VLA.smolvla.joint_angle_alias import AngleAliasSimStatePolicy
        policy=AngleAliasSimStatePolicy(env,str(checkpoint),projection=ROOT/"VLA/audits/recovery_r17_20261013/train",replan_steps=1,seed=seed)
        atomic_json(output/"angle_alias_range_proof.json",policy.angle_proof)
    else:policy=SimStateSmolVLA(env,str(checkpoint),replan_steps=1,seed=seed)
    def save(status,error=None):
        atomic_json(output/"progress.json",{"status":status,"pid":os.getpid(),"seed":seed,
            "planned_trials":4,"completed_trials":len(rows),"successes":sum(r["success"] for r in rows),
            "rows":rows,"error":error,"source_optimizer_updates":updates,"privileged_model_inputs":True,
            "oracle_controller_at_deployment":False,"NOT_final5000_SR120":True,
            "final_checkpoint_selection_UNCHANGED_5000":True,"final_test_used":False})
    try:
        save("running_provisional4")
        with VideoRecorder(folder/"rollout.mp4",fps=10,overview_render=lambda:env.render("overview")) as recorder:
            for task in tasks:
                obs=env.reset(task);policy.reset(task)
                if env.is_semantic_success():raise ValueError("No initial-success scene replacement")
                _save_observation_snapshot(folder,task.name,obs,env)
                done=False;trace=[]
                while not done:
                    before=_interaction_audit(env);state=feedback_state(env,obs);raw_q=state[:6].copy()
                    if angle_alias:
                        from VLA.smolvla.joint_angle_alias import aliases
                        state[:6],_=aliases(state[:6],policy.angle_means)
                    action=policy.predict(obs)
                    if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError("Invalid learned action")
                    obs,_,done,info=env.step(action)
                    trace.append({"step":obs.step,"action":action.tolist(),"pre_action_PRIVILEGED_state18":state.tolist(),
                        "raw_encoder_angles_before_alias":raw_q.tolist(),"equivalent_angle_preprocessing":angle_alias,
                        "interaction_before_audit":before,"interaction_after_audit":_interaction_audit(env),
                        "success":bool(info["success"]),"safety_violation":info.get("safety_violation")})
                    recorder.add(obs,f"{updates}-update diagnostic ONLY | "+task.prompt)
                    if obs.step%10==0:_save_observation_snapshot(folder,task.name,obs,env)
                _save_observation_snapshot(folder,task.name,obs,env)
                traces[task.name]=trace;atomic_json(folder/"trajectory.json",traces)
                rows.append({"seed":seed,"task":task.name,"success":bool(info["success"]),"steps":len(trace),
                    "safety_violation":info.get("safety_violation"),"trajectory":str(folder/"trajectory.json")})
                save("running_provisional4")
        if Counter(r["task"] for r in rows)!={t:1 for t in TASKS}:raise ValueError("Incomplete fixed4")
        save("completed4_NOT_final120")
    except Exception as error:
        save("preserved_error_no_reroll",str(error));raise
    finally:policy.close();env.close()


def run(cycle,output,updates,angle_alias=False):
    output.mkdir(parents=True,exist_ok=False)
    fit=ROOT/"VLA/smolvla_simstate"/cycle.name/"simstate_train"
    checkpoint=fit/"checkpoints"/f"{updates*8:06d}"/"pretrained_model"
    deadline=time.monotonic()+6*3600
    while True:
        state=json.loads((cycle/"status.json").read_text())
        if state["status"]!="running" or not pid_alive(state["pid"]):
            raise ValueError("Owned main trainer/cycle not live; do not infer a saved checkpoint")
        step=checkpoint.parent/"training_state/training_step.json"
        # The official last pointer is updated ONLY after the entire model,
        # processor and optimizer save has finished; no half-written reads.
        last=(fit/"checkpoints/last").resolve()
        committed=last.name.isdigit() and int(last.name)>=updates*8
        if step.exists() and (checkpoint/"model.safetensors").exists() and committed:break
        atomic_json(output/"progress.json",{"status":f"waiting_ACTUAL_saved{updates}","pid":os.getpid(),
            "completed_trials":0,"planned_trials":4,"NOT_final5000_SR120":True})
        if time.monotonic()>deadline:raise TimeoutError("No declared saved checkpoint; no phantom evaluation")
        time.sleep(15)
    child=json.loads((cycle/"active_child.json").read_text())
    gpu=gpu_state()
    if child["stage"]!="simstate_train" or set(gpu["compute_pids"])-{str(child["pid"])} or gpu["free_mib"]<3500:
        raise ValueError("Unknown GPU owner/insufficient headroom; do not overlap")
    from VLA.run_simstate_repair_cycle import finalize_checkpoint
    if not (checkpoint/"benchmark_fingerprint.json").exists():
        finalize_checkpoint(checkpoint.parents[2],updates,8,ROOT/"VLA/audits/recovery_r17_20261013/train")
    result=subprocess.run([sys.executable,"-m","VLA.fingerprint_checkpoint",str(checkpoint),"--verify-existing"],cwd=ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
    if result.returncode:raise RuntimeError("Saved diagnostic checkpoint fingerprint failed")
    import torch
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(3000*1024**2/total,device=0)
    evaluate(cycle,output,checkpoint,updates,angle_alias)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--updates",type=int,choices=(250,1000),default=250)
    parser.add_argument("--angle-alias",action="store_true",help="Explicit same-checkpoint equivalent-angle diagnostic")
    args=parser.parse_args();cycle=args.cycle.resolve();output=args.output.resolve()
    if args.launch:
        marker=output.with_name(output.name+"_launch.json")
        if output.exists() or marker.exists():raise FileExistsError("No repeated diagnostic launch")
        with output.with_name(output.name+"_driver.log").open("x") as log:
            command=[sys.executable,"-m","VLA.diagnose_simstate_partial","--cycle",str(cycle),"--output",str(output),"--updates",str(args.updates)]
            if args.angle_alias:command.append("--angle-alias")
            p=subprocess.Popen(command,
                cwd=ROOT,env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1",
                    "TRANSFORMERS_OFFLINE":"1","PYTHONUNBUFFERED":"1"},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":p.pid,"declared_updates":args.updates,"planned_diagnostic_trials":4,
            "equivalent_angle_preprocessing":args.angle_alias,
            "diagnostic_NOT_final_cohort":True});print(p.pid)
    else:run(cycle,output,args.updates,args.angle_alias)
