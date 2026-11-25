"""Same weights, first10 locked DEVELOPMENT seeds: input-only ablation40.

Separate cohort, no scorer/controller changes or selected final-test use.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from VLA.benchmark import _interaction_audit,_save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.evaluation_statistics import success_statistics
from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.recorder import VideoRecorder,NullVideoRecorder
from VLA.common.tasks import randomize_rl_scene,select_tasks
from VLA.cross_vla120_trial import atomic_json
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.run_cross_vla120 import ROOT,SEEDS,TASKS,PROFILE_PROTOCOL
from VLA.simstate_feedback import feedback_state
from VLA.smolvla.joint_angle_alias import aliases
from VLA.smolvla.robust_encoder_input import RobustEncoderSimStatePolicy,nearstatic_clip


def run(checkpoint,projection,output):
    import torch
    if output.exists():raise FileExistsError("No identical reroll")
    output.mkdir(parents=True)
    torch.cuda.set_per_process_memory_fraction(3000*1024**2/torch.cuda.get_device_properties(0).total_memory)
    identity = verify_fingerprint(checkpoint,full=True)
    profile = checkpoint_profile(checkpoint); template = json.loads(PROFILE_PROTOCOL.read_text())
    if profile_hash(profile) != template["observation_profile_sha256"]:raise ValueError("No camera/robot/physics change")
    rows = []
    def save(status,error=None):
        atomic_json(output/"progress.json",{"model":"smolvla_simstate_robust_encoder","status":status,
            "completed_trials":len(rows),"planned_trials":40,"trials_per_action":10,"successes":sum(r["success"] for r in rows),
            "rows":rows,"error":error,"checkpoint_content_sha256":identity["content_tree_sha256"],
            "weights_retrained_or_controller_changed":False,"privileged_model_inputs":True,
            "oracle_controller_at_deployment":False,"final_test_used":False,"NOT_original_raw_state18_SR120":True,
            "statistics":success_statistics(rows) if rows and len(rows)%4 == 0 else None})
    save("running_separate_input_ablation")
    try:
        for seed in SEEDS[:10]:
            rng=np.random.default_rng(seed); tasks=[randomize_rl_scene(t,rng) for t in select_tasks("all")]
            env=environment_class_for_profile(profile)(observation_profile=profile,robot="ur5e",image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=template["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=template["success"]["semantic_thresholds"])
            policy=RobustEncoderSimStatePolicy(env,str(checkpoint),projection,replan_steps=1,seed=seed)
            folder=ROOT/"VLA/smolvla_simstate"/output.name/f"test_seed{seed}"
            folder.mkdir(parents=True,exist_ok=False);traces={}
            try:
                atomic_json(folder/"input_recipe.json",{**policy.angle_proof,
                    "nearstatic_std_ceiling_rad":.001,"small_excursion_bound_rad":.01,
                    "nearstatic_clipping_is_NOT_exact_pose_equivalence":True,"pose_RGB_controller_unchanged":True})
                recorder=VideoRecorder(folder/"rollout.mp4",fps=10,overview_render=lambda:env.render("overview")) if seed==SEEDS[0] else NullVideoRecorder()
                with recorder:
                    for task in tasks:
                        obs=env.reset(task);policy.reset(task);done=False;trace=[]
                        if env.is_semantic_success():raise ValueError("No initial-success substitution")
                        _save_observation_snapshot(folder,task.name,obs,env)
                        while not done:
                            raw=feedback_state(env,obs);processed=raw.copy()
                            processed[:6],turns=aliases(processed[:6],policy.angle_means)
                            processed[:6],clipped=nearstatic_clip(processed[:6],policy.encoder_stats)
                            before=_interaction_audit(env);action=policy.predict(obs)
                            obs,_,done,info=env.step(action)
                            trace.append({"step":obs.step,"action":action.tolist(),"raw_pre_action_state18":raw.tolist(),
                                "actual_MODEL_state18_before_normalization":processed.tolist(),"whole2pi_turns":turns.tolist(),
                                "nearstatic_channels_clipped":clipped.tolist(),"before":before,"after":_interaction_audit(env),
                                "success":bool(info["success"]),"safety_violation":info.get("safety_violation")})
                            recorder.add(obs,"Input-only robust encoder ablation | "+task.prompt)
                            if obs.step%10==0:_save_observation_snapshot(folder,task.name,obs,env)
                        _save_observation_snapshot(folder,task.name,obs,env)
                        traces[task.name]=trace;atomic_json(folder/"trajectory.json",traces)
                        rows.append({"seed":seed,"task":task.name,"success":bool(info["success"]),"steps":len(trace),
                            "safety_violation":info.get("safety_violation"),"trajectory":str(folder/"trajectory.json")})
                        save("running_separate_input_ablation")
            finally:policy.close();env.close()
        if Counter(r["task"] for r in rows)!={t:10 for t in TASKS}:raise ValueError("Incomplete40")
        save("completed40_separate_input_ablation")
    except Exception as error:save("preserved_error_no_reroll",str(error));raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint","projection","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--launch",action="store_true");args=parser.parse_args();output=args.output.resolve()
    if args.launch:
        marker=output.with_name(output.name+"_launch.json")
        if output.exists() or marker.exists():raise FileExistsError("Registered input experiment already exists")
        with output.with_name(output.name+"_driver.log").open("x") as log:
            child=subprocess.Popen([sys.executable,"-m","VLA.evaluate_robust_encoder40","--checkpoint",str(args.checkpoint.resolve()),
                "--projection",str(args.projection.resolve()),"--output",str(output)],cwd=ROOT,
                env={**os.environ,"OMP_NUM_THREADS":"2","MKL_NUM_THREADS":"2","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":child.pid,"planned_trials":40,"separate_input_only_variant":True});print(child.pid)
    else:run(args.checkpoint.resolve(),args.projection.resolve(),output)
