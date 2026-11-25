"""Same locked120 scenes/scorer: demonstration teacher feasibility diagnostic.

NOT neural success and NEVER training data. Planner failure is not proof of
physical impossibility. Every declared scene/outcome is preserved; no reroll.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from VLA.benchmark import _interaction_audit,_save_observation_snapshot
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder,NullVideoRecorder
from VLA.common.tasks import randomize_rl_scene,select_tasks
from VLA.cross_vla120_trial import atomic_json
from VLA.run_cross_vla120 import ROOT,SEEDS,TASKS,PROFILE_PROTOCOL


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--launch",action="store_true")
    args=parser.parse_args()
    output=args.output.resolve()
    if args.launch:
        marker=output.parent/(output.name+"_launch.json")
        if output.exists() or marker.exists(): raise FileExistsError("Preserve prior120 diagnostic")
        with (output.parent/(output.name+"_driver.log")).open("x") as log:
            process=subprocess.Popen([sys.executable,"-m","VLA.audit_locked120_teacher_feasibility","--output",str(output)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        atomic_json(marker,{"pid":process.pid,"detached_session":True});print(process.pid);return
    if output.exists(): raise FileExistsError("No repeat scene selection")
    output.mkdir(parents=True)
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    profile=json.loads(Path(protocol["observation_profile_file"]).read_text())
    if profile_hash(profile)!=protocol["observation_profile_sha256"]:
        raise ValueError("Do not change the camera/controller")
    common=ROOT/"VLA/audits/cross_vla120_per_model_20261013/common_scene_resets"
    rows=[]
    def save(status,error=None):
        atomic_json(output/"progress.json",{"status":status,"pid":os.getpid(),"planned_trials":120,
            "completed_teacher_trials":len(rows),"teacher_successes":sum(r["success"] for r in rows),"rows":rows,
            "neural_trials":0,"use_as_training_data":False,"model_success_established":False,
            "no_outcome_seed_filter_or_reroll":True,"error":error,"updated_at_utc":datetime.now(timezone.utc).isoformat()})
    save("running_teacher_reference_NOT_neural")
    try:
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            tasks=[randomize_rl_scene(t,rng) for t in select_tasks("all")]
            env=environment_class_for_profile(profile)(observation_profile=profile,robot="ur5e",image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=protocol["success"]["semantic_thresholds"])
            folder=ROOT/"VLA/oracle"/output.name/f"test_seed{seed}"
            folder.mkdir(parents=True,exist_ok=False)
            teacher=CollisionPlannedTeacher(env)
            recorder=VideoRecorder(folder/"rollout.mp4",fps=10,overview_render=lambda:env.render("overview")) if seed==SEEDS[0] else NullVideoRecorder()
            traces={}
            try:
                with recorder:
                    for task in tasks:
                        observation=env.reset(task)
                        reset=json.loads((common/f"seed{seed}"/task.name/"reset.json").read_text())
                        if json.loads(json.dumps(task.to_dict()))!=reset["task"] or env.is_semantic_success():
                            raise ValueError("Locked scene identity or initial success differs")
                        from PIL import Image
                        for camera in ("front","wrist"):
                            expected=np.asarray(Image.open(common/f"seed{seed}"/task.name/f"{camera}.png"))
                            if not np.array_equal(expected,getattr(observation,camera+"_rgb")):
                                raise ValueError("Reference teacher reset pixels differ from locked model scene")
                        _save_observation_snapshot(folder,task.name,observation,env)
                        trace=[];info={"success":False};planner_error=None
                        try:
                            teacher.reset(task);done=False
                            while not done:
                                before=_interaction_audit(env)
                                action=teacher.predict(observation)
                                observation,_,done,info=env.step(action)
                                trace.append({"step":observation.step,"action":action.tolist(),"phase":teacher.phase,
                                    "before":before,"after":_interaction_audit(env),"success":bool(info["success"]),
                                    "safety_violation":info.get("safety_violation")})
                                recorder.add(observation,"TEACHER feasibility ONLY | "+task.prompt)
                        except Exception as error:
                            planner_error=f"{type(error).__name__}: {error}"
                        _save_observation_snapshot(folder,task.name,observation,env)
                        traces[task.name]=trace;atomic_json(folder/"trajectory.json",traces)
                        rows.append({"seed":seed,"task":task.name,"prompt":task.prompt,"success":bool(info["success"]) and planner_error is None,
                            "planner_error_NOT_impossibility_proof":planner_error,"safety_violation":info.get("safety_violation"),
                            "steps":len(trace),"phase":teacher.phase,"trajectory":str(folder/"trajectory.json")})
                        save("running_teacher_reference_NOT_neural")
                        print(f"Teacher reference {len(rows)}/120: {task.name} success={rows[-1]['success']} error={planner_error}",flush=True)
            finally:
                teacher.close();env.close()
        if Counter(r["task"] for r in rows)!={t:30 for t in TASKS}:
            raise ValueError("Incomplete locked120 reference")
        atomic_json(output/"audit.json",{"passed":True,"scope":"Locked120 teacher-reference feasibility ONLY, NOT neural success or training qualification",
            "rows":rows,"completed_trials":120,"teacher_successes":sum(r["success"] for r in rows),
            "neural_trials":0,"use_as_training_data":False,"failed_teacher_NOT_impossibility_proof":True,
            "scorer_protocol_sha256":sha256_file(PROFILE_PROTOCOL),"seeds_unchanged":SEEDS,
            "no_scoring_camera_robot_or_asset_change":True})
        save("completed120_teacher_reference_requires_failure_review")
    except Exception as error:
        save("stopped_reference_integration_error_NO_reroll",str(error));raise


if __name__=="__main__":
    main()
