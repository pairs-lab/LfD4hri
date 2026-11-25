"""Separate scene67 integration8 or locked development120; no oracle fallback."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.benchmark import _interaction_audit, _save_observation_snapshot
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.recorder import VideoRecorder, NullVideoRecorder
from VLA.common.tasks import TaskSpec, select_tasks, randomize_rl_scene
from VLA.common.artifact_io import atomic_json
from VLA.common.evaluation_statistics import success_statistics
from VLA.run_cross_vla120 import SEEDS, PROFILE_PROTOCOL
from VLA.smolvla.scene67_adapter import Scene67SmolVLA


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--smoke-source",type=Path)
    p.add_argument("--resume-from",type=Path,help="Preserve completed terminal trajectories; never reroll them")
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("No hidden reroll")
    a.output.mkdir(parents=True);profile=checkpoint_profile(a.checkpoint)
    protocol=json.loads(PROFILE_PROTOCOL.read_text())
    if profile_hash(profile)!=protocol["observation_profile_sha256"]:raise ValueError("Same robot/objects/physics/profile required")
    smoke=a.smoke_source is not None;rows=[];controls=[];preserved={}
    if a.resume_from is not None:
        if smoke:raise ValueError("Smoke integration cannot resume an episode cohort")
        from VLA.resume_scene67_evidence import recover
        rows,preserved=recover(a.resume_from.resolve(),a.output,a.checkpoint.resolve(),protocol)
    def save(status,error=None):
        atomic_json(a.output/"progress.json",{"status":status,"model":"smolvla_scene67","rows":rows,
            "neural_controls":len(controls),"controls":controls if smoke else [],"completed_trials":len(rows),
            "successes":sum(r["success"] for r in rows),"planned_trials":0 if smoke else 120,
            "integration8_NOT_model_SR":smoke,"privileged_inputs":True,"oracle_fallback":False,
            "final_test_used":False,"error":error,"checkpoint":str(a.checkpoint.resolve()),
            "resumed_from":str(a.resume_from.resolve()) if a.resume_from else None,
            "statistics":success_statistics(rows) if rows and not smoke and len(rows)%4==0 else None})
    save("running");source_rows=json.loads((a.smoke_source/"index.json").read_text()) if smoke else None
    try:
        for seed in ([20261015] if smoke else SEEDS):
            if not smoke and all((seed,t.name) in preserved for t in select_tasks("all")):continue
            if smoke:
                tasks=[TaskSpec.from_dict(next(r for r in source_rows if r["task"]["name"]==t.name)["task"])
                    for t in select_tasks("all")]
            else:
                np_rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,np_rng) for t in select_tasks("all")]
            env=environment_class_for_profile(profile)(observation_profile=profile,robot="ur5e",image_size=384,
                control_hz=10,assist_grasp=True,seed=seed,required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=protocol["success"]["semantic_thresholds"])
            policy=Scene67SmolVLA(env,str(a.checkpoint),replan_steps=1,seed=seed)
            folder=a.output/f"test_seed{seed}";folder.mkdir(exist_ok=a.resume_from is not None)
            traces=json.loads((folder/"trajectory.json").read_text()) if (folder/"trajectory.json").exists() else {}
            recorder=VideoRecorder(folder/"rollout.mp4",fps=10,overview_render=lambda:env.render("overview")) if seed in (20261015,SEEDS[0]) else NullVideoRecorder()
            try:
                with recorder:
                    for task in tasks:
                        obs=env.reset(task);policy.reset(task);trace=[];done=False
                        if (seed,task.name) in preserved:
                            # Keep the original reset sequence, but never
                            # execute a previously completed neural control.
                            continue
                        if env.is_semantic_success():raise ValueError("Initial success invalid; no replacement")
                        _save_observation_snapshot(folder,task.name,obs,env)
                        while not done and (not smoke or len(trace)<2):
                            before=_interaction_audit(env);action=policy.predict(obs)
                            if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError("Malformed neural output")
                            raw=policy.last_input["scene64"].copy();encoded=policy.last_input["numeric67"].copy()
                            obs,_,done,info=env.step(action)
                            entry={"task":task.name,"step":obs.step,"action":action.tolist(),"pre_action_scene64":raw.tolist(),
                                "actual_numeric67_input":encoded.tolist(),"before":before,"after":_interaction_audit(env),
                                "success":bool(info["success"]),"safety_violation":info.get("safety_violation")}
                            trace.append(entry);controls.append(entry);recorder.add(obs,"SmolVLA+SimScene67 | "+task.prompt)
                            if smoke or obs.step%10==0:_save_observation_snapshot(folder,task.name,obs,env)
                            if smoke and done:raise ValueError("Integration probe ended early; preserve actual actions")
                        traces[task.name]=trace;atomic_json(folder/"trajectory.json",traces)
                        if not smoke:rows.append({"seed":seed,"task":task.name,"success":bool(info["success"]),
                            "steps":len(trace),"safety_violation":info.get("safety_violation"),"trajectory":str((folder/"trajectory.json").resolve())})
                        save("running")
            finally:policy.close();env.close()
        if smoke and len(controls)!=8:raise ValueError("Incomplete actual neural integration8")
        if not smoke and len(rows)!=120:raise ValueError("Incomplete development120")
        save("completed8_integration_NOT_SR" if smoke else "completed120_privileged_input_VLA")
    except Exception as error:save("stopped_preserve_actual_trials",f"{type(error).__name__}: {error}");raise


if __name__ == "__main__":main()
