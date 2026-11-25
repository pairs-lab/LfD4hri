"""Fixed recorded neural-action stresses, NOT new learned robot/SR trials."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.mimic_lag_guard import install,ACTIVATION_RAD
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import PROFILE_PROTOCOL
from VLA.simstate_feedback import feedback_state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--progress",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--cases",type=int,default=6);a=p.parse_args()
    if a.output.exists():raise FileExistsError("Preserve every mechanical experiment")
    a.output.mkdir(parents=True);progress=json.loads(a.progress.read_text())
    selected=[r for r in progress["rows"] if r.get("safety_violation") and "mimic_error" in r["safety_violation"]][:a.cases]
    if not selected:raise ValueError("No actual completed linkage-abort traces to stress")
    protocol=json.loads(PROFILE_PROTOCOL.read_text());profile=json.loads(Path(protocol["observation_profile_file"]).read_text())
    atomic_json(a.output/"input_completed_rows_snapshot.json",selected)
    results=[]
    for row in selected:
        # Snapshot only this completed task: other task/seed files can continue
        # accumulating under their existing coordinator without interruption.
        trace=json.loads(Path(row["trajectory"]).read_text())[row["task"]]
        rng=np.random.default_rng(row["seed"]);tasks=[randomize_rl_scene(t,rng) for t in select_tasks("all")]
        task=next(t for t in tasks if t.name==row["task"])
        for governed in (False,True):
            env=environment_class_for_profile(profile)(observation_profile=profile,robot="ur5e",image_size=384,
                control_hz=10,assist_grasp=True,seed=row["seed"],required_success_steps=1,use_semantic_success=True,
                success_threshold_scale=protocol["success"]["reward_xml_threshold_scale"],
                semantic_success_thresholds=protocol["success"]["semantic_thresholds"])
            result={"seed":row["seed"],"task":row["task"],"governed":governed,"steps":0,"error":None,
                "safety_violation":None,"baseline_before_state18_max_error":0.}
            try:
                obs=env.reset(task)
                if governed:install(env.gripper)
                for entry in trace:
                    if not governed:
                        error=float(np.max(np.abs(feedback_state(env,obs)-entry["pre_action_PRIVILEGED_state18"])))
                        result["baseline_before_state18_max_error"]=max(result["baseline_before_state18_max_error"],error)
                    obs,_,done,info=env.step(np.asarray(entry["action"],np.float32));result["steps"]+=1
                    if info.get("safety_violation"):
                        result["safety_violation"]=info["safety_violation"];break
                result.update(mimic_error_max_rad=env.safety_max_mimic_error,
                    governor_ticks=getattr(env.gripper,"mimic_governor_ticks",0),
                    actual_fixed_commands=len(trace),task_success_NOT_neural_trial=bool(info["success"]))
            except Exception as error:result["error"]=f"{type(error).__name__}: {error}"
            finally:env.close()
            results.append(result);atomic_json(a.output/"progress.json",{"rows":results,"new_model_SR":None})
            print(json.dumps(result),flush=True)
    atomic_json(a.output/"audit.json",{"scope":"fixed_recorded_action_mechanical_diagnostic_NOT_new_model_SR",
        "rows":results,"governor_activation_rad":ACTIVATION_RAD,"new_neural_controls":0,
        "existing_safety_thresholds_unchanged":True,"production_deployment_qualified":False,
        "guard_implementation_sha256":sha256_file(Path(__file__).parent/"common/mimic_lag_guard.py"),
        "input_snapshot_sha256":sha256_file(a.output/"input_completed_rows_snapshot.json")})


if __name__ == "__main__":main()
