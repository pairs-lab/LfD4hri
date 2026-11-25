"""Fixed DEVELOPMENT action replay checks the no-file shim, not a model SR."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT,TASKS,read
from .media import discard_snapshot


def run(method,out,original):
    from VLA.qualitative_r57.common import make_env,tasks
    from VLA.benchmark import _save_observation_snapshot,_interaction_audit
    from VLA.scene64_conditioning import model_input
    progress=read(method['development_progress']);seed=min(r['seed'] for r in progress['rows'])
    source=next(r['trajectory'] for r in progress['rows'] if r['seed']==seed)
    traces=read(source);records=[];outcomes=[];env=make_env(seed)
    try:
        for task in tasks(seed):
            obs=env.reset(task)
            if original:_save_observation_snapshot(out/'original_snapshot_probe',task.name,obs,env)
            else:discard_snapshot(out,task.name,obs,env)
            for row in traces[task.name]:
                raw=model_input(env,obs)['scene64']
                values=(obs.front_rgb,obs.wrist_rgb,obs.joint_positions,obs.ee_position,obs.ee_rotation_6d,raw)
                records.append({'task':task.name,'step':int(obs.step),
                    'hashes':[hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in values],
                    'interaction':_interaction_audit(env)})
                obs,_,done,info=env.step(np.asarray(row['action'],np.float32))
                # Native evaluator records its first seed only. Preserve the
                # original renderer call in BOTH passes; only file IO differs.
                env.render('overview')
            outcomes.append({'task':task.name,'steps':int(obs.step),'success':bool(info['success'])})
    finally:env.close()
    return records,outcomes,source


def main(a):
    a.output.mkdir(parents=True,exist_ok=False);method=read(a.method)['method']
    first,fo,source=run(method,a.output,True);second,so,_=run(method,a.output,False)
    atomic_json(a.output/'native_snapshot_and_render.json',first)
    atomic_json(a.output/'discard_file_shim.json',second)
    passed=first==second and fo==so and len(first)>0
    atomic_json(a.output/'audit.json',{'passed':passed,'controls_checked_each_pass':len(first),
        'all_step_RGB_numeric_and_interaction_bit_exact':first==second,'outcomes_equal':fo==so,
        'genuine_DEVELOPMENT_actions':str(source),'trajectory_sha256':sha256_file(source),
        'not_new_model_SR_trials':True,'original_overview_cadence_preserved':True,
        'snapshot_masks_preserved_without_saving_final_snapshots':True,
        'no_video_file_created':not any(a.output.rglob('*.mp4')),'outcomes':fo})
    if not passed:raise ValueError('No-file shim changed a policy image/state/interaction/outcome')
    print({'passed':True,'all_control_steps_checked':len(first)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
