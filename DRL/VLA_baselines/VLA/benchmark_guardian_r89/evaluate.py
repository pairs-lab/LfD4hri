"""Generalize the already qualified R88 resume bookkeeping to any committed prefix."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import requests
from VLA.recovery_r85.evaluate import atomic_json as safe_atomic_json
from VLA.scene64_conditioning import model_input
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.privileged_visual_r53.scene_map import task_from_instruction
from VLA.receipt_recovery_r88.evaluate import compile_resume
from VLA.receipt_recovery_r88.map_reference import render
from .common import complete_rows,atomic_json,read


def incomplete_prefix(journal,pending,completed):
    prefix={}
    if Path(journal).exists() and pending:
        with Path(journal).open() as f:
            for line in f:
                if not line.endswith('\n'):continue
                q=json.loads(line)
                if q['seed']==pending[0] and (q['seed'],q['task']) not in completed:
                    key=(q['seed'],q['task'],q['index'])
                    if key in prefix:raise ValueError('Duplicate incomplete native query')
                    prefix[key]=q
        tasks={(s,t) for s,t,i in prefix}
        if len(tasks)>1:raise ValueError('More than one interrupted task in chronological prefix')
        if sorted(i for s,t,i in prefix)!=list(range(len(prefix))):
            raise ValueError('Noncontiguous original native query prefix')
    return prefix


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    prior,rows,completed=complete_rows(a.resume_progress)
    controls=sum(r['steps'] for r in rows)
    journal=Path(prior['original_progress_path']).parent/'NEURAL_QUERIES.jsonl'
    pending=[s for s in range(20284001,20284201) if any((s,t) not in completed for t in old.TASKS)]
    # Verify the actual original neural prefix of only the incomplete task.
    # Incomplete JSON line after an abrupt process death is retained on disk
    # but not admitted as a complete response/control.
    prefix=incomplete_prefix(journal,pending,completed)
    if hashlib.sha256(Path(prior['original_progress_path']).read_bytes()).hexdigest()!=prior['original_progress_sha256']:
        raise ValueError('Stopped source ledger changed after resume snapshot')
    source_method=read(Path(prior['original_progress_path']).parent/'PLAN.json')['method']
    method=read(a.method)
    for key in ('weights_sha256','prefixes','family','checkpoint'):
        if method.get(key)!=source_method.get(key):raise ValueError('Frozen policy changed during technical recovery: '+key)
    atomic_json(a.output.parent/'RESUME_LEDGER.json',{'source_snapshot':str(a.resume_progress),
        'source_snapshot_sha256':hashlib.sha256(a.resume_progress.read_bytes()).hexdigest(),
        'reused_completed_trials':len(rows),'complete_control_count':controls,
        'incomplete_controls_not_counted_as_finished_trials':prior['motor_controls']-controls,
        'remaining_actual_trials':800-len(rows),'same_frozen_model_and_metric':True})
    def initial():return copy.deepcopy(rows),controls,controls
    def missing(cohort):return [s for s in cohort['seeds'] if any((s,t) not in completed for t in old.TASKS)]
    def restore(env,task,seed):
        row=completed.get((seed,task.name))
        if row is None:return False
        obs=env.reset(task);discard_snapshot(a.output,task.name,obs,env)
        trace=json.loads(Path(row['trajectory']).read_text())[task.name]
        for entry in trace:
            if not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['pre_action_scene64'],np.float32)):
                raise ValueError('Restored pre-action scene differs from recorded actual prefix')
            obs,_,done,info=env.step(np.asarray(entry['action'],np.float32))
            if not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['post_action_scene64'],np.float32)):
                raise ValueError('Restored post-action scene differs from recorded actual prefix')
        with (a.output.parent/'RESTORE_ONLY.jsonl').open('a') as f:
            f.write(json.dumps({'seed':seed,'task':task.name,'steps':len(trace),'counted_as_new_trial':False})+'\n')
        return True
    class Logged(requests.Session):
        def post(self,url,**kwargs):
            response=super().post(url,**kwargs)
            if response.status_code==200:
                p=kwargs['json'];r=response.json();task=task_from_instruction(p['instruction'])
                expected=hashlib.sha256(render(p['scene64'],task).tobytes()).hexdigest()
                if r.get('map_RGB_sha256')!=expected:
                    atomic_json(a.output.parent/'RECEIPT_MISMATCH.json',{'payload':p,'reply':r,'expected_map_sha256':expected})
                q=prefix.get((p['seed'],task,p['inference_index']))
                if q and (q['input_sha256']!=r['input_sha256'] or
                          np.max(abs(np.asarray(q['actions'])-np.asarray(r['actions'])))>1e-6):
                    atomic_json(a.output.parent/'PREFIX_DIVERGENCE.json',{'index':p['inference_index'],'payload':p,'reply':r})
                    raise ValueError('Restarted incomplete neural prefix differs; refuse to combine outcomes')
            return response
    def write(path,value):
        if Path(path).name=='progress.json':
            value={**value,'reused_completed_trials':len(rows),'prior_progress':prior['original_progress_path'],
                'resume_elapsed_wall_s':value['elapsed_wall_s'],
                'elapsed_wall_s':value['elapsed_wall_s']+prior['elapsed_wall_s'],
                'no_model_or_success_threshold_change':True}
        safe_atomic_json(path,value)
    compile_resume(old,initial,missing,restore,SimpleNamespace(Session=Logged),write)(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('method','cohort','output','resume-progress'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
