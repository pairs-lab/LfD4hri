"""Same R83 physical loop; generic verified technical resume, new locked cohort."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import requests

from .common import read, verify, metric_guard, approved_final, TASKS, atomic_json


def ledger(path, cohort):
    from VLA.autonomy_r83.common import metrics
    data = read(path)
    rows = copy.deepcopy(data['rows'])
    keys = [(r['seed'],r['task']) for r in rows]
    ordered = [(s,t) for s in cohort['seeds'] for t in TASKS]
    if (keys!=ordered[:len(keys)] or len(keys)!=len(set(keys))
        or data['completed_trials']!=len(rows) or data['planned_trials']!=len(ordered)
        or data['per_action']!=metrics(rows)):
        raise ValueError('Corrupt, duplicate or non-prefix actual outcome ledger')
    return data,rows,{(r['seed'],r['task']):r for r in rows}


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    from VLA.receipt_recovery_r88.evaluate import compile_resume
    from VLA.receipt_recovery_r88.map_reference import render
    from VLA.recovery_quality_r90.evaluate import diagnostic_class
    from VLA.continuous_r86.evaluate import observed_class
    from VLA.scene64_conditioning import model_input
    from VLA.cross_vla800_r62.media import discard_snapshot
    from VLA.privileged_visual_r53.scene_map import task_from_instruction
    method,cohort = read(a.method),read(a.cohort)
    metric_guard()
    if cohort['role']=='fresh_secondary_final800' and not approved_final(method,cohort):
        raise ValueError('Missing explicit frozen final800 authorization')
    prior,rows,completed = ledger(a.resume_progress,cohort) if a.resume_progress else ({},[],{})
    controls = sum(r['steps'] for r in rows)
    prefix = {}
    pending = [s for s in cohort['seeds'] if any((s,t) not in completed for t in TASKS)]
    if a.resume_progress:
        old_plan = read(a.resume_progress.parent/'PLAN.json')
        if old_plan['method']!=method or old_plan['cohort']!=cohort:
            raise ValueError('Resume changed checkpoint, method or cohort')
        journal = a.resume_progress.parent/'NEURAL_QUERIES.jsonl'
        if journal.exists():
            for line in journal.read_text().splitlines(keepends=True):
                if not line.endswith('\n'):
                    continue
                q = json.loads(line)
                if pending and q['seed']==pending[0] and (q['seed'],q['task']) not in completed:
                    key = q['seed'],q['task'],q['index']
                    if key in prefix:
                        raise ValueError('Duplicate incomplete query')
                    prefix[key] = q
    factory = old.environment_class_for_profile
    enabled = set(cohort['seeds'][:2]) if cohort['role']=='development_selection' else set()
    def wrapped(profile):
        base = diagnostic_class(factory(profile),a.output)
        return observed_class(base,a.output,enabled,lambda e,o:model_input(e,o)['scene64'])
    old.environment_class_for_profile = wrapped

    def initial():
        return copy.deepcopy(rows),controls,controls

    def missing(c):
        return [s for s in c['seeds'] if any((s,t) not in completed for t in TASKS)]

    def restore(env,task,seed):
        row = completed.get((seed,task.name))
        if row is None:
            return False
        obs = env.reset(task)
        discard_snapshot(a.output,task.name,obs,env)
        trace = read(row['trajectory'])[task.name]
        for entry in trace:
            if not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['pre_action_scene64'],np.float32)):
                raise ValueError('Restore pre-action scene differs from recorded state')
            obs,_,done,info = env.step(np.asarray(entry['action'],np.float32))
            if (not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['post_action_scene64'],np.float32))
                or bool(info['success'])!=entry['success'] or info.get('safety_violation')!=entry['safety_violation']):
                raise ValueError('Restore post-action/outcome differs')
        return True

    class Logged(requests.Session):
        def post(self,url,**kwargs):
            response = super().post(url,**kwargs)
            if response.status_code==200:
                p,r = kwargs['json'],response.json()
                task = task_from_instruction(p['instruction'])
                expected = hashlib.sha256(render(p['scene64'],task).tobytes()).hexdigest()
                if r.get('map_RGB_sha256')!=expected:
                    raise ValueError('Native map pixel receipt mismatch')
                q = prefix.get((p['seed'],task,p['inference_index']))
                if q and (q['input_sha256']!=r['input_sha256'] or
                    np.max(abs(np.asarray(q['actions'])-np.asarray(r['actions'])))>1e-6):
                    raise ValueError('Restarted incomplete neural prefix differs')
            return response

    def write(path,value):
        if Path(path).name=='progress.json':
            value = {**value,'reused_completed_trials':len(rows),
                'elapsed_wall_s':value['elapsed_wall_s']+prior.get('elapsed_wall_s',0),
                'no_threshold_model_or_physics_change_during_resume':True}
        from VLA.recovery_r85.evaluate import atomic_json as safe_json
        safe_json(path,value)
    loop = compile_resume(old,initial,missing,restore,SimpleNamespace(Session=Logged),write)
    # Only the eligibility callback in this NEW namespace is generalized.
    # The source AST transform still asserts its four exact bookkeeping edits.
    loop.__globals__['authorized_frozen'] = approved_final
    try:
        loop(a)
    finally:
        old.environment_class_for_profile = factory


if __name__=='__main__':
    p = argparse.ArgumentParser()
    for name in ('method','cohort','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--resume-progress',type=Path)
    main(p.parse_args())
