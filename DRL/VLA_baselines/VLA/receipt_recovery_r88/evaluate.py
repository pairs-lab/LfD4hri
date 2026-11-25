"""Resume the243 immutable completed outcomes; same native physical loop.

Only receipt reference arithmetic and resume bookkeeping change. Three completed
tasks of the partially finished scene are command-replayed to restore exact reset
history, never counted as extra model trials. Remaining tasks use actual Dita.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import requests
from VLA.common.artifact_io import atomic_json
from VLA.recovery_r85.evaluate import atomic_json as safe_atomic_json
from VLA.dita800_r87.evaluate import authorized_frozen
from VLA.scene64_conditioning import model_input
from VLA.cross_vla800_r62.media import discard_snapshot
from VLA.privileged_visual_r53.scene_map import task_from_instruction
from .map_reference import render

ROOT=Path('/work/LfD4hri/DRL')
SOURCE=ROOT/'VLA/audits/dita800_user_r87_20261021/final800/progress.json'
OUT=ROOT/'VLA/audits/receipt_recovery_r88_20261021'


def ledger(source):
    data=json.loads(Path(source).read_text());rows=copy.deepcopy(data['rows'])
    keys={(r['seed'],r['task']) for r in rows}
    if len(keys)!=len(rows) or len(rows)!=243:raise ValueError('Pinned243 complete outcomes required, no invented/duplicate trials')
    return data,rows,{(r['seed'],r['task']):r for r in rows}


def compile_resume(old,initial_ledger,pending_seeds,restore_completed,logged_requests,write_json):
    tree=ast.parse(inspect.getsource(old.main))
    class ResumeOnly(ast.NodeTransformer):
        counts={'eligibility':0,'ledger':0,'seeds':0,'completed_tasks':0}
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='method' and node.func.attr=='get' and len(node.args)==1
                and isinstance(node.args[0],ast.Constant) and node.args[0].value=='frozen_after_confirmation'):
                self.counts['eligibility']+=1
                return ast.copy_location(ast.Call(func=ast.Name(id='authorized_frozen',ctx=ast.Load()),
                    args=[ast.Name(id='method',ctx=ast.Load()),ast.Name(id='cohort',ctx=ast.Load())],keywords=[]),node)
            return node
        def visit_Assign(self,node):
            self.generic_visit(node)
            target=node.targets[0] if len(node.targets)==1 else None
            if isinstance(target,ast.Tuple) and [v.id if isinstance(v,ast.Name) else None for v in target.elts]==['rows','queries','controls']:
                self.counts['ledger']+=1
                node.value=ast.Call(func=ast.Name(id='initial_ledger',ctx=ast.Load()),args=[],keywords=[])
            return node
        def visit_For(self,node):
            self.generic_visit(node)
            if isinstance(node.target,ast.Name) and node.target.id=='seed':
                self.counts['seeds']+=1
                node.iter=ast.Call(func=ast.Name(id='pending_seeds',ctx=ast.Load()),
                    args=[ast.Name(id='cohort',ctx=ast.Load())],keywords=[])
            if isinstance(node.target,ast.Name) and node.target.id=='task':
                self.counts['completed_tasks']+=1
                node.body=ast.parse('if restore_completed(env,task,seed):\n    continue').body+node.body
            return node
    edit=ResumeOnly();tree=ast.fix_missing_locations(edit.visit(tree))
    if set(edit.counts.values())!={1}:raise ValueError('Unexpected evaluator structure: '+str(edit.counts))
    namespace={**old.__dict__,'authorized_frozen':authorized_frozen,'initial_ledger':initial_ledger,
        'pending_seeds':pending_seeds,'restore_completed':restore_completed,'requests':logged_requests,
        'render':render,'atomic_json':write_json}
    exec(compile(tree,'<R88 resume bookkeeping + native NumPy1 receipt reference only>','exec'),namespace)
    return namespace['main']


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    prior,rows,completed=ledger(SOURCE)
    controls=sum(r['steps'] for r in rows)
    old_queries={}
    with (SOURCE.parent/'NEURAL_QUERIES.jsonl').open() as f:
        for line in f:
            q=json.loads(line)
            if q['seed']==20284061 and q['task']=='pick_place':old_queries[q['index']]=q
    atomic_json(OUT/'RESUME_LEDGER.json',{'source_progress':str(SOURCE),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'reused_complete_trials':len(rows),
        'complete_trials_controls':controls,'old_incomplete_controls_excluded_from_final_trials':prior['motor_controls']-controls,
        'remaining_actual_trials':800-len(rows),'no_failure_rerolled_or_removed':True,
        'only_technically_incomplete_episode_restarted':True,'model_weights_action_inputs_physics_unchanged':True})

    def initial():return copy.deepcopy(rows),controls,controls
    def pending(cohort):return [seed for seed in cohort['seeds'] if any((seed,t) not in completed for t in old.TASKS)]
    def restore(env,task,seed):
        row=completed.get((seed,task.name))
        if row is None:return False
        obs=env.reset(task);discard_snapshot(a.output,task.name,obs,env)
        trace=json.loads(Path(row['trajectory']).read_text())[task.name]
        for entry in trace:
            raw=model_input(env,obs)['scene64']
            if not np.array_equal(raw,np.asarray(entry['pre_action_scene64'],np.float32)):
                raise ValueError('State-restore replay does not match old pre-action state')
            obs,_,done,info=env.step(np.asarray(entry['action'],np.float32))
            if not np.array_equal(model_input(env,obs)['scene64'],np.asarray(entry['post_action_scene64'],np.float32)):
                raise ValueError('State-restore replay differs from recorded actual outcome')
        with (OUT/'COMPLETED_TASK_REPLAY.jsonl').open('a') as f:
            f.write(json.dumps({'seed':seed,'task':task.name,'controls':len(trace),'all_states_exact':True,
                'restore_only_NOT_new_model_trial':True})+'\n')
        return True

    class LoggedSession(requests.Session):
        def post(self,url,**kwargs):
            response=super().post(url,**kwargs)
            if response.status_code==200:
                payload=kwargs['json'];reply=response.json()
                task=task_from_instruction(payload['instruction'])
                expected=hashlib.sha256(render(payload['scene64'],task).tobytes()).hexdigest()
                if reply.get('map_RGB_sha256')!=expected:
                    atomic_json(OUT/'RECEIPT_MISMATCH.json',{'payload':payload,'reply':reply,'expected_map_sha256':expected})
                if payload['seed']==20284061 and task=='pick_place' and payload['inference_index'] in old_queries:
                    q=old_queries[payload['inference_index']]
                    if reply['input_sha256']!=q['input_sha256'] or np.max(abs(np.asarray(reply['actions'])-np.asarray(q['actions'])))>1e-6:
                        atomic_json(OUT/'PREFIX_DIVERGENCE.json',{'index':payload['inference_index'],'payload':payload,'reply':reply})
                        raise ValueError('Restarted incomplete trial differs from actual original neural prefix')
            return response

    def write(path,value):
        if Path(path).name=='progress.json':
            value={**value,'reused_completed_trials':len(rows),'prior_progress':str(SOURCE),
                'resume_elapsed_wall_s':value['elapsed_wall_s'],
                'elapsed_wall_s':value['elapsed_wall_s']+prior['elapsed_wall_s'],
                'no_model_or_success_threshold_change':True}
        safe_atomic_json(path,value)
    compile_resume(old,initial,pending,restore,SimpleNamespace(Session=LoggedSession),write)(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('method','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
