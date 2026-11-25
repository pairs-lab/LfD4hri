"""Every native chunk/action is matched to its chronological actual RPC reply."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from .common import TASKS,SEEDS,read,matrix,verify,utc


def audit(model,folder):
    freeze=read(folder/'FROZEN_BEFORE_800.json');verify(freeze['checkpoint_files_sha256'])
    verify(freeze['selection_evidence_sha256'])
    p=read(folder/'episodes/progress.json');matrix(p['rows'])
    if p['checkpoint']!=freeze['method']['checkpoint']:raise ValueError('Wrong measured checkpoint')
    if p.get('planned_trials')!=800 or p.get('completed_trials')!=800:raise ValueError('Incomplete final cohort')
    journal=[json.loads(s) for s in (folder/'NEURAL_QUERY_RECEIPTS.jsonl').read_text().splitlines()]
    queries=controls=0;counts={t:Counter() for t in TASKS};lookup={(r['seed'],r['task']):r for r in p['rows']}
    for seed in SEEDS:
        traces=read(folder/f'episodes/test_seed{seed}/trajectory.json')
        for task in TASKS:
            summary=lookup[seed,task];trace=traces[task]
            if not trace or len(trace)!=summary['steps'] or bool(trace[-1]['success'])!=summary['success']:
                raise ValueError('Outcome differs between summary and actual trace')
            c=counts[task];c['trials']+=1;c['successes']+=summary['success']
            c['safety_aborts']+=bool(summary.get('safety_violation'))
            held=any(v[phase]['correct_object_held'] for v in trace for phase in ('before','after'))
            if summary['success']:
                c['success_with_any_prior_verified_hold']+=held
                c['success_without_prior_verified_hold']+=not held
            groups=[]
            for v in trace:
                q=v['neural_receipt']['query_step'] if model in ('smolvla','pi0') else v['chunk_query_step']
                if not groups or groups[-1][0]!=q:groups.append((q,[]))
                groups[-1][1].append(v)
            for _,chunk in groups:
                if queries>=len(journal):raise ValueError('Actual RPC receipt missing')
                receipt=journal[queries];queries+=1
                expected=np.asarray(receipt['actions_float32'],np.float32)
                if hashlib.sha256(expected.tobytes()).hexdigest()!=receipt['native_actions_float32_SHA256']:
                    raise ValueError('Saved native chunk hash changed')
                expected_views=3 if freeze['method']['mode']=='map3' else 2
                if receipt['native_views']!=expected_views:raise ValueError('Incorrect actual visual input count')
                if expected_views==3 and not receipt.get('GT_map_SHA256'):raise ValueError('Actual GT view receipt absent')
                for i,v in enumerate(chunk):
                    digest=v['neural_receipt']['input_sha256'] if model in ('smolvla','pi0') else v['neural_chunk_input_sha256']
                    key='action' if model in ('smolvla','pi0') else 'native_neural_action'
                    if digest!=receipt['input_sha256'] or not np.array_equal(np.asarray(v[key],np.float32),expected[i]):
                        raise ValueError('Executed prefix differs from actual native model chunk')
                controls+=len(chunk)
    if queries!=len(journal) or queries!=p.get('queries',p.get('native_queries')):
        raise ValueError('Neural query count mismatch')
    if controls!=p.get('neural_controls',p.get('motor_controls')):raise ValueError('Actual motor control count mismatch')
    if any((folder/'episodes').rglob('*.mp4')) or any((folder/'episodes').rglob('*.png')):
        raise ValueError('Unexpected final video/snapshot in no-file evaluation')
    result={'passed':True,'UTC':utc(),'model':model,'actual_trials':800,'actual_trials_each_action':200,
            'all_native_executed_prefix_values_matched_actual_RPC_chunks':True,
            'chronological_receipts_handle_duplicate_input_hashes_without_ambiguity':True,
            'actual_neural_queries':queries,'actual_motor_controls':controls,
            'checkpoint_and_selection_evidence_unchanged':True,'all_safety_aborts_in_denominator':True,
            'per_task_primary_and_posthoc_grasp_diagnostic':{t:dict(c) for t,c in counts.items()},
            'PUT_primary_requires_release_or_prior_hold':False,'no_final_video_or_snapshot_files':True,
            'model_intrinsic_failure_causation_NOT_established':True}
    atomic_json(folder/'FINAL_AUDIT.json',result)
    return p
