"""Read-only TRAIN/real-neural input-support audit; not proof of causality.

Never feeds expert actions/desired waypoints to a model. No GPU allocation.
TRAIN state payloads are independently hashed against the full projection
proof; heavyweight image-source replay is inherited and explicitly not rerun.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.scene64_conditioning import encode_numeric, NUMERIC_NAMES
from VLA.smolvla.bounded_feedback_r38 import load_stats, bound, LIMIT


def describe(values, mean, std):
    z = np.abs((values-mean)/(std+np.float32(1e-8)))
    _, changed = bound(values, mean, std)
    return {'frames':len(values), 'frames_with_any_bound10':int(changed.any(-1).sum()),
        'bound10_component_fraction':float(changed.mean()),
        'features':[{'name':name, 'saved_std':float(std[i]),
            'abs_normalized_median_p99_max':np.quantile(z[:,i], [.5,.99,1]).tolist(),
            'frames_over10':int((z[:,i]>LIMIT).sum())} for i,name in enumerate(NUMERIC_NAMES)]}


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    mean, std = load_stats(a.checkpoint)
    proof = json.loads((a.projection/'audit.json').read_text())
    if not proof.get('passed') or proof.get('split')!='train' or proof['frames']!=116257:
        raise ValueError('Complete original TRAIN projection required')
    states=[]
    for row in proof['rows']:
        path=a.projection/'states'/row['file']
        if sha256_file(path)!=row['state_payload_sha256']:raise ValueError('TRAIN state payload changed')
        with np.load(path,allow_pickle=False) as z: states.append(z['state'].copy())
    raw=np.concatenate(states)
    if len(raw)!=proof['frames']:raise ValueError('Incomplete TRAIN states')
    admission=json.loads(a.admission.read_text()); candidate=Path(admission['candidate']); corrections=[]
    if not admission['passed'] or admission['frames']!=1479:raise ValueError('Actual38 correction proof required')
    for row in admission['rows']:
        # Avoid claiming a new full image-archive integrity/replay check.
        with np.load(candidate/row['file'],allow_pickle=False) as z:
            corrections.append(z['audit_pre_action_PRIVILEGED_state64'].copy())
    correction=np.concatenate(corrections)
    if len(correction)!=admission['frames']:raise ValueError('Incomplete corrective timeline')
    train=np.stack([encode_numeric(s) for s in np.concatenate((raw,correction))])
    progress_bytes=a.progress.read_bytes(); progress=json.loads(progress_bytes); cache={}; rollout=[]; identities=[]
    source_hashes={}; terminal=[]
    for row in progress['rows']:
        path=Path(row['trajectory'])
        if path not in cache:
            payload=path.read_bytes();cache[path]=json.loads(payload);source_hashes[str(path)]=hashlib.sha256(payload).hexdigest()
        trace=cache[path][row['task']]
        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=bool(row['success']):raise ValueError('Terminal evidence changed')
        for frame in trace:
            rollout.append(encode_numeric(frame['pre_action_scene64']))
            identities.append([row['seed'],row['task'],frame['step'],bool(row['success'])])
        terminal.append((row['seed'],row['task'],bool(row['success']),row['steps']))
    values=np.stack(rollout); z=np.abs((values-mean)/(std+np.float32(1e-8)))
    worst=[]
    for i in np.argsort(z.max(0))[::-1][:15]:
        k=int(z[:,i].argmax());worst.append({'feature':NUMERIC_NAMES[i], 'maximum_abs_normalized':float(z[k,i]),
            'actual_raw_value':float(values[k,i]),'case_seed_task_step_success':identities[k]})
    result={'passed_transport_audit':True,'scope':'TRAIN_vs_saved_actual_neural_state_support_NOT_causal_model_failure_proof',
        'original_TRAIN_frames':len(raw),'correction_frames':len(correction),'completed_neural_trials_snapshot':len(terminal),
        'successes_snapshot':sum(r[2] for r in terminal),'terminal_rows_snapshot':terminal,
        'TRAIN':describe(train,mean,std),'actual_neural_rollout':describe(values,mean,std),'worst_actual_fields':worst,
        'wrong_held_TRAIN_frames':int((np.r_[raw[:,30],correction[:,30]]>0).sum()),
        'bound10_is_untrained_input_ablation_NOT_proven_fix':True,'bound10_limit_chosen_before_candidate_outcomes':LIMIT,
        'image_source_physics_replay_rerun':False,'source_projection_and_correction_proofs_inherited':True,
        'final_test_used_for_training':False,'model_or_action_or_score_changed':False,
        'source_progress_sha256':hashlib.sha256(progress_bytes).hexdigest(),'trajectory_snapshot_sha256':source_hashes,
        'bound_files':{str(p.resolve()):sha256_file(p) for p in (Path(__file__),
            Path(__file__).parent/'smolvla/bounded_feedback_r38.py', a.projection/'audit.json',a.admission,
            a.checkpoint/'policy_preprocessor_step_5_normalizer_processor.safetensors')}}
    atomic_json(a.output/'audit.json',result)
    print(json.dumps({k:result[k] for k in ('completed_neural_trials_snapshot','successes_snapshot','wrong_held_TRAIN_frames','worst_actual_fields')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','projection','admission','progress','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
