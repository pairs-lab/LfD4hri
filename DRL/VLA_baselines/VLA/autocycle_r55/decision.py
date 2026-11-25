"""Only explicit DEVELOPMENT cohorts may choose recipes/checkpoints."""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import read,TASKS


def qualified_progress(path):
    d=read(path)
    if d['completed_trials']!=40 or d['planned_trials']!=40 or len(d['rows'])!=40:
        raise ValueError('Require all40 actual DEVELOPMENT outcomes')
    if len({(r['seed'],r['task']) for r in d['rows']})!=40:
        raise ValueError('Duplicated development trials')
    seeds={r['seed'] for r in d['rows']}
    if seeds not in (set(range(20268011,20268021)),set(range(20268101,20268111))):
        raise ValueError('Refuse final, reserved, partial or unknown cohort for model selection')
    if any(sum(r['task']==t for r in d['rows'])!=10 for t in TASKS):raise ValueError('Unequal task exposure')
    if sum(bool(r['success']) for r in d['rows'])!=d['successes']:raise ValueError('Success receipt mismatch')
    return d


def score(path):
    d=qualified_progress(path)
    return d['successes'],-sum(bool(r.get('safety_violation')) for r in d['rows'])


def diagnose(path):
    p=qualified_progress(path);cache={};counts=Counter();examples=[]
    for row in p['rows']:
        if row['success']:continue
        file=Path(row['trajectory'])
        if file not in cache:cache[file]=read(file)
        trace=cache[file][row['task']]
        if not trace:raise ValueError('Missing actual failed trajectory')
        raw=np.asarray([v['pre_action_scene64'] for v in trace],np.float32)
        if raw.shape!=(len(trace),64) or not np.isfinite(raw).all():raise ValueError('Malformed measured failure evidence')
        held=bool(np.any(raw[:,17]>.5));contacts=bool(np.any(raw[:,28:30]>.5))
        key=('safety_abort' if row.get('safety_violation') else 'touch_distance_or_contact_miss' if row['task']=='touch'
             else 'held_then_goal_or_release_miss' if held else 'contact_without_verified_hold' if contacts else 'never_reached_correct_hold')
        counts[key]+=1
        examples.append({'seed':row['seed'],'task':row['task'],'observed_symptom_NOT_proven_intrinsic_cause':key,
            'measured_held_before_action':held,'minimum_target_XY_error_m':float(np.linalg.norm(raw[:,11:13],axis=1).min()),
            'trace':str(file)})
    acquisition=counts['never_reached_correct_hold']+counts['contact_without_verified_hold']
    profile='acquisition' if acquisition>=counts['held_then_goal_or_release_miss'] else 'transport'
    return {'source_DEVELOPMENT_progress':str(Path(path).resolve()),'failure_counts':dict(counts),
        'selected_predeclared_TRAIN_sampler_profile':profile,'rows':examples,
        'labels_are_observed_symptoms_NOT_causal_model_failure_proof':True,'final_outcomes_read':False}


def stable(candidate):
    a,b=score(candidate['development_progress']),score(candidate['confirmation_progress'])
    tasks_ok=all(sum(bool(r['success']) for r in qualified_progress(path)['rows'] if r['task']==task)>=3
                 for path in (candidate['development_progress'],candidate['confirmation_progress']) for task in TASKS)
    return min(a[0],b[0])>=24 and min(a[1],b[1])>=-4 and tasks_ok


def candidate_key(candidate):
    a,b=score(candidate['development_progress']),score(candidate['confirmation_progress'])
    return a[0]+b[0],min(a[0],b[0]),a[1]+b[1],bool(candidate.get('preserved_baseline'))


def continue_after_round(previous,current,round_number):
    # A second expensive continuation needs an actual gain, not only a lower
    # loss or a demand to keep the machine occupied indefinitely.
    if round_number>=2 or stable(current):return False
    return candidate_key(current)[0]>=candidate_key(previous)[0]+2
