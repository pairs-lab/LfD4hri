"""Actual R98 diagnosis and explicitly conditional five-cohort prediction."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from .common import OUT,SOURCE,TASKS,read,atomic_json,utc
from VLA.common.provenance import sha256_file
from VLA.threshold_sensitivity_r96.analyze import forecast


def main():
    OUT.mkdir(exist_ok=True)
    certificate=read(SOURCE/'CERTIFIED_RESULT.json')
    source=Path(certificate['source'])
    if sha256_file(source)!=certificate['source_sha256'] or not certificate['audit_passed']:
        raise ValueError('Require unchanged certified R98 result')
    data=read(source);rows=data['rows']
    if len(rows)!=800 or len({(r['seed'],r['task']) for r in rows})!=800:
        raise ValueError('Require800 unique actual episodes')
    result={};diagnostics=defaultdict(Counter);evidence=[]
    for task in TASKS:
        subset=[r for r in rows if r['task']==task]
        k=sum(r['success'] for r in subset)
        if len(subset)!=200 or k!=certificate['per_action'][task]['successes']:
            raise ValueError('Raw per-action counts differ from certificate')
        result[task]=forecast(k,n=200,rng_seed=202699+TASKS.index(task))
    cached=None;traces=None
    for row in rows:
        if cached!=row['trajectory']:
            cached=row['trajectory'];traces=read(cached)
        trace=traces[row['task']]
        if len(trace)!=row['steps'] or trace[-1]['success']!=row['success']:
            raise ValueError('Outcome/trajectory mismatch')
        pre=np.asarray([e['pre_action_scene64'] for e in trace],np.float32)
        post=np.asarray([e['post_action_scene64'] for e in trace],np.float32)
        actions=np.asarray([e['action'] for e in trace],np.float32)
        held=(pre[:,17]>.5)|(post[:,17]>.5)
        close=actions[:,6]>0
        changes=np.flatnonzero(close[1:]!=close[:-1])+1
        ever=bool(held.any());safety=bool(row['safety_violation'])
        count=diagnostics[row['task']]
        count['episodes']+=1;count['successes']+=int(row['success'])
        count['safety_aborts']+=int(safety)
        count['ever_correct_bilateral_hold']+=int(ever)
        count['failed_without_ever_correct_hold']+=int(not row['success'] and not ever)
        count['failed_after_correct_hold']+=int(not row['success'] and ever)
        count['lost_hold_any_time']+=int(np.any((pre[:,17]>.5)&(post[:,17]<.5)))
        count['held_open_command_steps']+=int(np.sum((pre[:,17]>.5)&~close))
        count['gripper_sign_changes']+=len(changes)
        count['rapid_gripper_reversal_within2steps']+=int(np.sum(np.diff(changes)<=2))
        count['low_empty_descend_with_xy_error_over2cm_steps']+=int(np.sum(
            (pre[:,31]<.5)&(pre[:,9]<.14)&(np.linalg.norm(pre[:,11:13],axis=1)>.02)&(actions[:,2]<-.005)))
        evidence.append({**row,'ever_correct_hold':ever,'gripper_sign_changes':len(changes),
            'terminal_target_relative_xyz':pre[-1,11:14].tolist(),
            'terminal_requested_action':actions[-1].tolist()})
    atomic_json(OUT/'FIVE_COHORT_PREDICTION.json',{'UTC':utc(),'source':str(source),
        'source_sha256':sha256_file(source),'checkpoint':certificate['checkpoint'],
        'per_action':result,'assumptions':['same frozen R98 checkpoint and200 episodes/action/cohort',
        'independent exchangeable scenes conditional on shared unknown p',
        'one observed cohort plus four unrun cohorts; JeffreysBeta(.5,.5)',
        'no model-training seed variance estimated; no expected gain from new refinement included'],
        'measured_five_seed_mean_or_std':False,'not_robot_rollouts':True})
    atomic_json(OUT/'R98_TRACE_DIAGNOSIS.json',{'UTC':utc(),'source_sha256':sha256_file(source),
        'per_action':{k:dict(v) for k,v in diagnostics.items()},'episodes':evidence,
        'R98_is_development_for_R99_not_new_final_holdout':True,
        'causal_model_architecture_failure_NOT_proven_by_state_diagnostics':True})
    text='# R98: conditional five-cohort forecast, NOT measured five seeds\n\n'
    text+='One observed cohort + four future cohorts,200 episodes/action, SAME frozen checkpoint.\n'
    text+='Mean is posterior predictive expectation; ± is median predicted sample SD (ddof1), not a confidence interval.\n\n'
    text+='|Action|Actual successes|Predicted five-cohort mean ± SD (percentage points)|95% predictive interval of mean|\n|---|---:|---:|---:|\n'
    for task,r in result.items():
        lo,hi=r['predictive_mean_95_percent']
        text+=f"|{task}|{r['observed_successes']}/200|{r['expected_five_cohort_mean_percent']:.2f} ± {r['median_predictive_sample_std_percentage_points']:.2f}|[{lo:.1f}, {hi:.1f}]|\n"
    text+='\nThese decimal estimates are calculations, not invented seed outcomes. Actual future five-cohort means are quantized in0.1 percentage-point steps.\n'
    text+='Scene correlations, domain changes and retraining may increase variation; one cohort cannot establish training-seed variability.\n'
    text+='The new R99 checkpoint requires its own measured final800 and cannot inherit these forecasts.\n'
    (OUT/'PREDICTION.md').write_text(text)
    print({k:{'mean':v['expected_five_cohort_mean_percent'],'predicted_SD':v['median_predictive_sample_std_percentage_points']}
        for k,v in result.items()},flush=True)
    print({k:dict(v) for k,v in diagnostics.items()},flush=True)


if __name__=='__main__':main()
