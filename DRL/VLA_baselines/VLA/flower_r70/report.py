"""Per-action measured results and trace-grounded failure labels, not guesses."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.flower_r70.common import read,OUT


def classify(row,trace):
    if row['success']:return 'success'
    if row.get('safety_violation'):return 'safety_abort'
    if any(t['after']['held_object'] is not None and not t['after']['correct_object_held'] for t in trace):
        return 'wrong_object_grasp'
    if row.get('events',{}).get('drop') is not None:return 'grasp_lost'
    if any(t['after']['correct_object_held'] for t in trace):return 'goal_not_reached_after_grasp'
    raw=np.asarray([t['post_action_scene64'] for t in trace])
    if len(raw) and np.any(raw[:,28:30]):return 'target_contact_without_completed_task'
    if len(raw) and np.min(np.linalg.norm(raw[:,11:13],axis=1))>.08:
        return 'target_approach_localization_miss'
    return 'timeout_without_verified_grasp_or_touch'


def main(a):
    result={};lines=['# FLOWER — measured development/confirmation by action','',
        'These are actual development trials, not final800 or five independent training seeds.',
        'PUT uses the same goal-region first-hit criterion as X-VLA; it does not require release/settle.','']
    for name in ('rolling_development40','final_development40','final_confirmation40'):
        folder=a.output/name
        if not (folder/'progress.json').exists():continue
        p=read(folder/'progress.json');errors=Counter();cases=[]
        for row in p['rows']:
            trace=read(row['trajectory'])[row['task']]
            label=classify(row,trace);errors[label]+=1;cases.append({**row,'observed_failure_label':label})
        result[name]={'checkpoint':p['checkpoint'],'completed_trials':p['completed_trials'],
            'planned_trials':p['planned_trials'],'per_action':p['per_action'],'failure_counts':dict(errors),'cases':cases}
        lines+=['## '+name,'','|Action|Successes / trials|Measured SR|','|---|---:|---:|']
        for task,v in p['per_action'].items():
            sr='not measured' if v['SR_percent'] is None else f"{v['SR_percent']:.1f}%"
            lines.append(f"|{task}|{v['successes']}/{v['n']}|{sr}|")
        lines+=['','Failure labels describe trajectories, not a demonstrated internal reasoning or memory defect.',
                'Counts: '+', '.join(f'{k}={v}' for k,v in errors.items()),'']
    atomic_json(a.output/'MEASURED_PER_ACTION.json',result)
    (a.output/'MEASURED_REPORT.md').write_text('\n'.join(lines)+'\n')
    print({k:v['per_action'] for k,v in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);main(p.parse_args())
