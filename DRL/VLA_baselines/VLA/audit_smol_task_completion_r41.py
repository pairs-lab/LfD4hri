"""Supplement predeclared success with actual attachment-history evidence.

Does not change/relabel broad target-region move/place scores. Distinguishes
actual bilateral before-action evidence from merely retained attachment IDs;
the generic interaction-audit 'correct_object_held' is NOT itself bilateral.
No training, model inference or new robot control is executed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def analyze(progress_path,out):
    progress=json.loads(progress_path.read_text());rows=progress['rows']
    if progress.get('completed_trials')!=progress.get('planned_trials') or not rows:
        raise ValueError('Completed cohort required; no partial final report')
    out.mkdir(parents=True,exist_ok=False);cases=[];cache={};hashes={}
    for row in rows:
        path=Path(row['trajectory'])
        if path not in cache:
            cache[path]=json.loads(path.read_text());hashes[str(path)]=sha256_file(path)
        trace=cache[path][row['task']]
        if len(trace)!=row['steps'] or bool(trace[-1]['success'])!=row['success']:
            raise ValueError('Terminal result differs from actual trajectory')
        states=np.asarray([f['pre_action_scene64'] for f in trace],np.float64)
        if states.shape!=(len(trace),64) or not np.isfinite(states).all():raise ValueError('Full causal64 before actions required')
        # Correct identity in the generic interaction log is attachment only.
        attached=bool(any(f.get('before',{}).get('correct_object_held') or f.get('after',{}).get('correct_object_held') for f in trace))
        bilateral=bool(np.any(states[:,17]))
        bottom=states[:,9]+states[:,13]-.5*states[:,26]
        cases.append({'seed':row['seed'],'task':row['task'],'predeclared_success':row['success'],
            'safety_violation':row.get('safety_violation'),'correct_attachment_observed':attached,
            'verified_correct_bilateral_BEFORE_action_observed':bilateral,
            'max_before_action_bottom_lift_from_reset_m':float(np.max(bottom-bottom[0])),
            'last_after_step_bilateral_NOT_available_in_generic_audit':True,
            'pick_success_itself_requires_actual_bilateral':bool(row['success'] and row['task']=='pick')})
    summary={}
    for task in ('touch','pick','move','pick_place'):
        selected=[r for r in cases if r['task']==task];good=[r for r in selected if r['predeclared_success']]
        summary[task]={'trials':len(selected),'predeclared_successes':len(good),
            'successful_episodes_with_correct_attachment_observed':sum(r['correct_attachment_observed'] for r in good),
            'successful_episodes_with_before_action_bilateral_observed':sum(r['verified_correct_bilateral_BEFORE_action_observed'] for r in good),
            'successful_non_touch_without_any_correct_attachment_observed':sum(not r['correct_attachment_observed'] for r in good) if task!='touch' else None}
    if any(sha256_file(p)!=h for p,h in hashes.items()):raise ValueError('Completed trajectory changed during audit')
    atomic_json(out/'audit.json',{'passed_trace_integrity':True,'scope':'supplemental_completion_history_NOT_new_score_or_SR_improvement',
        'progress':str(progress_path),'trials':len(rows),'by_task':summary,'cases':cases,
        'move_place_primary_success_remains_goal_region_NO_release_required':True,
        'correct_attachment_is_NOT_equivalent_to_instantaneous_bilateral_contact':True,
        'generic_privileged_policy_input_false_means_this_audit_dict_NOT_the_separate_scene67_channel':True,
        'scene67_privileged_channel_ACTUALLY_enabled_in_these_runs':True,
        'no_primary_scores_actions_horizons_or_physics_changed':True,'GPU_models_or_robot_controls':0,
        'trajectory_sha256':hashes,'progress_sha256':sha256_file(progress_path),'implementation_sha256':sha256_file(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--progress',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.progress.resolve(),a.output.resolve())
