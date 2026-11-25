"""Same verified X-VLA native controls; explicit release-attempt gate revision.

Only a qualitative supervisor is changed: attempt opening for aligned bowl
centres within25mm instead of refusing based on a bounding-sphere heuristic.
No arm action correction and no task-success relaxation. Actual landing,
support, attachment loss and settle determine outcome, not this attempt gate.
"""
import argparse
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r57 import release as release_module
from VLA.qualitative_r57.replay import run
from .common import PRIOR,read

ORIGINAL_GATE=release_module.landing_gate


def attempt_gate(object_xy,goal_xy,place_name,place_bounds=None,object_extent=None):
    if place_bounds is not None and 'bowl' in (place_name or ''):
        xy=np.asarray(object_xy,float);low,high=map(np.asarray,place_bounds)
        error=float(np.linalg.norm(xy-np.asarray(goal_xy,float)))
        inside=bool(np.all(xy>=low[:2]+.003) and np.all(xy<=high[:2]-.003))
        return inside and error<=.025,{'kind':'aligned25mm_bowl_release_ATTEMPT_not_success',
            'actual_center_error_m':error,'maximum_center_error_m':.025,
            'no_arm_translation_or_rotation_correction':True,
            'actual_support_settle_and_visual_inside_bowl_required':True}
    return ORIGINAL_GATE(object_xy,goal_xy,place_name,place_bounds,object_extent)


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    candidate=next(r for r in read(PRIOR/'all_trace_candidates.json') if r['seed']==a.seed)
    release_module.landing_gate=attempt_gate
    atomic_json(a.output/'candidate.json',candidate)
    atomic_json(a.output/'release_protocol.json',{'variant':'R58_bowl25mm_attempt_gate',
        'source_sha256':sha256_file(Path(__file__)),'only_opening_at_existing_pose':True,
        'native_approach_carry_actions_checkpoint_camera_physics_success_unchanged':True,
        'not_pure_VLA_release_or_new_quantitative_SR':True})
    baseline=run(candidate,terminal_release=True)
    atomic_json(a.output/'baseline_replay.json',baseline)
    proof=run(candidate,a.output,baseline,terminal_release=True)
    proof['qualitative_release_attempt_protocol']=read(a.output/'release_protocol.json')
    atomic_json(a.output/'replay_proof.json',proof)
    atomic_json(a.output/'status.json',{'status':'exact_CPU_replay_complete_media_pending','seed':a.seed,
        'full_put_from_supervisor':bool((proof['terminal_release_supervisor'] or {}).get('qualitative_put_complete'))})
    print({'seed':a.seed,'ticks':proof['physics_ticks'],'terminal':
        {k:v for k,v in (proof['terminal_release_supervisor'] or {}).items() if k!='controls'}},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    try:main(a)
    except Exception as error:
        if a.output.exists():atomic_json(a.output/'error.json',{'error':f'{type(error).__name__}: {error}'})
        raise
