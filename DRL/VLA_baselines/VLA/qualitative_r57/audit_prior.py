"""Read-only frozen put-trace triage and release-label coverage, no new rollouts."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import COHORTS,CP,OUT,read,protocol


def describe(trace):
    raw=np.asarray([t['pre_action_scene64'] for t in trace],np.float32)
    held=np.asarray([t['after']['correct_object_held'] for t in trace],bool)
    commands=np.asarray([t['action'] for t in trace])
    positions=np.asarray([t['after']['object_base_xyz_m'] for t in trace])
    drops=np.flatnonzero(held[:-1]&~held[1:])+1
    close_to_goal=np.linalg.norm(raw[:,14:16],axis=1)<.09
    return {'steps':len(trace),'benchmark_success':bool(trace[-1]['success']),
        'verified_held_controls':int(held.sum()),'terminal_correct_held':bool(held[-1]),
        'hold_loss_control_indices':drops.tolist(),
        'held_open_request_controls':np.flatnonzero(held&(commands[:,6]<0)).tolist(),
        'near_goal_open_request_controls':np.flatnonzero(close_to_goal&(commands[:,6]<0)).tolist(),
        'object_max_speed_pre_action_m_per_s':float(np.linalg.norm(raw[:,32:35],axis=1).max()),
        'object_max_lift_m':float(positions[:,2].max()-positions[0,2]),
        'minimum_target_pad_XY_m':float(np.linalg.norm(raw[:,11:13],axis=1).min()),
        'terminal_goal_XY_m':float(np.linalg.norm(raw[-1,14:16])),
        'terminal_safety':trace[-1].get('safety_violation'),
        'full_put_complete_from_this_trace':False if held[-1] else None,
        'natural_drop_or_impact_NOT_confirmed_by_trace_only':True}


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    protocol();rows=[]
    for folder in COHORTS:
        progress=read(folder/'progress.json')
        if progress['checkpoint']!=str(CP):raise ValueError('Do not mine a different/worse VLA')
        for row in progress['rows']:
            if row['task']!='pick_place':continue
            trace=read(row['trajectory'])['pick_place']
            rows.append({**row,**describe(trace),'cohort':str(folder),
                'trajectory_sha256':sha256_file(row['trajectory']),
                'candidate_id':f'put_seed{row["seed"]}',
                'selection_is_qualitative_NOT_used_for_training_or_SR':True})
    from VLA.new_vlas.dataset import NativeData
    from VLA.family_transfer_r54.common import DATA
    bank=NativeData(DATA,verify_rgb_hashes=False);counts=Counter()
    for ei,row in enumerate(bank.rows):
        if row['task']!='pick_place':continue
        n=bank.numeric(ei);raw,action=n['scene64'],n['canonical7']
        near=np.linalg.norm(raw[:,14:16],axis=1)<.09
        held=raw[:,17]>.5;opened=action[:,6]<0
        counts['put_frames']+=len(action);counts['put_episodes']+=1
        counts['held_open_label_frames']+=int((held&opened).sum())
        counts['near_goal_held_open_label_frames']+=int((near&held&opened).sum())
        counts['terminal_open_label_episodes']+=int(opened[-1])
    # Shortlist for content validation, NOT preassigned success categories.
    # Prefer measured hold loss / speed / clear non-acquisition, with several
    # benchmark successes retained to audit the put-versus-hold distinction.
    priority=lambda r:(bool(r['hold_loss_control_indices']),not r['benchmark_success'],
        r['object_max_speed_pre_action_m_per_s'],r['object_max_lift_m'])
    failures=sorted([r for r in rows if not r['benchmark_success']],key=priority,reverse=True)[:6]
    successes=sorted([r for r in rows if r['benchmark_success']],key=priority,reverse=True)[:6]
    selected=failures+successes
    atomic_json(a.output/'audit.json',{'passed':True,'checkpoint':str(CP),'scope':'read_only_qualitative_trace_triage_NOT_new_SR',
        'put_traces':len(rows),'rows':rows,'TRAIN_release_label_coverage':dict(counts),
        'selected_first_batch':selected,'not_all_failures_causally_or_visually_verified_yet':True,
        'region_success_does_NOT_certify_full_put':True})
    atomic_json(a.output/'shortlist.json',selected)
    print({'put_traces':len(rows),'first_batch':len(selected),'TRAIN_release_label_coverage':dict(counts)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT/'audit/initial_trace_triage')
    main(p.parse_args())
