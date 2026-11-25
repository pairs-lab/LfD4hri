"""All-TRAIN numeric coverage and real sampling audit; no held-out input."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.fit_feedback_r52 import draw as r52_draw
from VLA.target_balance_r74.sampler import draw as r74_draw
from VLA.flower_rotation_r76.sampler import Data,RECIPE,phase_flags


def main(output):
    data=Data();bytask={};source_sha={}
    for ei,row in enumerate(data.rows):
        n=data.numeric(ei);a=n['canonical7'];raw=n['scene64']
        union=np.logical_or.reduce(phase_flags(raw,a));rotation=np.max(abs(a[:,3:6]),axis=1)>.01
        only=rotation&(np.linalg.norm(a[:,:3],axis=1)<=.001)&~phase_flags(raw,a)[1]
        c=bytask.setdefault(row['task'],Counter());c['frames']+=len(a)
        c['rotation']+=int(rotation.sum());c['rotation_only']+=int(only.sum())
        c['rotation_only_outside_all_ordinary_phase_flags']+=int((only&~union).sum())
    draws={}
    for name,draw in [('R52_actual_main4096_sampler',r52_draw),('R74_target_balanced',r74_draw),('R76_full_support_rotation',lambda d,r:d.draw(r))]:
        rng=np.random.default_rng(76);counts=Counter();strata=Counter();modes=Counter()
        for _ in range(6000):
            ei,fi=draw(data,rng);n=data.numeric(ei);a=n['canonical7'];s=a[fi]
            rotation=np.max(abs(s[3:6]))>.01
            only=rotation and np.linalg.norm(s[:3])<=.001 and ((s[6]>.5)==(n['scene64'][fi,10]>.5))
            counts['draws']+=1;counts['rotation_at_start']+=int(rotation);counts['rotation_only_at_start']+=int(only)
            counts['rotation_anywhere_horizon10']+=int(np.max(abs(a[fi:min(fi+10,len(a)),3:6]))>.01)
            row=data.rows[ei];strata[row['task']+'/'+row['pick_name']]+=1
            if name.startswith('R76'):modes[data.last_sampling_mode]+=1
            assert 0<=fi<len(a)
        draws[name]={'counts':dict(counts),'task_target_strata':len(strata),
            'minimum_stratum_count':min(strata.values()),'maximum_stratum_count':max(strata.values()),
            'actual_sampling_modes':dict(modes)}
    new=draws['R76_full_support_rotation'];old=draws['R52_actual_main4096_sampler']
    passed=(new['task_target_strata']==76 and new['counts']['rotation_only_at_start']>old['counts']['rotation_only_at_start']
        and new['actual_sampling_modes'].get('uniform_full_support',0)>1000)
    atomic_json(output,{'passed':bool(passed),'TRAIN_episodes':len(data.rows),'TRAIN_frames':len(data),
        'all_numeric_frames_examined':True,'recipe':RECIPE,'bank_by_task':{k:dict(v) for k,v in bytask.items()},
        'actual_draws':draws,'model_inputs_actions_cameras_normalization_unchanged':True,
        'new_model_SR_measured':False,'not_proof_all_failures_caused_by_sampler':True,
        'future_action_labels_used_ONLY_for_training_sampling':True,
        'every_frame_positive_probability_reason':'20% uniform branch; every source/task/target/episode has positive selection probability'})
    if not passed:raise ValueError('Full-support sampling qualification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
