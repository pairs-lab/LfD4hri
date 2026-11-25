"""Two predeclared TRAIN-only exposure fixes; no feedback controller."""
from collections import Counter
import numpy as np

PROFILES={
    'acquisition':{'phases':[.55,.30,.10,.05],'corrective_first8':.75},
    'transport':{'phases':[.40,.15,.35,.10],'corrective_first8':.25}}


def contract(profile):
    if profile not in PROFILES:raise ValueError('Unknown predeclared sampler')
    return {'schema':'R55_equal_task_target_TRAIN_rehearsal_50_15_35_'+profile,
        **PROFILES[profile],'all_original_frames_retained':True,
        'phase_labels_and_future_actions_NEVER_model_inputs':True}


def draw(data,rng,profile):
    spec=PROFILES[profile];task=rng.choice(sorted(data.groups));p=rng.random()
    kind='pose_corrective' if p<.35 else 'fresh_corrective' if p<.50 else 'complete_original'
    choices=data.groups[task][kind];target=rng.choice(sorted(choices));ei=int(rng.choice(choices[target]))
    n=data.numeric(ei);raw,action=n['scene64'],n['canonical7']
    if kind=='pose_corrective' and rng.random()<spec['corrective_first8']:
        fi=int(rng.integers(0,min(len(action),8)))
    else:
        lateral=np.linalg.norm(action[:,:2],axis=1)>.005
        grip=(action[:,6]>.5)!=(raw[:,10]>.5)
        vertical=~lateral&(abs(action[:,2])>.005)
        quiet=(np.linalg.norm(action[:,:3],axis=1)<=.001)&(np.max(abs(action[:,3:6]),axis=1)<=.001)
        flags=(lateral,grip,vertical,quiet);selected=flags[int(rng.choice(4,p=spec['phases']))]
        ids=np.flatnonzero(selected);fi=int(rng.choice(ids)) if len(ids) else int(rng.integers(len(action)))
    return ei,fi


def audit(data,profile):
    counts=Counter();rng=np.random.default_rng(202655)
    for _ in range(4000):
        ei,fi=draw(data,rng,profile);r=data.rows[ei];a=data.numeric(ei)['canonical7'][fi]
        counts['task_'+r['task']]+=1;counts['source_'+r['source_kind']]+=1
        counts['XY_over5mm']+=int(np.linalg.norm(a[:2])>.005)
        counts['rotation_nonzero']+=int(np.max(abs(a[3:6]))>1e-5)
    if min(counts['task_'+t] for t in data.groups)<700 or counts['XY_over5mm']<600:
        raise ValueError('Actual R55 task/motion exposure gate failed')
    return {'passed':True,'draws':4000,'contract':contract(profile),'counts':dict(counts),
        'CPU_sampling_only_NOT_neural_success':True}
