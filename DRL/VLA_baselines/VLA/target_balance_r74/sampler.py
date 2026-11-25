import numpy as np
from VLA.flower_r71.sampler import GraspData,MODES

RECIPE={'task_selection':'uniform4','requested_target_selection':'uniform19 BEFORE source/recovery mode',
    'non_touch_probability_transition_rehearsal':.5,'otherwise':'R52 source/phase weights conditional on selected target availability',
    'no_synthetic_examples_added':True,'input_action_normalizer_unchanged':True,
    'avoids_target_prior_change_from_five_target_recovery_oversampling':True,
    'causal_inputs_only_no_teacher_phase_or_action_advice':True}


def draw(data,rng):
    task=str(rng.choice(sorted(data.groups)))
    target=str(rng.choice(sorted(data.groups[task]['complete_original'])))
    modes=[mode for mode in MODES if data.transitions[task][mode].get(target)]
    if task!='touch' and modes and rng.random()<.5:
        mode=str(rng.choice(modes));choices=data.transitions[task][mode][target]
        ei,frames=choices[int(rng.integers(len(choices)))];fi=int(rng.choice(frames))
        data.last_sampling_mode='target_first/'+mode
        return int(ei),fi
    kinds=[k for k in ('complete_original','fresh_corrective','pose_corrective') if target in data.groups[task][k]]
    w={'complete_original':.5,'fresh_corrective':.15,'pose_corrective':.35}
    probability=np.asarray([w[k] for k in kinds]);probability/=probability.sum()
    kind=str(rng.choice(kinds,p=probability));ei=int(rng.choice(data.groups[task][kind][target]))
    n=data.numeric(ei);raw,actions=n['scene64'],n['canonical7']
    if kind=='pose_corrective' and rng.random()<.5:fi=int(rng.integers(min(len(actions),8)))
    else:
        lateral=np.linalg.norm(actions[:,:2],axis=1)>.005
        change=(actions[:,6]>.5)!=(raw[:,10]>.5)
        vertical=~lateral&(abs(actions[:,2])>.005)
        quiet=(np.linalg.norm(actions[:,:3],axis=1)<=.001)&(np.max(abs(actions[:,3:6]),axis=1)<=.001)
        phase=rng.random();flag=lateral if phase<.45 else change if phase<.65 else vertical if phase<.85 else quiet
        ids=np.flatnonzero(flag);fi=int(rng.choice(ids)) if len(ids) else int(rng.integers(len(actions)))
    data.last_sampling_mode='target_first/original/'+kind
    return ei,fi


class Data(GraspData):
    draw=draw
