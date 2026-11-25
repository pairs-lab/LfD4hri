from collections import defaultdict
import numpy as np
from VLA.target_balance_r74.sampler import Data as PreviousData
from VLA.flower_r71.sampler import MODES

RECIPE={'task':'uniform4','target':'uniform19 before source/mode',
    'every_frame_nonzero_probability':True,'uniform_start_probability':.20,
    'non_touch_rotation_start_probability':.25,'non_touch_grasp_transition_probability':.25,
    'non_touch_ordinary_phase_probability':.30,'touch_ordinary_phase_probability':.80,
    'rotation_label_threshold_rad':.01,
    'missing_modes_fall_back_to_uniform_same_task_target':True,
    'model_inputs_native_labels_normalizer_images_physics_unchanged':True,
    'teacher_phase_and_future_actions_NEVER_added_to_policy_input':True,
    'new_demonstrations_added':False}


def phase_flags(raw,a):
    lateral=np.linalg.norm(a[:,:2],axis=1)>.005
    change=(a[:,6]>.5)!=(raw[:,10]>.5)
    vertical=~lateral&(abs(a[:,2])>.005)
    quiet=(np.linalg.norm(a[:,:3],axis=1)<=.001)&(np.max(abs(a[:,3:6]),axis=1)<=.001)
    return lateral,change,vertical,quiet


class Data(PreviousData):
    def __init__(self):
        super().__init__()
        self.rotations=defaultdict(lambda:defaultdict(list))
        for ei,row in enumerate(self.rows):
            a=self.numeric(ei)['canonical7'];ids=np.flatnonzero(np.max(abs(a[:,3:6]),axis=1)>.01)
            if len(ids):self.rotations[row['task']][row['pick_name']].append((ei,ids))

    def draw(self,rng):
        task=str(rng.choice(sorted(self.groups)))
        target=str(rng.choice(sorted(self.groups[task]['complete_original'])))
        kinds=[k for k in ('complete_original','fresh_corrective','pose_corrective') if target in self.groups[task][k]]
        weights={'complete_original':.5,'fresh_corrective':.15,'pose_corrective':.35}
        probability=np.asarray([weights[k] for k in kinds]);probability/=probability.sum()
        kind=str(rng.choice(kinds,p=probability));ei=int(rng.choice(self.groups[task][kind][target]))
        n=self.numeric(ei);a=n['canonical7'];u=rng.random()
        mode='uniform_full_support'
        if u>=.20 and task!='touch' and u<.45 and self.rotations[task].get(target):
            choices=self.rotations[task][target];ei,ids=choices[int(rng.integers(len(choices)))]
            fi=int(rng.choice(ids));mode='rotation_start'
        elif u>=.45 and task!='touch' and u<.70:
            modes=[m for m in MODES if self.transitions[task][m].get(target)]
            if modes:
                m=str(rng.choice(modes));choices=self.transitions[task][m][target]
                ei,ids=choices[int(rng.integers(len(choices)))];fi=int(rng.choice(ids));mode=m
            else:fi=int(rng.integers(len(a)))
        elif u>=.70 or (task=='touch' and u>=.20):
            if kind=='pose_corrective' and rng.random()<.5:fi=int(rng.integers(min(len(a),8)))
            else:
                flags=phase_flags(n['scene64'],a);v=rng.random()
                flag=flags[0] if v<.45 else flags[1] if v<.65 else flags[2] if v<.85 else flags[3]
                ids=np.flatnonzero(flag);fi=int(rng.choice(ids)) if len(ids) else int(rng.integers(len(a)))
            mode='ordinary_phase/'+kind
        else:fi=int(rng.integers(len(a)))
        self.last_sampling_mode=mode
        return int(ei),int(fi)
