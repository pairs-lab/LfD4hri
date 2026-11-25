"""Two real past/current observations;10 future labels plus the native history slot."""
import numpy as np
from VLA.replacement_r84.data import normalize_actions,observation74
from VLA.privileged_visual_r53.scene_map import render

def sample(data,ei,fi,stats):
    n=data.numeric(ei);row=data.rows[ei];rgb=data.rgb(ei)
    history=np.maximum([fi-1,fi],0)
    ids=np.arange(fi-1,fi+10)
    valid=(ids>=0)&(ids<row['frames'])
    actions=n['canonical7'][np.clip(ids,0,row['frames']-1)].copy()
    actions[~valid,:6]=0
    normalized=normalize_actions(actions,stats)
    # The official OXE transform/evaluator uses OPEN01 (not signed-close).
    normalized[:,6]=(1-actions[:,6])/2
    s={'instruction':row['instruction'],'task':row['task'],
        'history74':np.stack([observation74(n['native8'][t,:6],n['scene64'][t],stats) for t in history]),
        'normalized_action11':normalized,'action_valid11':valid,
        'history_indices':history,'action_label_indices':ids,'episode':int(ei),'frame':int(fi),
        'history_images':{'front':[rgb['front_rgb'][t].copy() for t in history],
                          'wrist':[rgb['wrist_rgb'][t].copy() for t in history],
                          'map':[render(n['scene64'][t],row['task']) for t in history]}}
    if history.max()!=fi or s['normalized_action11'].shape!=(11,7) or valid.sum()<1:
        raise ValueError('Dita input/target chronology mismatch')
    return s
