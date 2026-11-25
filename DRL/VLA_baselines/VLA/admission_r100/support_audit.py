"""All-frame TRAIN support and saturation diagnostics, no held-out labels."""
from collections import Counter,defaultdict
import hashlib
import numpy as np
from .common import OUT,atomic_json,utc
from VLA.refinement_r99.data import Data
from VLA.recovery_quality_r90.audit import regimes


def main():
    data=Data();tasks=defaultdict(Counter);sources=defaultdict(Counter);groups=set();targets=defaultdict(set)
    matrices=[];actions=[];first=[]
    for ei,row in enumerate(data.rows):
        n=data.numeric(ei);raw=n['scene64'];action=n['canonical7']
        groups.add(row['scene_group']);targets[row['task']].add(row['pick_name'])
        tasks[row['task']]['trajectories']+=1;tasks[row['task']]['frames']+=len(raw)
        sources[row['source_kind']]['trajectories']+=1;sources[row['source_kind']]['frames']+=len(raw)
        for key,flag in regimes(raw,action).items():tasks[row['task']][key]+=int(flag.sum())
        for sign,name in ((1,'close_commands'),(-1,'open_commands')):
            tasks[row['task']][name]+=int(((action[:,6]*sign)>.5).sum())
        matrices.append(raw);actions.append(action);first.append(raw[0])
    raw=np.concatenate(matrices);a=np.concatenate(actions);mean=np.asarray(data.stats['scene67']['mean'])
    std=np.asarray(data.stats['scene67']['std'])
    # Exact same sin/cos expansion as the native conditioning, vectorized
    # here only to audit125k frames without changing any model preprocessing.
    numeric=np.c_[raw[:,:35],np.stack((np.sin(raw[:,35:38]),np.cos(raw[:,35:38])),axis=-1).reshape(-1,6),raw[:,38:]]
    saturated=abs((numeric-mean)/std)>5
    if len(raw)!=125527 or len(data.rows)!=2337 or any(len(v)!=19 for v in targets.values()):
        raise ValueError('Complete four-task/target TRAIN support changed')
    value={'passed_support_inventory':True,'UTC':utc(),'split':'TRAIN only','trajectories':len(data.rows),
        'frames':len(raw),'unique_scene_groups':len(groups),
        'trajectory_variants_NOT_claimed_as_independent_scene_demonstrations':True,
        'per_action':dict(tasks),'per_source_kind':dict(sources),'targets_per_action':{k:sorted(v) for k,v in targets.items()},
        'action_min':a.min(0).tolist(),'action_max':a.max(0).tolist(),
        'first_state_sha256':hashlib.sha256(np.asarray(first,np.float32).tobytes()).hexdigest(),
        'causal67_regularized_standardized_abs_gt5_fraction_per_feature':saturated.mean(0).tolist(),
        'fraction_frames_with_any_clamped_causal_feature':float(saturated.any(1).mean()),
        'heldout_or_future_observations_added':False,'TRAIN_loss_or_support_DOES_NOT_prove_generalization':True}
    atomic_json(OUT/'TRAIN_SUPPORT_INVENTORY.json',value)
    print({'trajectories':len(data.rows),'frames':len(raw),'unique_scene_groups':len(groups),
        'any_feature_clamp_fraction':value['fraction_frames_with_any_clamped_causal_feature']},flush=True)


if __name__=='__main__':main()
