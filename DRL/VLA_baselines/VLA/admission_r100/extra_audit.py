"""Independent runtime-from-physical-payload parity, not reused cached prefixes."""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from .common import OUT,read,atomic_json,utc
from VLA.new_vlas.action_contracts import native8,rotation_matrix


def main(family):
    if family=='gr00t_n15':
        import torch
        from .groot_native import Data,imports,isolate_unused_radio,native_batch
        imports();isolate_unused_radio()
        from gr00t.model.transforms import GR00TTransform
        from gr00t.data.embodiment_tags import EmbodimentTag
        transform=GR00TTransform(max_state_dim=80,max_action_dim=32,state_horizon=1,action_horizon=16,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,training=True)
    else:from .seer_data import Data
    data=Data();maximum=0.;cases=[];frames=0
    for ei,row in enumerate(data.rows):
        n=data.numeric(ei);state=n['state20'];columns=state[:,3:9][:,[0,2,4,1,3,5]]
        if family=='gr00t_n15':
            calculated=native8(state[:,:3],columns,n['scene64'][:,27])[:,:6]
            expected=n['native8'][:,:6]
        else:
            calculated=np.c_[state[:,:3],Rotation.from_matrix(rotation_matrix(columns)).as_euler('xyz'),
                (n['scene64'][:,6]<.5).astype(np.float32)]
            expected=data.converted(ei)['native_state']
        error=float(abs(calculated-expected).max());maximum=max(maximum,error);frames+=len(state)
        if error>2e-5:raise ValueError('Actual physical-payload native state differs from cached training state')
        if family=='gr00t_n15' and row['source_kind']=='complete_original' and not any(c['task']==row['task'] for c in cases):
            for fi in (0,len(state)//2,len(state)-1):
                s=data.at(ei,fi);runtime={k:v for k,v in s.items() if k!='native_prefix6'}
                runtime.update(ee_position=state[fi,:3],ee_rotation_6d=columns[fi])
                a,_=native_batch(s,read(OUT/family/'DATA_STATS.json'),transform,True)
                b,_=native_batch(runtime,read(OUT/family/'DATA_STATS.json'),transform,False)
                for key in ('state','eagle_input_ids','eagle_pixel_values','eagle_attention_mask'):
                    if not torch.allclose(a[key],b[key],atol=2e-5,rtol=0):raise ValueError('Cached native prefix hid runtime preprocessing mismatch: '+key)
                cases.append({'task':row['task'],'episode':ei,'frame':fi,'physical_payload_not_cached_prefix':True})
    atomic_json(OUT/family/'PHYSICAL_PAYLOAD_TRAIN_RUNTIME_PARITY.json',{'passed':True,'UTC':utc(),
        'actual_TRAIN_trajectories':len(data.rows),'actual_TRAIN_frames':frames,'max_state_error':maximum,
        'native_processor_physical_payload_cases':cases,'all_states_recomputed_from_physical_pose_not_reused_native_prefix':True,
        'new_neural_success_rate_NOT_implied':True})
    print({'passed':True,'family':family,'frames':frames,'max_error':maximum,'processor_cases':len(cases)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('seer','gr00t_n15'),required=True)
    main(p.parse_args().family)
