"""Native pi0 EEF LoRA warm continuation, explicitly NEW Adam, not resume."""
import argparse
from dataclasses import replace
from pathlib import Path
from .common import DATA,R54_BASE,hashes,read
from .sampling import draw,contract


def main(a):
    from openpi.training.weight_loaders import CheckpointWeightLoader
    import VLA.family_transfer_r54.fit_pi0 as fit
    from VLA.family_transfer_r54.common import checkpoint_contract
    parent=checkpoint_contract(a.parent)
    if parent['model']!='pi0' or not 1<=a.updates<=2000:raise ValueError('Actual trained EEF pi0 parent and bounded new budget required')
    if a.updates==2 and a.qualification is not None:raise ValueError('New warm-start gate must not claim another gate')
    build=fit.build_config
    def config(stats,output,updates):
        if stats!=parent['stats']:raise ValueError('Warm continuation cannot reinterpret native normalization')
        cfg=build(stats,output,updates)
        return replace(cfg,weight_loader=CheckpointWeightLoader(str(a.parent/'params')))
    fit.build_config=config;fit.hashes=hashes
    # Check the actual initialized trainable values against the parent's
    # saved FP32 proof. A BF16 roundtrip could otherwise look like a valid
    # warm start after being promoted to FP32 again.
    from VLA.pi0 import training
    from flax.traverse_util import flatten_dict
    import hashlib
    import numpy as np
    precision=read(parent['precision']);assertion=training.assert_fp32_training_state
    def exact_initialization(state,cfg):
        result=assertion(state,cfg)
        actual=flatten_dict(state.params.filter(cfg.trainable_filter).to_pure_dict())
        for row in precision['trainable_tensors']:
            value=np.asarray(actual[tuple(row['path'])])
            if str(value.dtype)!='float32' or hashlib.sha256(value.tobytes()).hexdigest()!=row['saved_sha256']:
                raise ValueError('Warm pi0 initialization changed a saved learned FP32 value')
        return {**result,'every_parent_learned_value_exact_at_actual_init':True}
    training.assert_fp32_training_state=exact_initialization
    # Draw is a TRAIN index/phase sampler only; no phase/future label enters
    # the unchanged native observation, flow objective, or action readout.
    fit.PiData.draw=lambda data,rng:draw(data,rng,a.sampling_profile)
    atomic=fit.atomic_json
    def save(path,value):
        if isinstance(value,dict) and value.get('native_VLA_actions_only'):
            total=parent.get('r55_cumulative_optimizer_updates',parent['added_optimizer_updates'])+a.updates
            value={**value,'new_optimizer_from_official_foundation_NOT_resume_claim':False,
                'r55_new_Adam_from_prior_trained_EEF_weights_NOT_optimizer_resume':True,
                'r55_parent':str(a.parent.resolve()),'r55_cumulative_optimizer_updates':total,
                'r55_sampler':contract(a.sampling_profile),'code':hashes()}
        atomic(path,value)
    fit.atomic_json=save
    fit.main(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('parent','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--data',type=Path,default=DATA);p.add_argument('--input-proof',type=Path,default=R54_BASE/'pi0_input_final/audit.json')
    p.add_argument('--qualification',type=Path);p.add_argument('--updates',type=int,required=True)
    p.add_argument('--sampling-profile',choices=('acquisition','transport'),required=True);main(p.parse_args())
