"""Fresh-process every-weight exact reload and eight native predictions."""
import argparse
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import checkpoint_contract, read


def main(a):
    import torch
    from safetensors.torch import load_file
    from .data import Data
    from .smol import Runtime
    cfg=checkpoint_contract(a.checkpoint)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(13500*2**20/torch.cuda.get_device_properties(0).total_memory)
    runtime=Runtime(a.checkpoint,cfg['stats'])
    saved=load_file(str(a.checkpoint/'model.safetensors'),device='cpu');actual=runtime.model.state_dict()
    if set(saved)!=set(actual) or any(not torch.equal(v,actual[k].cpu()) for k,v in saved.items()):
        raise ValueError('Fresh native reload changes an actual saved tensor')
    del saved
    data=Data(cfg['data']);expected=read(a.checkpoint/'reload_expected.json');errors=[]
    for (ei,fi),value in zip(expected['anchors'],expected['predictions'],strict=True):
        prediction=runtime.predict(data.at(ei,fi),expected['seed'])
        errors.append(float(abs(prediction-np.asarray(value,np.float32)).max()))
    if len(errors)!=8 or max(errors)>1e-6:raise ValueError('Actual native train/save/fresh-infer parity failed')
    atomic_json(a.output,{'passed':True,'checkpoint':str(a.checkpoint.resolve()),'actual_neural_predictions':8,
        'all_saved_values_exact':True,'max_action_error':max(errors),'weights_sha256':sha256_file(a.checkpoint/'model.safetensors'),
        'GPU_gate_updates':cfg['added_optimizer_updates'],'success_rate_established':False})


if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
