"""Safely inspect authentic tensors and derive the exact released topology."""
import re
from collections import Counter
import torch
from .common import OUT,ASSETS,read,atomic_json,utc
from .assets import digest


def main():
    path=ASSETS/'seer/seer_droid.pth';receipt=read(OUT/'seer/WEIGHTS_DOWNLOADED.json')['receipt']
    if path.stat().st_size!=receipt['size'] or digest(path)!=receipt['sha256']:
        raise ValueError('Authentic Seer payload changed')
    value=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    if not isinstance(value,dict):raise ValueError('Expected authentic checkpoint dictionary')
    original=value.get('model_state_dict',value.get('state_dict',value))
    if not isinstance(original,dict) or not all(isinstance(v,torch.Tensor) for v in original.values()):
        raise ValueError('Expected native tensor state dictionary, not arbitrary pickle objects')
    state={k.removeprefix('module.'):v for k,v in original.items()}
    layers=sorted({int(m.group(1)) for k in state if (m:=re.match(r'transformer_backbone\.h\.(\d+)\.',k))})
    if layers!=list(range(len(layers))) or not layers:raise ValueError('Incomplete native transformer layers')
    topology={'transformer_layers':len(layers),'hidden_dim':state['text_projector.weight'].shape[0],
        'sequence_length':state['transformer_backbone_position_embedding'].shape[1],
        'num_resampler_query':state['perceiver_resampler.latents'].shape[0],
        'action_pred_steps':state['action_pred_token'].shape[2],
        'num_obs_token_per_image':state['obs_tokens'].shape[2]//2,
        'obs_pred':True,'transformer_heads':12,'transformer_heads_source':'native REAL script/default, not inferable uniquely from dense tensor shapes'}
    counts=Counter()
    for k,v in state.items():counts[k.split('.')[0]]+=v.numel()
    atomic_json(OUT/'seer/CHECKPOINT_TOPOLOGY.json',{'passed_tensor_inspection':True,'UTC':utc(),
        'path':str(path),'sha256':receipt['sha256'],'native_topology':topology,
        'state_keys':len(state),'state_tensor_elements':sum(v.numel() for v in state.values()),
        'per_top_module_tensor_elements':dict(counts),'contains_clip':any(k.startswith('clip_model.') for k in state),
        'contains_MAE':any(k.startswith('vision_encoder.') for k in state),
        'top_level_keys':list(value),'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()},
        'unsafe_pickle_execution_used':False,'strict_native_model_load_still_required':True})
    print({'topology':topology,'modules':dict(counts),'top_level':list(value)},flush=True)


if __name__=='__main__':main()
