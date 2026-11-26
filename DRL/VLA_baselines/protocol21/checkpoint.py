"""Exact trainable-state save/reload. Old-width adaptations cannot be relabelled."""
import json
from pathlib import Path


def parameters(runtime):
    if hasattr(runtime, 'modules'):
        return {name+'.'+key:p for name,module in runtime.modules.items()
                for key,p in module.named_parameters() if p.requires_grad}
    return {n:p for n,p in runtime.model.named_parameters() if p.requires_grad}


def save(runtime, family, stats, directory):
    import torch
    from .data import digest
    d=Path(directory);d.mkdir(parents=True,exist_ok=False)
    # Preserve native mixed precision (notably VLANeXt-S) without lossy reloads.
    values={n:p.detach().cpu().contiguous().clone() for n,p in parameters(runtime).items()}
    if not values or any(not torch.isfinite(v).all() for v in values.values()):
        raise ValueError('No finite trainables to save')
    torch.save(values,d/'trainables.pt')
    (d/'metadata.json').write_text(json.dumps({'schema':'vla21-adaptation-v1','family':family,
        'robot_state_dim':21,'stats':stats,'sha256':digest(d/'trainables.pt')},indent=2)+'\n')


def load(runtime, family, directory, stats):
    import torch
    from .data import digest
    d=Path(directory);meta=json.loads((d/'metadata.json').read_text())
    if (meta.get('schema')!='vla21-adaptation-v1' or meta.get('family')!=family
            or meta.get('robot_state_dim')!=21 or meta.get('stats')!=stats
            or digest(d/'trainables.pt')!=meta.get('sha256')):
        raise ValueError('Wrong family, observation protocol, statistics or checkpoint bytes')
    source=torch.load(d/'trainables.pt',map_location='cpu',weights_only=True)
    targets=parameters(runtime)
    if set(source)!=set(targets):raise ValueError('Exact trainable key set required')
    # Validate ALL tensors before mutating any weights.
    for name,p in targets.items():
        value=source[name]
        if value.shape!=p.shape or value.dtype!=p.dtype or not torch.isfinite(value).all():
            raise ValueError('Invalid adaptation tensor: '+name)
    with torch.no_grad():
        for name,p in targets.items():
            p.copy_(source[name].to(p.device))
