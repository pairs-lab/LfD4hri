"""Strict classification loading and explicit pretrain-to-classifier transfer."""
import argparse
import hashlib
from pathlib import Path
import pickle
import torch


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_checkpoint(path, expected_sha256=None):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError('Supply an MVD-S SSV2 fine-tuned checkpoint; no random-model fallback: '+str(path))
    checksum = sha256(path)
    if expected_sha256 and checksum.lower() != expected_sha256.lower():
        raise ValueError('Checkpoint SHA-256 mismatch')
    try:
        # Native upstream checkpoints store argparse.Namespace as metadata.
        with torch.serialization.safe_globals([argparse.Namespace]):
            payload = torch.load(path,map_location='cpu',weights_only=True)
    except pickle.UnpicklingError as exc:
        raise ValueError('Checkpoint is not weights-only compatible; export trusted model tensors separately. '
                         'Unsafe pickle loading is intentionally disabled.') from exc
    if not isinstance(payload, dict):
        raise ValueError('Expected a state dict or a checkpoint containing model/module/state_dict')
    values = next((payload[k] for k in ('model','module','state_dict')
                   if k in payload and isinstance(payload[k],dict)),payload)
    if not values or not all(isinstance(v,torch.Tensor) for v in values.values()):
        raise ValueError('Checkpoint does not contain a tensor-only model state')
    state = {}
    for key,value in values.items():
        key = key.removeprefix('module.')
        if key in state:
            raise ValueError('Duplicate model keys after DDP prefix removal')
        state[key] = value
    return state, {'path':str(path),'sha256':checksum,
                   'metadata':payload.get('mvd_metadata',{}) if payload is not values else {}}


def load_classifier(model, state):
    shape = (174,384)
    if 'head.weight' not in state or tuple(state['head.weight'].shape) != shape:
        raise ValueError('Need a fine-tuned SSV2 head.weight of shape (174,384), not pretrain-only or K400 weights')
    if 'head.bias' not in state or tuple(state['head.bias'].shape) != (174,):
        raise ValueError('Missing/mismatched 174-class head.bias')
    for key,value in state.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Nonfinite checkpoint tensor: '+key)
    try:
        model.load_state_dict(state,strict=True)
    except RuntimeError as exc:
        raise ValueError('Checkpoint is not the pinned native MVD ViT-S architecture: '+str(exc)) from exc


def initialize_for_training(model, state):
    """Transfer recognized MVD encoder weights; enumerate every discarded field."""
    encoder = {k.removeprefix('encoder.'):v for k,v in state.items() if k.startswith('encoder.')}
    backbone = {k.removeprefix('backbone.'):v for k,v in state.items() if k.startswith('backbone.')}
    values = encoder or backbone or state.copy()
    current = model.state_dict()
    dropped = []
    # Native MVD pretraining stores a frozen positional parameter, a final
    # encoder norm and two feature decoders. The classification model uses
    # fixed regenerated positions and a new mean-pooling fc_norm instead.
    allowed_pretrain_prefixes=('decoder_img.','decoder_vid.','down_img.','down_vid.')
    allowed_pretrain_names={'mask_token_img','mask_token_vid','pos_embed_img','pos_embed_vid',
                            'norm.weight','norm.bias'}
    for key in list(values):
        if key=='pos_embed' and key not in current:
            position=values[key]
            if position.shape!=model.pos_embed.shape or not torch.allclose(position.float(),model.pos_embed.float(),atol=1e-5,rtol=1e-5):
                raise ValueError('Pretrained positional embeddings do not match native fixed 16-frame MVD positions')
            dropped.append(key);del values[key];continue
        if key not in current and (key in allowed_pretrain_names or key.startswith(allowed_pretrain_prefixes)):
            dropped.append(key);del values[key];continue
        if key in {'head.weight','head.bias'} and (key not in current or values[key].shape != current[key].shape):
            dropped.append(key);del values[key]
    missing,unexpected = model.load_state_dict(values,strict=False)
    if unexpected or any(k not in {'head.weight','head.bias','fc_norm.weight','fc_norm.bias'} for k in missing):
        raise ValueError(f'Unqualified pretrained transfer: missing={missing}, unexpected={unexpected}')
    return {'missing_initialized':list(missing),'discarded_pretrain_fields':dropped,
            'encoder_prefix_removed':bool(encoder),'backbone_prefix_removed':bool(backbone)}
