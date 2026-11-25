"""Small pinned tokenizer/config assets, and safe mmap checkpoint inspection."""
import argparse
from collections import Counter
import os
import subprocess
import time
from urllib.request import urlopen
import json
from .common import ASSETS, OUT, REPO, REV, WEIGHT, WEIGHT_SHA, WEIGHT_SIZE, sha
from VLA.common.artifact_io import atomic_json


def small_assets():
    name = 'microsoft/kosmos-2-patch14-224'
    with urlopen('https://huggingface.co/api/models/'+name, timeout=60) as response:
        meta = json.load(response)
    revision = meta['sha']
    wanted = {'config.json', 'preprocessor_config.json', 'tokenizer_config.json',
        'special_tokens_map.json', 'tokenizer.json', 'sentencepiece.bpe.model', 'added_tokens.json'}
    present = {r['rfilename'] for r in meta['siblings']}
    target = ASSETS/'kosmos_assets'
    target.mkdir(parents=True, exist_ok=True)
    proof = {}
    for name0 in sorted(wanted & present):
        path = target/name0
        if not path.exists():
            with urlopen(f'https://huggingface.co/{name}/resolve/{revision}/{name0}', timeout=120) as response:
                payload = response.read(24*1024*1024+1)
            if len(payload) > 24*1024*1024:
                raise ValueError('Unexpected large tokenizer/config asset')
            with path.open('xb') as stream:
                stream.write(payload)
        proof[name0] = {'size': path.stat().st_size, 'sha256': sha(path)}
    atomic_json(OUT/'KOSMOS_ASSETS.json', {'repo': name, 'revision': revision, 'files': proof,
        'no_backbone_weight_or_dataset_downloaded': True})


def inspect():
    import torch
    if WEIGHT.stat().st_size != WEIGHT_SIZE or sha(WEIGHT) != WEIGHT_SHA:
        raise ValueError('Selected official FALCON WDepth checkpoint checksum mismatch')
    # Do NOT load arbitrary Python pickles or allocate16 GB of resident CPU RAM.
    value = torch.load(WEIGHT, map_location='cpu', weights_only=True, mmap=True)
    state = value['state_dict']
    esm = {k: list(v.shape) for k,v in state.items() if 'vggt_net' in k}
    if len(esm) < 500:
        raise ValueError('Main FALCON checkpoint must contain its native ESM')
    storage = {}
    for tensor in state.values():
        storage[(tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())] = tensor.untyped_storage().nbytes()
    atomic_json(OUT/'CHECKPOINT_INVENTORY.json', {'passed_official_download_and_safe_mmap': True,
        'repo': 'FALCON-VLA/FALCON-series', 'revision': REV, 'weight': str(WEIGHT),
        'sha256': WEIGHT_SHA, 'size': WEIGHT_SIZE, 'tensor_count': len(state),
        'dtypes': dict(Counter(str(v.dtype) for v in state.values())),
        'unique_serialized_storage_bytes': sum(storage.values()),
        'all_tensor_shapes': {k:list(v.shape) for k,v in state.items()},
        'native_ESM_included_skip_redundant_4_6GB_download': True,
        'NOT_native_strict_load_gradient_or_SR_yet': True})
    atomic_json(OUT/'CHECKPOINT_HPARAMS.json', value.get('hyper_parameters', {}))
    print({'native_ESM_tensors':len(esm), 'unique_stored_GB':sum(storage.values())/1e9}, flush=True)


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    actual = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
    if actual != 'a30c80aef8fc5e95badaa1b65ca40e25004bc87c':
        raise ValueError('Native source revision changed')
    small_assets()
    if a.wait:
        limit = time.monotonic()+3*3600
        partial = WEIGHT.with_suffix('.pt.partial')
        while not WEIGHT.exists():
            size = partial.stat().st_size if partial.exists() else 0
            atomic_json(OUT/'ASSET_STATUS.json', {'stage':'waiting_for_selected_checkpoint_download',
                'bytes':size,'expected_bytes':WEIGHT_SIZE,'pid':os.getpid(),
                'current_training_not_interrupted':True})
            if size == WEIGHT_SIZE:
                if sha(partial) != WEIGHT_SHA:
                    raise ValueError('Completed download SHA mismatch; keep partial for diagnosis')
                if WEIGHT.exists(): raise FileExistsError('Never overwrite a foundation')
                partial.rename(WEIGHT)
                break
            if time.monotonic()>limit: raise TimeoutError('Bounded download wait exceeded')
            time.sleep(20)
    inspect()
    atomic_json(OUT/'ASSET_STATUS.json', {'stage':'native_checkpoint_inventory_verified_NOT_trained',
        'weight':str(WEIGHT),'sha256':WEIGHT_SHA})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');main(p.parse_args())
