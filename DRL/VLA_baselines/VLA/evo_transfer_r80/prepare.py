"""Download one pinned action checkpoint and configuration-only InternVL assets."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/evo1_privileged_r80_20261020'
BASE = ROOT / 'VLA/checkpoints/evo1_libero_r80'
VLM = ROOT / 'VLA/checkpoints/evo1_internvl_metadata_r80'
RELEASE = '3ddc6c9cafc57d5e678d0c7c2129060066feaaf3'
VLM_RELEASE = '4415a3b810e636d11dfa86b0e9ba40bb00535aa8'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    from huggingface_hub import HfApi, hf_hub_download
    from VLA.common.artifact_io import atomic_json
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / 'VLA/audits/flower_neighbors_r78_20261020/PINNED_SOURCE_AND_CONFIG.json').read_text())
    action = next(row for row in inventory['models'] if row['repository'] == 'MINT-SJTU/Evo1_LIBERO')
    paths = {}
    for filename in ('config.json', 'norm_stats.json', 'checkpoint.json', 'mp_rank_00_model_states.pt'):
        atomic_json(OUT / 'prepare_status.json', {'stage': 'download', 'file': filename, 'revision': RELEASE})
        path = Path(hf_hub_download('MINT-SJTU/Evo1_LIBERO', filename,
                                   revision=RELEASE, local_dir=BASE))
        row = next(x for x in action['selected_files'] if x['path'] == filename)
        if row.get('bytes') != path.stat().st_size:
            raise ValueError('Official file size differs: ' + filename)
        digest = sha(path)
        if row.get('lfs_sha256') and digest != row['lfs_sha256']:
            raise ValueError('Official checkpoint SHA differs: ' + filename)
        paths[str(path)] = {'bytes': path.stat().st_size, 'sha256': digest}
    api = HfApi()
    files = api.list_repo_files('OpenGVLab/InternVL3-1B', revision=VLM_RELEASE)
    selected = [name for name in files if '/' not in name and
                (name.endswith(('.json', '.py', '.txt', '.model')) or name == 'LICENSE')]
    for name in selected:
        path = Path(hf_hub_download('OpenGVLab/InternVL3-1B', name,
                                   revision=VLM_RELEASE, local_dir=VLM))
        paths[str(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    atomic_json(OUT / 'DOWNLOAD.json', {'passed': True, 'official_checkpoint_revision': RELEASE,
        'InternVL_configuration_revision': VLM_RELEASE, 'files': paths,
        'redundant_InternVL_foundation_weights_downloaded': False,
        'actual_model_forward_optimizer_or_SR_passed': False})
    atomic_json(OUT / 'prepare_status.json', {'stage': 'completed_verified_download', 'passed': True})
    print(json.dumps({'passed': True, 'output': str(OUT)}), flush=True)


if __name__ == '__main__':
    main()
