import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/evo1_privileged_r80_20261020'
BASE = ROOT / 'VLA/checkpoints/evo1_libero_r80'
VLM = ROOT / 'VLA/checkpoints/evo1_internvl_metadata_r80'
SOURCE = ROOT / 'VLA/third_party/evo1/Evo_1'
PY = ROOT / 'VLA/.venvs/evo_r80/bin/python'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
PORT = 5950
ENV = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
       'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
       'WANDB_MODE': 'disabled', 'HF_HUB_OFFLINE': '1',
       'TRANSFORMERS_OFFLINE': '1', 'PYTHONPATH': str(ROOT)}


def read(path):
    return json.loads(Path(path).read_text())


def bindings():
    from VLA.flower_r70.common import freeze_test_code
    from VLA.common.provenance import sha256_file
    result = freeze_test_code()
    files = list(Path(__file__).parent.glob('*.py'))
    files += list(SOURCE.rglob('*.py'))
    files += [ROOT / ('VLA/' + p) for p in (
        'family_transfer_r54/data.py', 'flower_rotation_r76/sampler.py',
        'flower_r71/sampler.py', 'target_balance_r74/sampler.py')]
    files += [p for p in VLM.iterdir() if p.is_file()]
    files += [BASE / 'config.json', OUT / 'DOWNLOAD.json']
    result.update({str(p.resolve()): sha256_file(p) for p in files})
    return result


def verify(bound):
    from VLA.common.provenance import sha256_file
    for path, digest in bound.items():
        if sha256_file(path) != digest:
            raise ValueError('Bound source or evidence changed: ' + path)
