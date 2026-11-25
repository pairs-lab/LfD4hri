import json
import os
from pathlib import Path
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'VLA/third_party/internvla_a_series'
BASE = ROOT / 'VLA/checkpoints/candidates_r68/internvla_a15'
OUT = ROOT / 'VLA/audits/internvla_native_r72_20261019'
PY = ROOT / 'VLA/.venvs/intern_r72_312/bin/python'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
ENV = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
    'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled',
    'PYTHONPATH': str(ROOT) + ':' + str(SOURCE / 'src')}


def read(path): return json.loads(Path(path).read_text())


def verify(bound):
    for p, digest in bound.items():
        if sha256_file(p) != digest: raise ValueError('Bound source/artifact changed: ' + p)


def bindings():
    from VLA.flower_r70.common import freeze_test_code
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((SOURCE / 'src/lerobot/policies/internvla_a1_5').glob('*.py'))
    paths += [SOURCE / 'src/lerobot/transforms/utils.py',
        SOURCE / 'src/lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py',
        ROOT / 'VLA/flower_r71/sampler.py', ROOT / 'VLA/family_transfer_r54/data.py',
        BASE / 'config.json', BASE / 'qwen_assets/config.json',
        BASE / 'qwen_assets/preprocessor_config.json', BASE / 'fast_assets/processing_action_tokenizer.py']
    result = freeze_test_code()
    result.update({str(p): sha256_file(p) for p in paths})
    return result
