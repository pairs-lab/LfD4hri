import json
from pathlib import Path
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/audits/replacement_r84_20261020/dita'
REPO=ROOT/'VLA/third_party/dita'
PY=ROOT/'VLA/.venvs/dita_r84/bin/python'
BASE=ROOT/'VLA/checkpoints/replacement_r84/dita/official_oxe_policy.pth'
TEXT=ROOT/'VLA/checkpoints/replacement_r84/dita_clip_text'

def read(p):return json.loads(Path(p).read_text())

def bindings():
    files=list(Path(__file__).parent.glob('*.py'))
    files += list((REPO/'scripts/vision_tokenizers').glob('*.py'))
    files += list((REPO/'scripts/film_efficientnet').glob('*.py'))
    files += [REPO/'scripts/llama_dp.py',REPO/'config/model/llama_dp.yaml',
        REPO/'config/config_diffusion_openx.yaml',ROOT/'VLA/replacement_r84/data.py',
        ROOT/'VLA/flower_rotation_r76/sampler.py']
    return {str(p):sha256_file(p) for p in files}

def verify(bound):
    for p,digest in bound.items():
        if sha256_file(p)!=digest:raise ValueError('Dita bound file changed: '+p)
