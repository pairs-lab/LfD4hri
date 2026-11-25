import json
from pathlib import Path
from datetime import datetime, timezone
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/replacement_r84_20261020'
REPO = ROOT / 'VLA/third_party/vlanext'
BASE = ROOT / 'VLA/checkpoints/replacement_r84/qwen3.5-0.8b'
PY = ROOT / 'VLA/.venvs/vlanext_r84/bin/python'
REVISION = '2fc06364715b967f1860aea9cf38778875588b17'
SOURCE_REVISION = 'd911e6b7b5d5615540faee8365e88da9397341b9'
TASKS = ('touch', 'pick', 'move', 'pick_place')

def read(p):
    return json.loads(Path(p).read_text())

def utc():
    return datetime.now(timezone.utc).isoformat()

def bindings():
    folders = [ROOT / 'VLA/replacement_r84', REPO / 'src/models']
    files = [p for folder in folders for p in folder.rglob('*.py') if 'Wan2.2' not in p.parts]
    files += [REPO / 'config/VLANeXt-S_config.yaml',
              ROOT / 'VLA/flower_rotation_r76/sampler.py', ROOT / 'VLA/family_transfer_r54/data.py']
    return {str(p): sha256_file(p) for p in sorted(files)}

def verify(bound):
    for p, digest in bound.items():
        if sha256_file(p) != digest:
            raise ValueError('Source/artifact changed: ' + p)
