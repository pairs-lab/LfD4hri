from datetime import datetime,timezone
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/audits/native_admission_r100_20261025'
ASSETS=ROOT/'VLA/checkpoints/admission_r100'
CPU=Path('/home/uet/miniconda3/envs/robot/bin/python')
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
R99=ROOT/'VLA/audits/vlanext_refinement_r99_20261024'
ENV={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled',
    'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
TASKS=('touch','pick','move','pick_place')


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.now(timezone.utc).isoformat()


def bindings():
    return {str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


def verify(files):
    for path,digest in files.items():
        if sha256_file(path)!=digest:raise ValueError('R100 bound artifact changed: '+str(path))
