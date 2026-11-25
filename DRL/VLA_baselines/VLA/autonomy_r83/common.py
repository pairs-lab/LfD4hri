import json
import os
from pathlib import Path
from datetime import datetime, timezone
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/autonomous_vla_r83_20261020'
R80 = ROOT / 'VLA/audits/evo1_privileged_r80_20261020'
R82 = ROOT / 'VLA/audits/evo1_user_requested800_r82_20261020'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
EVO = ROOT / 'VLA/.venvs/evo_r80/bin/python'
ADAPTER = ROOT / 'VLA/.venvs/openvla/bin/python'
PORT = 5963
TASKS = ('touch', 'pick', 'move', 'pick_place')
ENV = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '2',
       'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'}
DEV = list(range(20283001, 20283011))
CONF = list(range(20283101, 20283111))
FINAL = list(range(20284001, 20284201))


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def code():
    from VLA.evo_transfer_r80.common import bindings
    from VLA.family_transfer_r54.common import hashes
    result = {**bindings(), **hashes()}
    for folder in ('autonomy_r83', 'privileged_visual_r53', 'new_vlas'):
        result.update({str(p.resolve()): sha256_file(p) for p in (ROOT / 'VLA' / folder).glob('*.py')})
    return result


def verify(bound):
    for text, digest in bound.items():
        if sha256_file(text) != digest:
            raise ValueError('Frozen source/weight changed: ' + text)


def metrics(rows):
    result = {}
    for task in TASKS:
        selected = [r for r in rows if r['task'] == task]
        successes = sum(bool(r['success']) for r in selected)
        result[task] = {'n': len(selected), 'successes': successes,
                        'SR_percent': 100 * successes / len(selected) if selected else None}
    return result


def score(progress):
    per = metrics(progress['rows'])
    if len(progress['rows']) != 40 or any(v['n'] != 10 for v in per.values()):
        raise ValueError('Selection requires complete40, ten per action')
    return (sum(v['successes'] for v in per.values()), min(v['successes'] for v in per.values()),
            -sum(bool(r.get('safety_violation')) for r in progress['rows']))


def ready(progress):
    total, worst, _ = score(progress)
    return total >= 20 and worst >= 2
