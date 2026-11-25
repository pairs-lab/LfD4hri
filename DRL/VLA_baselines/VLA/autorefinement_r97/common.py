from datetime import datetime, timezone
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/dita_autorefinement_r97_20261023'
PARENT = ROOT / 'VLA/audits/continuous_r86_20261021/dita/fit10000'
R95 = ROOT / 'VLA/audits/dita_terminal800_r95_20261023'
RECOVERY = ROOT / 'VLA/audits/recovery_quality_r90_20261022/qualified_recovery'
PROTOCOL = ROOT / 'VLA/evaluation_protocol_v9_postfit_10_per_action.json'
CPU = Path('/home/uet/miniconda3/envs/robot/bin/python')
PY = ROOT / 'VLA/.venvs/dita_r84/bin/python'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
PORT = 5998
TASKS = ('touch', 'pick', 'move', 'pick_place')
DEV = list(range(20295001, 20295011))
CONF = list(range(20296001, 20296011))
FINAL = list(range(20297001, 20297201))
OLD_THRESHOLDS = {'touch_distance_m': .04, 'pick_min_lift_m': .05,
    'move_xy_radius_m': .08, 'move_vertical_below_m': .03, 'move_vertical_above_m': .20,
    'pick_place_xy_radius_m': .08, 'pick_place_vertical_below_m': .06, 'pick_place_vertical_above_m': .10}
ENV = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def verify(files):
    for path, digest in files.items():
        if sha256_file(path) != digest:
            raise ValueError('Bound artifact changed: ' + str(path))


def bindings():
    from VLA.dita_transfer_r84.common import bindings as native_bindings
    from VLA.autonomy_r83.common import code
    result = {**native_bindings(), **code()}
    for folder in ('autorefinement_r97', 'recovery_quality_r90', 'receipt_recovery_r88', 'continuous_r86'):
        result.update({str(p.resolve()): sha256_file(p) for p in (ROOT/'VLA'/folder).glob('*.py')})
    result.update({str(p): sha256_file(p) for p in (PROTOCOL, RECOVERY/'ADMISSION.json', RECOVERY/'index.json')})
    return result


def metric_guard():
    success = read(PROTOCOL)['success']
    if (success['semantic_thresholds'] != OLD_THRESHOLDS or success['primary_mode'] != 'semantic_target_region'
        or success['required_consecutive_steps'] != 1 or success['reward_xml_threshold_scale'] != 1.25):
        raise ValueError('Old metric must remain exact; do not silently relax or tighten it')
    return success


def score(progress):
    from VLA.autonomy_r83.common import metrics
    if (progress['completed_trials'] != 40 or len(progress['rows']) != 40
        or len({(r['seed'], r['task']) for r in progress['rows']}) != 40
        or progress['status'] != 'completed_actual_development'
        or progress.get('cohort_role') not in ('development_selection','development_confirmation')):
        raise ValueError('Only a complete DEV/CONF40 is a score')
    per = metrics(progress['rows'])
    if per != progress['per_action'] or any(v['n'] != 10 for v in per.values()):
        raise ValueError('Not ten actual episodes per action')
    return (sum(v['successes'] for v in per.values()), min(v['successes'] for v in per.values()),
        -sum(bool(r.get('safety_violation')) for r in progress['rows']))


def approved_final(method, cohort):
    return bool(cohort['role'] == 'fresh_secondary_final800' and cohort['seeds'] == FINAL
        and method.get('family') == 'dita' and method.get('frozen_for_user_requested_benchmark') is True
        and method.get('R97_locked_before_final') is True
        and method.get('benchmark_authorization', {}).get('user_requested_800_after_refinement') is True)
