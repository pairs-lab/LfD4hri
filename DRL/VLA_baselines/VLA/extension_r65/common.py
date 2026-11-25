import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VLA/audits/jepa_fast800_r65_20261018'
R62 = ROOT / 'VLA/audits/cross_vla800_r62_20261018'
R56 = ROOT / 'VLA/audits/recovery_r56_20261017/automatic_v1'
DATA = ROOT / 'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/qualified_native_data'
BASE = ROOT / 'VLA/checkpoints/vla_jepa_pretrain_r65'
QWEN = ROOT / 'VLA/checkpoints/vla_jepa_qwen_metadata_r65'
NATIVE = ROOT / 'VLA/third_party/lerobot_jepa'
ORIGINAL = ROOT / 'VLA/third_party/vla_jepa'
PY = ROOT / 'VLA/.venvs/vla_jepa312/bin/python'
SIM = ROOT / 'VLA/third_party/lerobot/.venv/bin/python'
PI = ROOT / 'VLA/third_party/openpi/.venv/bin/python'
SCHEMA = 'R65_native_VLA_R52_TRAIN_GT3_EEF_no_Oracle_v1'
TASKS = ('touch', 'pick', 'move', 'pick_place')
HORIZONS = {'vla_jepa': 7, 'pi0_fast': 32}
BASE_REVISION = 'e946c3e5b538d760f4b4ff239d1b1c12090c041d'
ENV = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
       'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '.82'}
JEPA_ENV = {**ENV, 'PYTHONPATH': str(NATIVE/'src') + ':' + str(ROOT)}


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def privileged_coverage(model):
    from VLA.scene64_schema import FEATURE_NAMES
    from VLA.scene64_conditioning import NUMERIC_NAMES
    if model not in HORIZONS:
        raise ValueError('Unknown native privileged family')
    return {'reference':'X-VLA R53 causal scene64→numeric67 + front/wrist/GT3',
        'source_fields64':list(FEATURE_NAMES),'all_source_fields_available':True,
        'numeric_names67':list(NUMERIC_NAMES) if model=='vla_jepa' else None,
        'transport':'full numeric67 + original instruction' if model=='vla_jepa' else
                    'native32 state + complete scene64 text',
        'views':['unchanged_front_RGB','unchanged_wrist_RGB','same_causal_R53_GT_diagram'],
        'same_complete_TRAIN_and_recovery_sampler':True,
        'no_future_values_teacher_actions_waypoints_or_success_flags':True,
        'X_VLA_native_absolute20_decoder_NOT_copied_to_delta7_head':True}


def sources():
    from VLA.common.provenance import sha256_file
    files = list(Path(__file__).parent.glob('*.py'))
    files += list((NATIVE/'src/lerobot/policies/vla_jepa').glob('*.py'))
    files += list((ROOT/'VLA/third_party/openpi/src/openpi').rglob('*.py'))
    files += [ROOT/'VLA/third_party/openpi/scripts/train.py']
    files += [ROOT/'VLA/pi0'/name for name in ('training.py','scene64_training.py',
        'scene32_training.py','eef_scene32_training.py','fast_tokenizer.py',
        'fast_decoder_integrity.py','audit_scene32_saved.py','scene32_inputs.py')]
    files += [ROOT/'VLA/privileged_native_server.py']
    # Bind processor/config/chat-template bytes too, without rehashing the6GB
    # foundation on every process heartbeat. Foundation SHA is checked on load.
    files += [p for folder in (BASE,QWEN) for p in folder.iterdir()
              if p.is_file() and p.suffix in ('.json','.txt','.jinja')]
    for folder in ('family_transfer_r54', 'cross_vla800_r62'):
        files += list((ROOT/'VLA'/folder).glob('*.py'))
    bound = {str(p.resolve()): sha256_file(p) for p in files}
    bound.update(read(R62/'CODE_AND_PLANT_BINDING.json'))
    return bound


def verify(bound):
    from VLA.common.provenance import sha256_file
    for path, digest in bound.items():
        if sha256_file(path) != digest:
            raise ValueError('Bound source/evidence changed: ' + path)


def model_env(model):
    return JEPA_ENV if model == 'vla_jepa' else ENV


def model_python(model):
    return PY if model == 'vla_jepa' else PI


def predecessor_busy():
    from VLA.cross_vla800_r62.common import alive
    s = read(R62/'owner_status.json')
    return {k: s[k] for k in ('pid', 'active_child_pid', 'active_server_pid') if alive(s.get(k))}


def resource_ready():
    from VLA.cross_vla800_r62.common import resources, admissible
    r = resources()
    return admissible(r), r


def native_contract(cp):
    cp = Path(cp).resolve()
    c = read(cp/'r65_contract.json')
    if c['schema'] != SCHEMA or not c['actual_neural_VLA_actions_only']:
        raise ValueError('Not a qualified R65 native checkpoint')
    verify(c['code'])
    from VLA.common.provenance import sha256_file
    if sha256_file(DATA/'audit.json') != c['data_audit_sha256']:
        raise ValueError('Actual qualified TRAIN bank changed')
    if c['model']=='vla_jepa':
        verify({str(BASE/'model.safetensors'):c['base_weights_sha256'],
                str(cp/'learned.safetensors'):c['learned_sha256']})
    else:
        verify({c['precision']:c['precision_sha256']})
    return c
