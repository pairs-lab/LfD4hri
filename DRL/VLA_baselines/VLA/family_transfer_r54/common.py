import json
from pathlib import Path
from VLA.common.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/qualified_native_data'
PRIOR = ROOT/'VLA/audits/privileged_visual_r53_20261016/automatic_ablation_v2'
BASE = ROOT/'VLA/audits/family_transfer_r54_20261017'
SMOL = ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PI = ROOT/'VLA/third_party/openpi/.venv/bin/python'
TASKS = ('touch', 'pick', 'move', 'pick_place')
SCHEMA = 'native_Smol_pi0_R52_TRAIN_recovery_R53_map_v1_r54'


def read(path): return json.loads(Path(path).read_text())


def hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'VLA/fit_feedback_r52.py', ROOT/'VLA/new_vlas/dataset.py',
              ROOT/'VLA/privileged_visual_r53/scene_map.py',
              ROOT/'VLA/scene64_conditioning.py', ROOT/'VLA/pi0/eef_scene32_training.py',
              ROOT/'VLA/pi0/training.py', ROOT/'VLA/pi0/audit_scene32_saved.py',
              ROOT/'VLA/smolvla/precision.py', ROOT/'VLA/evaluate_native_chunks_r45.py',
              ROOT/'VLA/common/environment.py', ROOT/'VLA/common/body_environment.py',
              ROOT/'VLA/common/body_gripper.py', ROOT/'VLA/common/physics_safety.py',
              ROOT/'VLA/common/tasks.py', ROOT/'VLA/common/environment_factory.py',
              ROOT/'VLA/run_cross_vla120.py']
    return {str(p.resolve()): sha256_file(p) for p in sorted(paths)}


def verify(bound):
    for p, digest in bound.items():
        if sha256_file(p) != digest: raise ValueError('Frozen R54 dependency changed: '+p)


def checkpoint_contract(checkpoint):
    cp = Path(checkpoint).resolve()
    cfg = read(cp/'r54_contract.json')
    if cfg.get('schema') != SCHEMA or not cfg.get('native_VLA_actions_only'):
        raise ValueError('Not an own native R54 checkpoint')
    verify(cfg['code'])
    if sha256_file(Path(cfg['data'])/'audit.json') != cfg['data_audit_sha256']:
        raise ValueError('Qualified TRAIN receipt changed')
    return cfg
