"""Frozen budgets, cohorts and provenance for the final FALCON continuation."""
from pathlib import Path
from VLA.admission_r100.common import ROOT, CPU, SIM, ENV, TASKS, read, utc, atomic_json, verify
from VLA.common.provenance import sha256_file

OUT = ROOT / 'VLA/audits/falcon_final_r105_20261026'
R100 = ROOT / 'VLA/audits/native_admission_r100_20261025'
R104 = ROOT / 'VLA/audits/direct800_r104_20261025'
PARENT = R100 / 'falcon_esm/adapt2048_attempt00/committed_2050'
PYTHON = ROOT / 'VLA/.venvs/falcon_esm_r91/bin/python'
DEV = list(range(20411001, 20411011))
CONF = list(range(20412001, 20412011))
SMOKE = [20413001]
FINAL = list(range(20410001, 20410201))
TARGET_UPDATES = 4098
MODULE = 'VLA.falcon_final_r105.workflow'


def bindings():
    return {str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


def approval(method, cohort):
    return bool(cohort.get('role') == 'fresh_secondary_final800'
        and cohort.get('seeds') == FINAL and method.get('family') == 'falcon_esm'
        and method.get('R100_frozen_before_final') is True
        and method.get('R105_new_holdout_frozen_before_final') is True
        and method.get('frozen_for_user_requested_benchmark') is True
        and method.get('actual_checkpoint_optimizer_updates') in (2050, TARGET_UPDATES))


def verify_cohorts(old_final_seeds):
    cohorts = [set(FINAL), set(DEV), set(CONF), set(SMOKE), set(old_final_seeds)]
    if [len(x) for x in cohorts[:4]] != [200, 10, 10, 1]:
        raise ValueError('Wrong reserved cohort sizes')
    for i, a in enumerate(cohorts):
        for b in cohorts[i+1:]:
            if a & b:
                raise ValueError('Evaluation cohorts overlap')
    return True


def percentage(value):
    return '—' if value is None else f'{float(value):.2f}%'
