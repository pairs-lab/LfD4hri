"""Independent outcome, coverage and neural-action provenance checks."""
from pathlib import Path
from types import FunctionType
import math

from .common import read, verify, TASKS, approved_final
from .evaluate import ledger
from VLA.common.provenance import sha256_file


def audit(path, require_complete=True):
    from VLA.autonomy_r83.common import metrics
    from VLA.benchmark_guardian_r89.audit import audit_chain
    path = Path(path)
    plan = read(path.parent/'PLAN.json')
    cohort,method = plan['cohort'],plan['method']
    data,rows,keys = ledger(path,cohort)
    expected = len(cohort['seeds'])*4
    complete = len(rows)==expected and data['status'].startswith('completed_actual_')
    if require_complete and not complete:
        raise ValueError('Incomplete actual evaluation cannot be certified')
    verify(method['code']); verify(method['weights_sha256'])
    if cohort['role']=='fresh_secondary_final800' and not approved_final(method,cohort):
        raise ValueError('Wrong final authorization/locked seed block')
    # Reuse the independent outcome->trace->query auditor, generalizing only
    # its ledger validator in a NEW function namespace (no historical edits).
    chain = FunctionType(audit_chain.__code__,{**audit_chain.__globals__,
        'complete_rows':lambda p:ledger(p,cohort)},audit_chain.__name__)(path)
    if chain['actual_neural_controls_verified']!=sum(r['steps'] for r in rows):
        raise ValueError('Missing neural controls')
    if complete and chain['actual_neural_controls_verified']!=data['motor_controls']:
        raise ValueError('Final has uncommitted controls')
    per = metrics(rows)
    if complete and any(v['n']!=len(cohort['seeds']) for v in per.values()):
        raise ValueError('Unequal per-action coverage')
    for task,v in per.items():
        n,k = v['n'],v['successes']
        if not n:
            continue
        p,z = k/n,1.959963984540054
        denominator = 1+z*z/n
        center = (p+z*z/(2*n))/denominator
        radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
        v['Wilson95_percent'] = [100*(center-radius),100*(center+radius)]
        v['safety_aborts'] = sum(bool(r.get('safety_violation')) for r in rows if r['task']==task)
    return {'audit_passed':True,'complete':complete,'actual_trials':len(rows),
        'per_action':per,'actual_cohorts_of800':int(complete and expected==800),
        'not_five_measured_seeds':True,'metric':'original R62 semantic first-hit; NOT strict1cm; put release/settle not required',
        'source':str(path),'source_sha256':sha256_file(path),
        'checkpoint':method['checkpoint'],'method_weights_sha256':method['weights_sha256'],
        'final_outcomes_used_to_tune_this_checkpoint':False,**chain}
