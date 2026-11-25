from contextlib import contextmanager
from pathlib import Path

from VLA.postfit800_r98.common import (ROOT, CPU, PY, SIM, ENV, TASKS, read, utc,
    verify, identity, alive, durable_json, tail, retryable, Heartbeat,
    progress_signature, atomic_json)
from VLA.common.provenance import sha256_file

OUT = ROOT/'VLA/audits/vlanext_refinement_r99_20261024'
SOURCE = ROOT/'VLA/audits/vlanext_postfit800_r98_20261024'
PARENT = ROOT/'VLA/audits/dita_terminal800_r95_20261023/other_models/vlanext_s/recovery_fit9216'
RECOVERY = ROOT/'VLA/audits/recovery_quality_r90_20261022/qualified_recovery'
PORT = 6000
DEV = list(range(20301001, 20301011))
CONF = list(range(20302001, 20302011))
FINAL = list(range(20300001, 20300201))


def bindings():
    from VLA.postfit800_r98.common import bindings as existing
    result = existing()
    result.update({str(p.resolve()):sha256_file(p)
        for p in Path(__file__).parent.glob('*.py')})
    return result


def approved(method, cohort):
    return bool(cohort['role']=='fresh_secondary_final800' and cohort['seeds']==FINAL
        and method.get('family')=='vlanext_s'
        and method.get('R99_locked_before_final') is True
        and method.get('frozen_for_user_requested_benchmark') is True
        and method.get('benchmark_authorization',{}).get('user_requested_800_after_refinement') is True)


@contextmanager
def callbacks():
    from VLA.postfit800_r98.common import native_s_callbacks
    import VLA.autorefinement_r97.evaluate as evaluator
    import VLA.autorefinement_r97.audit as auditor
    with native_s_callbacks():
        saved=evaluator.approved_final,auditor.approved_final
        evaluator.approved_final=auditor.approved_final=approved
        try:yield
        finally:evaluator.approved_final,auditor.approved_final=saved


def audit_result(path, require_complete=True):
    from VLA.autorefinement_r97.audit import audit
    with callbacks():return audit(path, require_complete=require_complete)


def learning_rate(offset, horizon):
    import math
    if not 0 <= offset < horizon:raise ValueError('Offset outside finite LR phase')
    return 1e-6 + 19e-6*min(1., (offset+1)/16)*.5*(1+math.cos(math.pi*offset/horizon))
