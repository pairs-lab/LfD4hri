"""Independent coverage/ledger audit for authorized800, without a false CONF claim."""
from pathlib import Path
from VLA.autonomy_r83.common import read,metrics,verify
from VLA.common.provenance import sha256_file
from .evaluate import authorized_frozen


def final_audit(path):
    path=Path(path);progress=read(path);plan=read(path.parent/'PLAN.json')
    if not authorized_frozen(plan['method'],plan['cohort']):raise ValueError('Wrong frozen benchmark authorization')
    verify(plan['method']['code']);verify(plan['method']['weights_sha256'])
    rows=progress['rows'];keys=[(int(r['seed']),r['task']) for r in rows]
    expected={(s,t) for s in range(20284001,20284201) for t in ('touch','pick','move','pick_place')}
    if (len(rows)!=800 or len(set(keys))!=800 or set(keys)!=expected
        or progress['status']!='completed_actual_final800' or progress['completed_trials']!=800
        or progress['planned_trials']!=800):raise ValueError('Incomplete or duplicate actual800')
    actual=metrics(rows)
    if actual!=progress['per_action'] or any(v['n']!=200 for v in actual.values()):
        raise ValueError('Per-action ledger mismatch')
    if sum(r['steps'] for r in rows)!=progress['motor_controls']:
        raise ValueError('Actual episode/control coverage mismatch')
    return {'audit_passed':True,'complete':True,'actual_trials':800,'per_action':actual,
        'source':str(path),'source_sha256':sha256_file(path),'actual_independent_cohorts_of800':1,
        'confirmation_gate_passed':False,'user_requested_benchmark_override':True,
        'final_outcomes_used_to_tune_this_frozen_checkpoint':False,
        'metric':'R62 semantic first-hit; NOT strict1cm; put release/settle not required'}
