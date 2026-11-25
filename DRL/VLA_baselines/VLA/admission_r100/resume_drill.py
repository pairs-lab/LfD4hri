"""One controlled interruption of an owned development SMOKE, never final800."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .common import OUT,read,atomic_json,utc
from .guard import checked_stop
from VLA.continuous_r86.owner import identity,alive


def main(family,owner_pid):
    expected=read(OUT/'WORKFLOW_OWNER.json');root=OUT/family;receipt=root/'CONTROLLED_RESUME_DRILL.json'
    if expected['pid']!=owner_pid or identity(owner_pid)!=expected:raise ValueError('Wrong live workflow owner')
    if receipt.exists():raise FileExistsError('One controlled drill only; retain its evidence')
    deadline=time.monotonic()+48*3600;snapshot=None
    while time.monotonic()<deadline:
        if identity(owner_pid)!=expected or not alive(owner_pid):raise RuntimeError('Owner ended before smoke-resume drill')
        state=read(OUT/'WORKFLOW_STATUS.json');folder=root/'integration4_attempt00';progress=folder/'progress.json'
        if state.get('family')==family and state.get('stage')=='integration4' and progress.exists():
            value=read(progress);plan=read(folder/'PLAN.json')
            if plan['cohort']['role']!='development_selection' or len(plan['cohort']['seeds'])!=1:
                raise ValueError('NEVER interrupt a quantitative or non-smoke cohort')
            if 1<=value['completed_trials']<4:
                finished={(r['seed'],r['task']) for r in value['rows']};queries=[]
                for line in (folder/'NEURAL_QUERIES.jsonl').read_text().splitlines(keepends=True):
                    if not line.endswith('\n'):continue
                    q=json.loads(line)
                    if (q['seed'],q['task']) not in finished:queries.append(q)
                child=identity(state.get('child_pid')) if state.get('child_pid') else None
                if len(queries)>=8 and child and child['ppid']==owner_pid and 'VLA.admission_r100.evaluate' in child['command']:
                    snapshot={'UTC':utc(),'family':family,'purpose':'real incomplete native-prefix physical resume verification',
                        'target_identity':child,'completed_rows_before_interruption':value['rows'],
                        'incomplete_native_queries_before_interruption':queries,'R99_or_final800_signals_sent':0,
                        'not_a_new_measured_performance_cohort':True}
                    atomic_json(root/'CONTROLLED_RESUME_DRILL_PENDING.json',snapshot)
                    checked_stop(child);break
            elif value['completed_trials']==4:raise RuntimeError('Missed smoke window; no process signalled')
        if (root/'WORKFLOW_ERROR.json').exists():raise RuntimeError('Native admission requires diagnosis before drill')
        time.sleep(1)
    if snapshot is None:raise TimeoutError('No eligible owned SMOKE window within48h')
    complete=root/'integration4_COMPLETE.json'
    while time.monotonic()<deadline and not complete.exists():
        if (root/'WORKFLOW_ERROR.json').exists():raise RuntimeError('Controlled interruption exposed a native resume issue')
        if not alive(owner_pid):raise RuntimeError('Owner ended before real resume completion')
        time.sleep(2)
    if not complete.exists():raise TimeoutError('Smoke resume completion deadline')
    resumed=Path(read(complete)['progress']);result=read(resumed);audit=read(resumed.parent/'AUDIT.json')
    if resumed.parent==folder or not audit['audit_passed'] or result['completed_trials']!=4:
        raise ValueError('No actual new-attempt complete smoke/independent chain proof')
    rows=snapshot['completed_rows_before_interruption']
    if result['rows'][:len(rows)]!=rows:raise ValueError('Actual completed outcomes changed/dropped on resume')
    actual={}
    for line in (resumed.parent/'NEURAL_QUERIES.jsonl').read_text().splitlines():
        q=json.loads(line);actual[q['seed'],q['task'],q['index']]=q
    errors=[]
    for old in snapshot['incomplete_native_queries_before_interruption']:
        new=actual[old['seed'],old['task'],old['index']]
        if old['input_sha256']!=new['input_sha256']:raise ValueError('Resumed current RGB/state/depth input hash differs')
        error=float(abs(np.asarray(old['actions'])-np.asarray(new['actions'])).max());errors.append(error)
        if error>1e-6:raise ValueError('Resumed native prediction prefix differs')
    atomic_json(receipt,{'passed':True,'UTC':utc(),'family':family,'completed_rows_preserved':len(rows),
        'actual_incomplete_native_queries_reproduced':len(errors),'max_action_error':max(errors),
        'resumed_progress':str(resumed),'actual_neural_controls_independently_verified':audit['actual_neural_controls_verified'],
        'full_native_observation_and_query_prefix_hashes_equal':True,'R99_or_final800_signals_sent':0,
        'not_an_estimate_or_synthetic_fixture':True})
    print(read(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('seer','gr00t_n15','falcon_esm'),required=True)
    p.add_argument('--owner-pid',type=int,required=True);a=p.parse_args();main(a.family,a.owner_pid)
