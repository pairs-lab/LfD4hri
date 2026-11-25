"""External, read-only audit owner; never changes an active benchmark source."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time

from VLA.common.artifact_io import atomic_json
from VLA.quantitative_r59.common import ROOT,OUT,read,TASKS

AUDIT_PY=Path('/home/uet/miniconda3/envs/robot/bin/python')


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def final_report():
    primary=read(OUT/'FINAL_AUDIT.json');results=read(OUT/'RESULTS.json')
    evidence=read(OUT/'INDEPENDENT_COMPLETED_PREFIX_AUDIT.json')
    if (not primary['passed'] or not results['complete'] or not evidence['passed_for_completed_prefix']
        or evidence['completed_trials_checked']!=800):raise ValueError('Both independent final gates required')
    lines=['# Independent R59 final interpretation', '',
        'Actual800episodes completed,200per action. Primary results are in REPORT.md; '
        'this read-only audit did not modify weights, action chunks, task seeds, physics, camera, safety or success.', '',
        '## Primary versus supplementary grasp diagnostic', '',
        '|Task|Primary success/200|Success with any verified prior grasp|Success without prior grasp|Safety aborts (retained as failures)|',
        '|---|---:|---:|---:|---:|']
    for t in TASKS:
        c=evidence['per_action'][t]
        lines.append(f"|{t}|{c.get('successes',0)}/200|{c.get('success_with_any_prior_verified_attachment',0)}|{c.get('success_without_any_verified_attachment',0)}|{c.get('safety_aborts',0)}|")
    lines += ['',
        'TOUCH does not require grasp. PICK primary already requires correct bilateral grasp and5cm lift. '
        'MOVE/PUT primary checks the object target region and vertical band; it does not require prior grasp, release or settling. '
        'The grasp diagnostic was added after inspecting the partial results: it is **post-hoc supplementary evidence**, not a replacement for the predeclared primary score.', '',
        'Safety aborts are retained in every primary denominator. They are not silently omitted and do not establish that the model alone caused the failure; '
        'controller/linkage/trajectory interactions remain competing causes.', '',
        '## Four additional seeds', '',
        'FUTURE_COHORT_PREDICTION_NOT_MEASURED_SEEDS.json contains explicitly modelled predictions for4further environment-seed cohorts with the same frozen checkpoint. '
        'It is not measured mean±std across5training seeds, and it does not establish variation from retraining. '
        'Do not invent4per-seed success rates. Use the measured800rates plus Wilson/bootstrap uncertainty in the main table, '
        'or run4additional complete cohorts if actual5seed mean±std is required.', '',
        '## Qualitative', '',
        'The22main +8backup clips and44freshly inspected contact sheets remain independently documented at '
        '../../qualitative/ur5e_xvla_put_r59_independent_review_20261017/index.html. '
        'Folder RL means controlled-fault privileged Oracle, not a learned RL policy. '
        'X-VLA qualitative success includes disclosed terminal opening; these clips are not extra quantitative episodes.', '']
    (OUT/'INDEPENDENT_FINAL_REVIEW.md').write_text('\n'.join(lines))
    atomic_json(OUT/'INDEPENDENT_FINAL_CERTIFICATE.json', {'passed':True,
        'primary_and_secondary_reports_have_separate_meaning':True,
        'actual_episodes':800,'actual_additional_four_cohorts_executed':False,
        'model_only_failure_attribution_NOT_claimed':True,
        'UTC':datetime.now(timezone.utc).isoformat()})


def main():
    state=OUT/'independent_watch_status.json'
    if state.exists() and alive(read(state)['pid']):raise ValueError('Independent audit watch already active')
    audit=OUT/'INDEPENDENT_COMPLETED_PREFIX_AUDIT.json'
    last=read(audit)['completed_trials_checked'] if audit.exists() else 0
    def status(value,**kw):
        atomic_json(state,{'status':value,'pid':os.getpid(),'last_audited_trials':last,
            'read_only_no_benchmark_or_training_changes':True,'UTC':datetime.now(timezone.utc).isoformat(),**kw})
    try:
        while True:
            owner=read(OUT/'owner_status.json')
            p=OUT/'episodes/progress.json';n=read(p)['completed_trials'] if p.exists() else 0
            if n-last>=100 or (n==800 and last!=800):
                with (OUT/f'independent_audit_at{n}.log').open('x') as log:
                    code=subprocess.call([str(AUDIT_PY),'-u','-m','VLA.qualitative_r59.live_quant_audit'],
                        cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'},
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if code:raise RuntimeError('Independent trace/frozen-contract audit failed; inspect retained log')
                last=read(audit)['completed_trials_checked']
            if owner['status']=='completed':
                if n!=800 or last!=800:raise ValueError('Owner completion without full800 audit')
                final_report();status('completed_independent_final_review');return
            if owner['status']!='running' or not alive(owner['pid']):
                status('owner_integration_error_NOT_model_failure',benchmark_owner=owner);return
            status('monitoring_native800',completed_trials=n)
            time.sleep(20)
    except Exception as error:
        status('independent_audit_error_requires_attention',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
