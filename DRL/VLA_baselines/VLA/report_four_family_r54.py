"""Read-only outcome aggregation; never supplies final outcomes to training.

Optional finite watcher writes only reporting artifacts after native workers
have chosen their own frozen methods. It never launches/stops a model job.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import ROOT,PRIOR,BASE,TASKS,read


def complete_result(path):
    p=read(path)
    if p['completed_trials']!=120 or p['planned_trials']!=120:return None
    expected={(seed,task) for seed in range(20269801,20269831) for task in TASKS}
    if len(p['rows'])!=120 or {(r['seed'],r['task']) for r in p['rows']}!=expected:
        raise ValueError('Refuse a different/duplicated/missing final cohort: '+str(path))
    counts={t:sum(bool(r['success']) for r in p['rows'] if r['task']==t) for t in TASKS}
    if sum(counts.values())!=p['successes']:raise ValueError('Inconsistent actual successes')
    return {'checkpoint':p['checkpoint'],'successes':p['successes'],'trials':120,
        'per_task_successes':counts,'safety_aborts':sum(bool(r.get('safety_violation')) for r in p['rows']),
        'progress':str(path.resolve()),'scope':'same_frozen_final120_NOT_qualitative_selected_clips'}


def report(run):
    mapping={'xvla':PRIOR/'xvla/final120/progress.json','vla_adapter':PRIOR/'vla_adapter/final120/progress.json',
             'smolvla':run/'smolvla_final120/progress.json','pi0':run/'pi0_final120/progress.json'}
    results={};lines=['# Four native VLAs + simulator GT — common final cohort','',
        'Exactly the same 30 seeds/action, 120 episodes/model. No pooled development pilots or selected videos.',
        'Primary place metric is the existing goal-region criterion; it does not require release/settle.',
        'No new Oracle policy, geometric action repair, success-threshold or physics changes. The existing common low-level motor/IK/contact-assist interface is retained.','',
        '| Model | Touch /30 | Pick /30 | Move /30 | Place region /30 | Total /120 |',
        '|---|---:|---:|---:|---:|---:|']
    for model,p in mapping.items():
        result=complete_result(p) if p.exists() else None
        if result is None:
            partial=read(p)['completed_trials'] if p.exists() else 0
            lines.append(f'| {model} | pending | pending | pending | pending | pending ({partial}/120 complete) |')
            results[model]={'status':'pending_not_zero_SR','completed_trials':partial,'progress':str(p)}
        else:
            results[model]=result;c=result['per_task_successes']
            lines.append('| '+model+' | '+' | '.join(str(c[t]) for t in TASKS)+f" | {result['successes']}/120 ({100*result['successes']/120:.1f}%) |")
    ready=all(r.get('trials')==120 for r in results.values())
    lines+=['',f'All four complete: {ready}. Pending/integration errors are never counted as zero success.',
        'Input mode, selected checkpoint, execution prefix and any two-view fallback must be disclosed with each model.',
        'RDT is a fifth candidate with an older privileged checkpoint, not yet an admitted R52/R54 recovery-fit model.']
    (BASE/'FOUR_FAMILY_RESULTS.md').write_text('\n'.join(lines)+'\n')
    atomic_json(BASE/'four_family_results.json',{'updated_at_utc':datetime.now(timezone.utc).isoformat(),
        'all_four_complete':ready,'results':results,'report_only_never_training_input':True})
    return ready


def main(a):
    until=time.monotonic()+36*3600
    while True:
        ready=report(a.run)
        if ready or not a.watch:return
        if time.monotonic()>until:
            atomic_json(BASE/'report_watcher_status.json',{'status':'finite36h_timeout_preserve_partial_NOT_success','pid':os.getpid()});return
        atomic_json(BASE/'report_watcher_status.json',{'status':'waiting_actual_final_cohorts_report_only','pid':os.getpid(),
            'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        with (BASE/'report_watcher.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.report_four_family_r54','--run',str(a.run.resolve()),'--watch'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'report_watcher_pid':child.pid}))
    else:main(a)
