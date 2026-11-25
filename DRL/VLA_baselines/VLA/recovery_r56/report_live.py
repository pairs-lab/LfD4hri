"""Report-only process: outputs are NEVER consumed by training/selection."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import os
import time

from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import ROOT,read
from .run import alive


def main(a):
    base=a.root.resolve();until=time.monotonic()+32*3600
    while True:
        jobs={name:read(base/name/'status.json') for name in ('automatic_v1','adapter_stage_restart_v1')
            if (base/name/'status.json').exists()}
        fitting={}
        smol=base/'automatic_v1/smol_fit2500/status.json'
        if smol.exists():fitting['smolvla']=read(smol)
        adapter=base/'adapter_stage_restart_v1/fit1500/status.json'
        if adapter.exists():fitting['vla_adapter']=read(adapter)
        for budget in (1500,3000):
            pi=base/f'automatic_v1/pi0_fit{budget}/status.json'
            if pi.exists():fitting[f'pi0_fit{budget}']=read(pi)
        results={}
        for name in ('smolvla','pi0','xvla','vla_adapter'):
            path=base/f'automatic_v1/{name}_final120/progress.json'
            if path.exists():results[name]=read(path)
        path=base/'adapter_stage_restart_v1/final120/progress.json'
        if path.exists():results['vla_adapter_LR_stage_selected']=read(path)
        updated=datetime.now(timezone.utc).isoformat()
        lines=['# R56 — live report (report-only)','', 'Updated UTC: '+updated,'',
            'Privileged simulator VLA; same robot/objects/physics/safety/success protocol. No new Oracle policy.',
            'Training/gates are not success-rate estimates. Old X-VLA result86/120 is preserved and belongs to a different frozen cohort.', '',
            '## Jobs','']
        for name,s in jobs.items():
            lines.append(f"- {name}: {s['status']} / {s['stage']} / owner PID {s['pid']} / child {s.get('active_child_pid')}")
            if s.get('error'):lines.append('  Error: '+s['error'])
        lines+=['','## Actual fit status','']
        for name,s in fitting.items():
            counter=s.get('added_optimizer_updates',s.get('actual_optimizer_updates'))
            lines.append(f"- {name}: {s['status']}, reported actual updates {counter}; checkpoint completion/reload must still pass.")
        lines+=['','## New frozen final rollouts','',
            '| Model | Completed / planned | Success / completed | SR (partial until complete) |',
            '|---|---:|---:|---:|']
        if not results:lines.append('| No new final outcomes yet | — | — | pending |')
        for name,r in results.items():
            n=r['completed_trials'];successes=sum(bool(v['success']) for v in r['rows'])
            if successes!=r.get('successes',successes):raise ValueError('Final report count mismatch')
            percentage='pending' if not n else f'{100*successes/n:.1f}%'
            lines.append(f"| {name} | {n}/{r['planned_trials']} | {successes}/{n} | {percentage} |")
        lines+=['','Development selection40 and confirmation40 are not independent final estimates.',
            'Place primary metric is goal-region entry; release/settling is not required by this preserved metric.',
            'Errors/incomplete gates must not be counted as zero model SR. All old artifacts remain unchanged.',
            'Evidence and known limitations: [REPAIR_REVIEW.md](REPAIR_REVIEW.md).']
        (base/'LIVE_RESULTS.md').write_text('\n'.join(lines)+'\n')
        done=len(jobs)==2 and not any(alive(s['pid']) for s in jobs.values())
        atomic_json(base/'live_report_status.json',{'status':'completed' if done else 'running',
            'pid':os.getpid(),'updated_at_utc':updated,'report_only_NOT_used_by_training_or_selection':True})
        if done or time.monotonic()>until:break
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'VLA/audits/recovery_r56_20261017')
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        import subprocess,sys
        with (a.root/'live_report.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.recovery_r56.report_live','--root',str(a.root.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'reporter_pid':child.pid},flush=True)
    else:main(a)
