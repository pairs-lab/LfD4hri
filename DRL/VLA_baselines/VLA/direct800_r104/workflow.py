"""Preserve R103 failure evidence; fix reporting before the first query."""
import argparse,subprocess,time
from pathlib import Path
from VLA.admission_r100.common import ROOT,CPU,ENV,read,utc,atomic_json,verify
from VLA.admission_r100.guard import same_owned_process
from VLA.direct800_r103 import workflow as previous

OUT=ROOT/'VLA/audits/direct800_r104_20261025'
MODULE='VLA.direct800_r104.workflow'
ORIGINAL_FACTORY=previous.owner_class


def percentage(value):
    return '—' if value is None else f'{float(value):.2f}%'


def owner_class():
    parent=ORIGINAL_FACTORY()
    class NullSafeOwner(parent):
        def status(self,status='running',**extra):
            # Bypass only the defective R103 Markdown formatter. Keep the
            # exact R100 atomic JSON/status behavior and physical evaluator.
            import VLA.admission_r100.workflow as base
            base.Owner.status(self,status,**extra)
            state=read(OUT/'WORKFLOW_STATUS.json');progress=state.get('actual_progress') or {}
            per=progress.get('per_action',extra.get('per_action',{}))
            lines=['# FALCON direct800 after completed fit','',f'UTC: {utc()}',
                f'Stage: {self.stage}; status: {status}.','',
                'Direct800 authorized by the user: no DEV/CONF performance gate or additional fit. '
                'Checkpoint reload, safety, source/weight/receipt integrity and actual result audit remain required.', '',
                f"Completed: {progress.get('completed_trials',0)}/800; actual queries: {progress.get('native_queries',0)}."]
            if per:
                lines+=['','|Action|Successes/n|SR|','|---|---:|---:|']
                for task,v in per.items():lines.append(f"|{task}|{v['successes']}/{v['n']}|{percentage(v.get('SR_percent'))}|")
            lines+=['','200 actual episodes/action,800 total, no video. Same frozen checkpoint and original semantic metric; '
                'NOT strict1cm; PUT release/settle not required. No five-seed results are invented.', '',
                'R103 stopped with0 episodes and0 neural queries due to a null-SR Markdown formatting error. '
                'Its evidence is preserved separately. R104 changes reporting only, not the native policy, inputs, controller or metric.']
            (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return NullSafeOwner


def configure():
    previous.OUT=OUT;previous.MODULE=MODULE;previous.owner_class=owner_class
    previous.configure()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--watchdog',action='store_true')
    p.add_argument('--owner-pid',type=int);a=p.parse_args();configure()
    if a.watchdog:previous.watchdog(a.owner_pid)
    elif a.launch:
        verify(read(OUT/'AUTOMATION_QUALIFICATION.json')['code'])
        if (OUT/'WORKFLOW_OWNER.json').exists() and same_owned_process(read(OUT/'WORKFLOW_OWNER.json')):
            raise ValueError('Duplicate direct owner')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'WORKFLOW_STATUS.json')},flush=True)
    else:previous.main()
