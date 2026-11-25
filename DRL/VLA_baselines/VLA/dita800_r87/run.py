"""Dedicated fixed800; do not stop or mutate the existing R86 fine-tune owner."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.recovery_r85.common import ROOT,read,verify,utc
from VLA.dita_transfer_r84.common import PY
from VLA.continuous_r86.owner import qualified
from .audit import final_audit

OUT=ROOT/'VLA/audits/dita800_user_r87_20261021'
CP=ROOT/'VLA/audits/continuous_r86_20261021/dita/fit1024'
PORT=5990
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
ENV={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}


def bindings():return {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


class Owner:
    def __init__(self):self.server=self.child=None;self.stage='initialize';self.bound=bindings()
    def status(self,status='running',**extra):
        atomic_json(OUT/'status.json',{'status':status,'stage':self.stage,'UTC':utc(),'pid':os.getpid(),
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'evaluator_pid':self.child.pid if self.child and self.child.poll() is None else None,**extra})
    def stop(self):
        for p in (self.child,self.server):
            if p and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
    def main(self):
        from VLA.cross_vla800_r62.common import resources
        self.stage='verify_exact_frozen_checkpoint';self.status()
        meta=qualified(CP)
        if meta['updates']!=1024 or not read(CP/'FRESH_RELOAD_cuda.json')['passed']:
            raise ValueError('Only completed CUDA-qualified Dita1024 is in this requested benchmark')
        method=read(ROOT/'VLA/audits/continuous_r86_20261021/dita/method1024.json')
        verify(method['code']);verify(method['weights_sha256'])
        method.update(code={**method['code'],**self.bound},frozen_after_confirmation=False,
            frozen_for_user_requested_benchmark=True,
            benchmark_authorization={'user_requested_800_despite_failed_pilot_gate':True,
                'request':'dita ổn đấy chạy 800 di','prior_pilot_successes':6,'prior_pilot_trials':40,
                'NOT_predeclared_promotion_gate_passed':True},
            concurrent_training_does_not_change_this_checkpoint=True,
            no_final_data_used_to_select_or_update_this_frozen_method=True)
        atomic_json(OUT/'FROZEN_METHOD.json',method)
        atomic_json(OUT/'COHORT.json',{'role':'fresh_secondary_final800','seeds':list(range(20284001,20284201)),
            'user_requested_fixed_benchmark_NOT_confirmed_high_SR':True})
        self.stage='resource_admission';deadline=time.monotonic()+12*3600
        while True:
            r=resources()
            if r['GPU_free_MiB']>=6500 and r['RAM_available_MiB']>=4500 and r['disk_free_GiB']>=15:break
            if time.monotonic()>deadline:raise TimeoutError('No safe concurrent resource window; no foreign jobs stopped')
            self.status(resource_wait=r);time.sleep(15)
        atomic_json(OUT/'RESOURCE_ADMISSION.json',{'UTC':utc(),'resources':r,
            'concurrent_jobs':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader'],text=True,timeout=15),
            'synchronous_simulation_10Hz_unchanged':True,'latency_not_an_exclusive_GPU_comparison':True})
        self.stage='load_actual_Dita1024_server';self.status()
        with (OUT/'server.log').open('x') as log:
            self.server=subprocess.Popen([str(PY),'-u','-m','VLA.dita_transfer_r84.server',
                '--method',str(OUT/'FROZEN_METHOD.json'),'--port',str(PORT)],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            session=requests.Session();session.trust_env=False
            try:
                for _ in range(600):
                    if self.server.poll() is not None:raise RuntimeError('Native server failed; see server.log, not model SR0')
                    try:
                        response=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2);response.raise_for_status()
                        if response.json()['method_sha256']!=sha256_file(OUT/'FROZEN_METHOD.json'):
                            raise ValueError('Different method occupies requested port')
                        break
                    except requests.RequestException:self.status();time.sleep(2)
                else:raise TimeoutError('Actual neural server readiness')
                self.stage='actual_frozen800_no_video';self.status()
                with (OUT/'evaluate.log').open('x') as trace:
                    self.child=subprocess.Popen([str(SIM),'-u','-m','VLA.dita800_r87.evaluate',
                        '--method',str(OUT/'FROZEN_METHOD.json'),'--cohort',str(OUT/'COHORT.json'),
                        '--output',str(OUT/'final800'),'--port',str(PORT)],cwd=ROOT,env=ENV,
                        stdin=subprocess.DEVNULL,stdout=trace,stderr=subprocess.STDOUT)
                    limit=time.monotonic()+48*3600
                    while self.child.poll() is None:
                        if time.monotonic()>limit:raise TimeoutError('48h finite actual cohort budget')
                        p=OUT/'final800/progress.json'
                        progress=read(p) if p.exists() else None
                        if progress:progress={k:v for k,v in progress.items() if k!='rows'}
                        self.status(actual_progress=progress);time.sleep(15)
                    if self.child.returncode:raise RuntimeError('Actual evaluator failed; preserve partial rows, no fabricated outcomes')
                p=OUT/'final800/progress.json';audit=final_audit(p)
                if audit['actual_trials']!=800 or any(v['n']!=200 for v in audit['per_action'].values()):
                    raise ValueError('Not800 unique actual trials with200/action')
                atomic_json(OUT/'FINAL_AUDIT.json',audit)
                results=read(p)['per_action']
                report='# Dita1024 — user-requested800\n\n'
                report+='Actual800 episodes,200/action; one real cohort, not five seeds. No video.\n\n'
                report+='User overrode promotion gate after6/40 pilot; independent CONF40 was NOT passed.\n\n'
                report+='| Action | Successes / trials | SR |\n|---|---:|---:|\n'
                for task,v in results.items():report+=f"| {task} | {v['successes']}/{v['n']} | {v['SR_percent']:.2f}% |\n"
                report+='\nR62 semantic first-hit, NOT strict1cm; put release/settle not required.\n'
                report+='\nConcurrent training was independent; wall-clock latency is not an exclusive-GPU latency benchmark.\n'
                (OUT/'REPORT.md').write_text(report)
                self.stage='completed_actual800';self.status('completed',per_action=results)
            finally:session.close()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'status.json').exists():raise FileExistsError('No duplicate benchmark owner')
        owner=Owner()
        try:owner.main()
        except BaseException as e:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'error':str(e),'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('integration_error_NOT_model_SR',error=str(e));raise
        finally:owner.stop()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-u','-m','VLA.dita800_r87.run'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'progress':str(OUT/'final800/progress.json')})
    else:main()
