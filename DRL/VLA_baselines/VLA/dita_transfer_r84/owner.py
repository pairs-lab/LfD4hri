"""Second native candidate, automatically qualified after the active GPU study."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback
import requests
from .common import ROOT,OUT,PY,read,bindings,verify
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.replacement_r84.common import TASKS,utc

ENV={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
    'MKL_NUM_THREADS':'2','WANDB_MODE':'disabled'}
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PORT=5985

class Owner:
    def __init__(self):
        self.child=self.server=self.log=None;self.stage='initialize';self.bound=bindings()
    def status(self,status='running',**extra):
        atomic_json(OUT/'owner_status.json',{'status':status,'stage':self.stage,'pid':os.getpid(),'UTC':utc(),
            'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,**extra})
    def phase(self,label):self.stage=label;self.status()
    def stop(self):
        for p in (self.child,self.server):
            if p and p.poll() is None:
                p.terminate()
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=10)
        self.child=self.server=None
        if self.log:self.log.close();self.log=None
    def resource(self):
        from VLA.cross_vla800_r62.common import resources
        deadline=time.monotonic()+12*3600
        while True:
            r=resources()
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
            if not compute and r['GPU_free_MiB']>13000 and r['RAM_available_MiB']>5500 and r['disk_free_GiB']>25:return
            if time.monotonic()>deadline:raise TimeoutError('No safe GPU admission window')
            self.status(resource_wait=r);time.sleep(15)
    def run(self,python,module,args,label,hours=12):
        verify(self.bound);self.phase(label)
        with (OUT/(label+'.log')).open('x') as log:
            self.child=subprocess.Popen([str(python),'-u','-m',module,*map(str,args)],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+hours*3600
            while self.child.poll() is None:
                if time.monotonic()>deadline:raise TimeoutError('Finite phase limit '+label)
                self.status();time.sleep(10)
            code=self.child.returncode;self.child=None
        if code:raise RuntimeError(label+f' exit{code}; see preserved log; NOT SR0')
    def method(self,cp,name,frozen=False):
        from VLA.autonomy_r83.common import code as common_code
        m=read(cp/'TRAINING.json')
        if m['device']!='cuda' or not read(cp/'FRESH_RELOAD_cuda.json')['passed']:
            raise ValueError('Actual CUDA optimization/reload required')
        p=OUT/(name+'.json')
        atomic_json(p,{'family':'dita','checkpoint':str(cp),'variant':'native_full_FT_OPEN01_rgb3_sensor74',
            'code':{**self.bound,**common_code()},'weights_sha256':{
                str(cp/'weights.safetensors'):m['weights_sha256'],**read(OUT/'TEXT_FOUNDATION.json')['files']},
            'prefixes':{t:1 for t in TASKS},'frozen_after_confirmation':frozen,
            'timing':'10Hz actual observations/queries, native10 chunk, execute1, real2-frame context',
            'no_Oracle_or_geometric_target_action_residual':True,
            'metric':'same R62 semantic firsthit; NOT strict1cm or release/settle'})
        return p
    def evaluate(self,method,label,role,seeds):
        self.resource();self.phase(label+'_server');cohort=OUT/(label+'_COHORT.json')
        atomic_json(cohort,{'role':role,'seeds':seeds})
        self.log=(OUT/(label+'_server.log')).open('x')
        self.server=subprocess.Popen([str(PY),'-u','-m','VLA.dita_transfer_r84.server',
            '--method',str(method),'--port',str(PORT)],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
            stdout=self.log,stderr=subprocess.STDOUT)
        session=requests.Session();session.trust_env=False
        try:
            for _ in range(300):
                if self.server.poll() is not None:raise RuntimeError('Native Dita server failed')
                try:
                    r=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2);r.raise_for_status()
                    if r.json()['method_sha256']!=sha256_file(method):raise ValueError('Wrong method occupying port')
                    break
                except requests.RequestException:self.status();time.sleep(2)
            else:raise TimeoutError('Native model startup timed out')
            self.run(SIM,'VLA.autonomy_r83.evaluate',['--method',method,'--cohort',cohort,
                '--output',OUT/label,'--port',PORT],label,hours=36 if role=='fresh_secondary_final800' else 4)
            p=read(OUT/label/'progress.json')
            if p['completed_trials']!=4*len(seeds):raise ValueError('Incomplete actual trials')
            return p
        finally:session.close();self.stop()
    def main(self):
        self.phase('wait_CPU_native2_and_frozen_CLIP_preparation')
        cp=OUT/'cpu_gate2_v3';limit=time.monotonic()+2*3600
        while not (cp/'TRAINING.json').exists():
            if (cp/'ERROR.json').exists() or time.monotonic()>limit:raise RuntimeError('CPU native gate failed or timed out')
            self.status();time.sleep(15)
        self.run(PY,'VLA.dita_transfer_r84.train',['reload','--checkpoint',cp,'--device','cpu'],'cpu_gate_reload',hours=1)
        self.phase('queued_behind_active_VLANeXt_GPU_study_NO_preemption')
        limit=time.monotonic()+72*3600
        while True:
            other=read(ROOT/'VLA/audits/replacement_r84_20261020/owner_status.json')
            if other['status']!='running':break
            if time.monotonic()>limit:raise TimeoutError('Prior study exceeded declared72h admission wait')
            self.status(prior_stage=other['stage']);time.sleep(20)
        self.resource()
        gpu=OUT/'gpu_gate4'
        self.run(PY,'VLA.dita_transfer_r84.train',['fit','--output',gpu,'--updates',4,
            '--accumulation',4,'--resume',cp],'gpu_gate4',hours=1)
        self.run(PY,'VLA.dita_transfer_r84.train',['reload','--checkpoint',gpu],'gpu_gate_reload',hours=1)
        parent=gpu;options=[]
        for step in (1024,4096,10000):
            cp=OUT/f'fit{step}';self.resource()
            self.run(PY,'VLA.dita_transfer_r84.train',['fit','--output',cp,'--updates',step,
                '--accumulation',32,'--resume',parent],f'fit{step}',hours=24)
            self.run(PY,'VLA.dita_transfer_r84.train',['reload','--checkpoint',cp],f'fit{step}_reload',hours=1)
            method=self.method(cp,f'method{step}')
            dev=self.evaluate(method,f'dev40_fit{step}','development_selection',list(range(20283001,20283011)))
            total=sum(r['success'] for r in dev['rows']);worst=min(v['successes'] for v in dev['per_action'].values())
            options.append({'checkpoint':str(cp),'updates':step,'successes':total,'worst':worst})
            atomic_json(OUT/'DEV_SELECTION.json',{'actual_options':options,'rule':'24/40 total +4/10 per action',
                'confirmation_and_final_NOT_used_for_selection':True})
            if total>=24 and worst>=4:break
            parent=cp
        chosen=max(options,key=lambda x:(x['successes'],x['worst'],-x['updates']))
        if chosen['successes']<24 or chosen['worst']<4:
            self.phase('finite_native_adaptation_not_promoted')
            self.status('current_adaptation_not_promising_NOT_model_intrinsic_limit');return
        cp=Path(chosen['checkpoint']);method=self.method(cp,'selected_method')
        conf=self.evaluate(method,'confirmation40','development_confirmation',list(range(20283101,20283111)))
        if sum(r['success'] for r in conf['rows'])<24 or min(v['successes'] for v in conf['per_action'].values())<4:
            self.phase('independent_confirmation_failed_NO_final800');self.status('not_promoted');return
        method=self.method(cp,'FROZEN_FINAL_METHOD',frozen=True)
        atomic_json(OUT/'FROZEN_FINAL.json',{'checkpoint':str(cp),'selection':chosen,
            'method_sha256':sha256_file(method),'no_final_tuning':True})
        final=self.evaluate(method,'final800','fresh_secondary_final800',list(range(20284001,20284201)))
        if len({(r['seed'],r['task']) for r in final['rows']})!=800 or any(v['n']!=200 for v in final['per_action'].values()):
            raise ValueError('Final actual coverage mismatch')
        atomic_json(OUT/'FINAL_AUDIT.json',{'passed':True,'actual_trials':800,'per_action':final['per_action'],
            'one_real_cohort_NOT_five_empirical_seeds':True})
        self.phase('completed_frozen_native_final800');self.status('completed',per_action=final['per_action'])

def main():
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'owner_status.json').exists():raise FileExistsError('Never duplicate an existing owner')
        owner=Owner()
        try:owner.main()
        except BaseException as e:
            atomic_json(OUT/'PIPELINE_ERROR.json',{'classification':'integration_or_resource_NOT_SR0',
                'stage':owner.stage,'error':str(e),'traceback':traceback.format_exc()})
            owner.status('quarantined_requires_integration_audit_NOT_model_zero',error=str(e));raise
        finally:owner.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(PY),'-u','-m','VLA.dita_transfer_r84.owner'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'status':str(OUT/'owner_status.json')})
    else:main()
