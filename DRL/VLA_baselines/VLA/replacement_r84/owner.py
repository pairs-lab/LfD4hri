"""Finite native qualification -> adapt -> DEV40 -> CONF40 -> frozen800.

Never calls an Oracle for model actions. Never invents a missing model adapter,
edits live source, counts integration errors as SR0, or relaunches retired jobs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import requests
from .common import ROOT,OUT,PY,TASKS,bindings,verify,read,utc
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

PORT=5984
ENV={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
     'MKL_NUM_THREADS':'2','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','WANDB_MODE':'disabled'}
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'

class Owner:
    def __init__(self):
        self.child=self.server=self.server_log=None
        self.stage='initialize'
        self.bound=bindings()

    def status(self,status='running',**extra):
        state={'status':status,'stage':self.stage,'UTC':utc(),'pid':os.getpid(),
            'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,**extra}
        atomic_json(OUT/'owner_status.json',state)
        text=f"# Live replacement study\n\nUpdated UTC: {state['UTC']}\n\nStatus: {status}\nStage: {self.stage}\n\n"
        text+='VLANeXt-S only has a complete implemented training/evaluation adapter in R84.\n'
        text+='Dita/FALCON remain qualification candidates; they are NOT silently counted as queued/running finetunes.\n'
        text+='Resource/integration errors are not model success rates. Final800 requires independent CONF40.\n'
        (OUT/'LIVE.md').write_text(text)

    def phase(self,name):
        self.stage=name
        with (OUT/'EVENTS.jsonl').open('a') as log:
            log.write(json.dumps({'UTC':utc(),'stage':name})+'\n')
        self.status()

    def resource(self):
        from VLA.cross_vla800_r62.common import resources
        deadline=time.monotonic()+12*3600
        while True:
            r=resources()
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
            if not compute and r['GPU_free_MiB']>=13000 and r['RAM_available_MiB']>=5500 and r['disk_free_GiB']>=35:
                return
            if time.monotonic()>deadline:
                raise TimeoutError('No exclusive resource window; other processes not stopped')
            self.status(resource_wait=r);time.sleep(15)

    def run(self,python,module,args,label,*,hours=12):
        verify(self.bound);self.phase(label)
        logpath=OUT/'vlanext_s'/(label+'.log')
        with logpath.open('x') as log:
            self.child=subprocess.Popen([str(python),'-u','-m',module,*map(str,args)],cwd=ROOT,
                env=ENV,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            limit=time.monotonic()+hours*3600
            while self.child.poll() is None:
                if time.monotonic()>limit:
                    raise TimeoutError('Finite phase time budget: '+label)
                self.status(log=str(logpath));time.sleep(10)
            code=self.child.returncode
            self.child=None
        if code:
            raise RuntimeError(f'{label} exit{code}; retained log={logpath}; NOT a model SR')

    def stop(self):
        for proc in (self.child,self.server):
            if proc and proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
        self.child=self.server=None
        if self.server_log:self.server_log.close();self.server_log=None

    def method(self,checkpoint,name,*,frozen=False):
        meta=read(checkpoint/'TRAINING.json')
        if not read(checkpoint/'FRESH_RELOAD.json')['passed']:
            raise ValueError('Fresh reload not qualified')
        weights=str(checkpoint/'weights.safetensors')
        p=OUT/'vlanext_s'/(name+'.json')
        bound={**self.bound}
        from VLA.autonomy_r83.common import code as evaluator_code
        bound.update(evaluator_code())
        atomic_json(p,{'family':'vlanext_s','checkpoint':str(checkpoint),'parent':'Qwen/Qwen3.5-0.8B',
            'variant':'native_full_trainable_adam8_privileged74_map3',
            'weights_sha256':{weights:meta['weights_sha256']},'code':bound,
            'prefixes':{task:1 for task in TASKS},'frozen_after_confirmation':frozen,
            'timing':'10Hz observation and policy query; native chunk8, execute1; measured history8',
            'metric':'unchanged R62 semantic firsthit, NOT strict1cm, PUT release not required',
            'new_different_method_not_poolable_with_original_XVLA800':True})
        return p

    def evaluate(self,method,label,role,seeds):
        self.resource();self.phase(label+'_server')
        cohort=OUT/'vlanext_s'/(label+'_COHORT.json')
        atomic_json(cohort,{'role':role,'seeds':seeds})
        self.server_log=(OUT/'vlanext_s'/(label+'_server.log')).open('x')
        self.server=subprocess.Popen([str(PY),'-u','-m','VLA.replacement_r84.server',
            '--method',str(method),'--port',str(PORT)],cwd=ROOT,env=ENV,
            stdin=subprocess.DEVNULL,stdout=self.server_log,stderr=subprocess.STDOUT)
        session=requests.Session();session.trust_env=False
        try:
            for _ in range(300):
                if self.server.poll() is not None:
                    raise RuntimeError('Actual native server exited; inspect its log')
                try:
                    response=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
                    response.raise_for_status()
                    if response.json()['method_sha256']!=sha256_file(method):
                        raise ValueError('Port occupied by another method, no process was killed')
                    break
                except requests.RequestException:
                    self.status();time.sleep(2)
            else:raise TimeoutError('Native server readiness timeout')
            self.run(SIM,'VLA.autonomy_r83.evaluate',['--method',method,'--cohort',cohort,
                '--output',OUT/'vlanext_s'/label,'--port',PORT],label,
                hours=36 if role=='fresh_secondary_final800' else 4)
            p=read(OUT/'vlanext_s'/label/'progress.json')
            if p['completed_trials']!=len(seeds)*4:
                raise ValueError('Incomplete cohort is not a complete success estimate')
            return p
        finally:session.close();self.stop()

    def main(self):
        gate=OUT/'vlanext_s/native_gate4_adam8'
        self.phase('wait_actual_native_optimizer4_gate_already_started')
        deadline=time.monotonic()+3600
        while not (gate/'TRAINING.json').exists():
            if (gate/'ERROR.json').exists() or time.monotonic()>deadline:
                raise RuntimeError('Native qualification did not pass; do not train blind or blame model')
            self.status();time.sleep(10)
        self.resource()
        self.run(PY,'VLA.replacement_r84.train',['reload','--checkpoint',gate],'gate_reload',hours=1)
        options=[];parent=gate
        # Declared target-domain pilot recipe, NOT a claim of native20k/global128 reproduction.
        # No removal after a random-init tiny gate: allow progressively trained snapshots.
        for step in (512,2048,8192):
            fit=OUT/'vlanext_s'/f'fit{step}'
            self.resource()
            self.run(PY,'VLA.replacement_r84.train',['fit','--output',fit,'--updates',step,
                '--accumulation',16,'--optimizer','adamw8','--resume',parent],f'fit{step}',hours=24)
            self.run(PY,'VLA.replacement_r84.train',['reload','--checkpoint',fit],f'fit{step}_reload',hours=1)
            method=self.method(fit,f'method{step}')
            dev=self.evaluate(method,f'dev40_fit{step}','development_selection',list(range(20283001,20283011)))
            total=sum(r['success'] for r in dev['rows'])
            worst=min(v['successes'] for v in dev['per_action'].values())
            options.append({'updates':step,'checkpoint':str(fit),'successes':total,'worst':worst,
                            'DEV40':str(OUT/'vlanext_s'/f'dev40_fit{step}'/'progress.json')})
            atomic_json(OUT/'vlanext_s/DEV_SELECTION.json',{'actual_options':options,
                'predeclared_promote_rule':'at least24/40 total, at least4/10 each action',
                'final_or_confirmation_used_to_train_or_tune':False})
            if total>=24 and worst>=4:
                break
            parent=fit
        chosen=max(options,key=lambda row:(row['successes'],row['worst'],-row['updates']))
        if chosen['successes']<24 or chosen['worst']<4:
            self.phase('not_promoted_after_finite_target_domain_pilot')
            self.status('current_adaptation_not_promising_NOT_model_intrinsic_limit',selection=chosen)
            return
        selected=Path(chosen['checkpoint']);method=self.method(selected,'selected_method')
        conf=self.evaluate(method,'confirmation40','development_confirmation',list(range(20283101,20283111)))
        total=sum(r['success'] for r in conf['rows']);worst=min(v['successes'] for v in conf['per_action'].values())
        if total<24 or worst<4:
            self.phase('independent_confirmation_failed_NO_final800')
            self.status('not_promoted',actual_confirmation_successes=total)
            return
        finalmethod=self.method(selected,'FROZEN_FINAL_METHOD',frozen=True)
        atomic_json(OUT/'vlanext_s/FROZEN_FINAL.json',{'selected':chosen,
            'confirmation':str(OUT/'vlanext_s/confirmation40/progress.json'),
            'method':str(finalmethod),'method_sha256':sha256_file(finalmethod),
            'code':self.bound,'no_further_tuning_on_final':True})
        final=self.evaluate(finalmethod,'final800','fresh_secondary_final800',list(range(20284001,20284201)))
        rows=final['rows']
        if len(rows)!=800 or len({(r['task'],r['seed']) for r in rows})!=800 or any(v['n']!=200 for v in final['per_action'].values()):
            raise ValueError('Final coverage/deduplication failed')
        atomic_json(OUT/'vlanext_s/FINAL_AUDIT.json',{'passed_coverage':True,'actual_trials':800,
            'actual_seeds_of_800':1,'five_seed_estimates_NOT_empirical_results':True,
            'per_action':final['per_action'],'method_sha256':sha256_file(finalmethod)})
        self.phase('frozen_native800_completed')
        self.status('completed',per_action=final['per_action'])

def main(a):
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'vlanext_s').mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'owner_status.json').exists():
            raise FileExistsError('Never duplicate/restart an existing owner without an explicit new run')
        owner=Owner()
        try:owner.main()
        except BaseException as error:
            atomic_json(OUT/'PIPELINE_ERROR.json',{'UTC':utc(),'stage':owner.stage,
                'classification':'integration_or_resource_issue_NOT_model_SR',
                'error':str(error),'traceback':traceback.format_exc()})
            owner.status('quarantined_requires_code_audit_NOT_model_zero',error=str(error))
            raise
        finally:owner.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(PY),'-u','-m','VLA.replacement_r84.owner'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'status':str(OUT/'owner_status.json')})
    else:main(a)
