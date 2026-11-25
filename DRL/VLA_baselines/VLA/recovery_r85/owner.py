"""Resume from completed512, fix actual test launch, then continue both existing recipes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import traceback
from .common import ROOT,R84,OUT,read,utc,verify,bindings
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
import VLA.replacement_r84.owner as flow
import VLA.dita_transfer_r84.owner as dita

# Separate recovery process and new output folders. No old source/metadata is
# rewritten or hash checks bypassed; physical model/metric/prefixes are unchanged.
flow.OUT=OUT
dita.OUT=OUT/'dita'
flow.PORT=5986
dita.PORT=5987

class Flow(flow.Owner):
    def __init__(self):
        super().__init__();self.bound.update(bindings())
    def status(self,status='running',**extra):
        atomic_json(OUT/'owner_status.json',{'status':status,'stage':self.stage,'UTC':utc(),'pid':os.getpid(),
            'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,**extra})
    def run(self,python,module,args,label,*,hours=12):
        if module=='VLA.autonomy_r83.evaluate':module='VLA.recovery_r85.evaluate'
        return super().run(python,module,args,label,hours=hours)
    def main(self):
        parent=R84/'vlanext_s/fit512';meta=read(parent/'TRAINING.json')
        verify(meta['code'])
        if meta['updates']!=512 or not read(parent/'FRESH_RELOAD.json')['passed']:
            raise ValueError('Do not repeat or fake the already completed512 native fit')
        options=[]
        for step in (512,2048,8192):
            if step==512:fit=parent
            else:
                fit=OUT/'vlanext_s'/f'fit{step}';self.resource()
                self.run(flow.PY,'VLA.replacement_r84.train',['fit','--output',fit,'--updates',step,
                    '--accumulation',16,'--optimizer','adamw8','--resume',parent],f'fit{step}',hours=24)
                self.run(flow.PY,'VLA.replacement_r84.train',['reload','--checkpoint',fit],f'fit{step}_reload',hours=1)
            method=self.method(fit,f'method{step}')
            dev=self.evaluate(method,f'dev40_fit{step}','development_selection',list(range(20283001,20283011)))
            total=sum(r['success'] for r in dev['rows']);worst=min(v['successes'] for v in dev['per_action'].values())
            options.append({'updates':step,'checkpoint':str(fit),'successes':total,'worst':worst})
            atomic_json(OUT/'vlanext_s/DEV_SELECTION.json',{'actual_options':options,
                'same_predeclared_rule':'24/40 total plus4/10 every action','final_used_for_tuning':False})
            if total>=24 and worst>=4:break
            parent=fit
        chosen=max(options,key=lambda r:(r['successes'],r['worst'],-r['updates']))
        if chosen['successes']<24 or chosen['worst']<4:
            self.stage='finite_adaptation_not_promoted';self.status('not_promoted');return
        cp=Path(chosen['checkpoint']);method=self.method(cp,'selected_method')
        conf=self.evaluate(method,'confirmation40','development_confirmation',list(range(20283101,20283111)))
        if sum(r['success'] for r in conf['rows'])<24 or min(v['successes'] for v in conf['per_action'].values())<4:
            self.stage='confirmation_failed_NO_final800';self.status('not_promoted');return
        method=self.method(cp,'FROZEN_FINAL_METHOD',frozen=True)
        atomic_json(OUT/'vlanext_s/FROZEN_FINAL.json',{'selected':chosen,'method':str(method),
            'method_sha256':sha256_file(method),'no_final_tuning':True})
        final=self.evaluate(method,'final800','fresh_secondary_final800',list(range(20284001,20284201)))
        if len({(r['seed'],r['task']) for r in final['rows']})!=800 or any(v['n']!=200 for v in final['per_action'].values()):
            raise ValueError('Actual800 coverage mismatch')
        atomic_json(OUT/'vlanext_s/FINAL_AUDIT.json',{'passed':True,'actual_trials':800,'per_action':final['per_action']})
        self.stage='frozen_final800_completed';self.status('completed',per_action=final['per_action'])

class Dita(dita.Owner):
    def __init__(self):
        super().__init__();self.bound.update(bindings())
    def run(self,python,module,args,label,hours=12):
        if module=='VLA.autonomy_r83.evaluate':module='VLA.recovery_r85.evaluate'
        elif module=='VLA.dita_transfer_r84.train':module='VLA.recovery_r85.dita_train'
        return super().run(python,module,args,label,hours=hours)
    def prepare_gate(self):
        self.resource();cpu=R84/'dita/cpu_gate2_v3';gpu=OUT/'dita/gpu_gate4'
        if not read(cpu/'FRESH_RELOAD_cpu.json')['passed']:raise ValueError('CPU native qualification missing')
        self.run(dita.PY,'VLA.dita_transfer_r84.train',['fit','--output',gpu,'--updates',4,
            '--accumulation',4,'--resume',cpu,'--four-task-gate'],'gpu_gate4',hours=1)
        self.run(dita.PY,'VLA.dita_transfer_r84.train',['reload','--checkpoint',gpu],'gpu_gate_reload',hours=1)
        self.stage='CUDA_gate_passed_queued_after_VLANeXt';self.status()
        return gpu
    def main_after_gate(self,parent):
        options=[]
        for step in (1024,4096,10000):
            cp=OUT/'dita'/f'fit{step}';self.resource()
            self.run(dita.PY,'VLA.dita_transfer_r84.train',['fit','--output',cp,'--updates',step,
                '--accumulation',32,'--resume',parent],f'fit{step}',hours=24)
            self.run(dita.PY,'VLA.dita_transfer_r84.train',['reload','--checkpoint',cp],f'fit{step}_reload',hours=1)
            method=self.method(cp,f'method{step}')
            dev=self.evaluate(method,f'dev40_fit{step}','development_selection',list(range(20283001,20283011)))
            total=sum(r['success'] for r in dev['rows']);worst=min(v['successes'] for v in dev['per_action'].values())
            options.append({'updates':step,'checkpoint':str(cp),'successes':total,'worst':worst})
            atomic_json(OUT/'dita/DEV_SELECTION.json',{'actual_options':options,'rule':'24/40 total +4/10/action'})
            if total>=24 and worst>=4:break
            parent=cp
        chosen=max(options,key=lambda r:(r['successes'],r['worst'],-r['updates']))
        if chosen['successes']<24 or chosen['worst']<4:
            self.stage='finite_adaptation_not_promoted';self.status('not_promoted');return
        cp=Path(chosen['checkpoint']);method=self.method(cp,'selected_method')
        conf=self.evaluate(method,'confirmation40','development_confirmation',list(range(20283101,20283111)))
        if sum(r['success'] for r in conf['rows'])<24 or min(v['successes'] for v in conf['per_action'].values())<4:
            self.stage='confirmation_failed_NO_final800';self.status('not_promoted');return
        method=self.method(cp,'FROZEN_FINAL_METHOD',frozen=True)
        atomic_json(OUT/'dita/FROZEN_FINAL.json',{'selected':chosen,'method_sha256':sha256_file(method),'no_final_tuning':True})
        final=self.evaluate(method,'final800','fresh_secondary_final800',list(range(20284001,20284201)))
        if len({(r['seed'],r['task']) for r in final['rows']})!=800 or any(v['n']!=200 for v in final['per_action'].values()):
            raise ValueError('Actual800 coverage mismatch')
        atomic_json(OUT/'dita/FINAL_AUDIT.json',{'passed':True,'actual_trials':800,'per_action':final['per_action']})
        self.stage='frozen_final800_completed';self.status('completed',per_action=final['per_action'])

def main():
    for name in ('vlanext_s','dita'):(OUT/name).mkdir(parents=True,exist_ok=True)
    # The inherited method writer needs the same already verified text receipt
    # in its new output namespace; weights/tokenizer files stay at original paths.
    atomic_json(OUT/'dita/TEXT_FOUNDATION.json',read(R84/'dita/TEXT_FOUNDATION.json'))
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'owner_status.json').exists():raise FileExistsError('Never duplicate a recovery owner')
        if not read(OUT/'REPAIR_TESTS.json')['passed']:raise ValueError('Repair unit tests missing')
        d=Dita();s=Flow();parent=None
        s.stage='qualify_corrected_Dita_GPU_gate_before_resuming_DEV40';s.status()
        # First qualify the previously blocked short CUDA gate, then resume S
        # from its completed512. Per-model failures do not silently kill the other.
        for name,owner,fn in [('dita_gate',d,d.prepare_gate),('vlanext_s',s,s.main)]:
            try:
                result=fn()
                if name=='dita_gate':parent=result
            except BaseException as e:
                atomic_json(OUT/(name+'_ERROR.json'),{'error':str(e),'stage':owner.stage,'traceback':traceback.format_exc(),'NOT_model_SR':True})
                owner.status('quarantined_integration_error',error=str(e))
            finally:owner.stop()
        if parent:
            try:d.main_after_gate(parent)
            except BaseException as e:
                atomic_json(OUT/'dita/PIPELINE_ERROR.json',{'error':str(e),'stage':d.stage,'traceback':traceback.format_exc(),'NOT_model_SR':True})
                d.status('quarantined_integration_error',error=str(e))
            finally:d.stop()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(flow.PY),'-u','-m','VLA.recovery_r85.owner'],cwd=ROOT,env=flow.ENV,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print({'owner_pid':child.pid,'S_status':str(OUT/'owner_status.json'),'Dita_status':str(OUT/'dita/owner_status.json')})
    else:main()
