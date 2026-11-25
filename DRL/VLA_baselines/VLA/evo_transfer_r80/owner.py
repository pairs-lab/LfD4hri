"""Finite automatic gate -> fit -> DEV selection -> CONF -> frozen800.

No background indefinite refits, changed success, model selection on final800,
or claims that unknown new-model SR already matches FLOWER/X-VLA.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from .common import ROOT,OUT,PY,SIM,ENV,PORT,read,bindings,verify


def alive(pid):
    try:os.kill(int(pid),0);return True
    except (OSError,TypeError,ValueError):return False


class Owner:
    def __init__(self):
        self.server=None;self.child=None;self.server_log=None;self.started=time.monotonic()

    def state(self,stage,**extra):
        atomic_json(OUT/'owner_status.json',{'status':'running','stage':stage,'pid':os.getpid(),
            'active_child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'active_server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'elapsed_s':time.monotonic()-self.started,**extra})

    def run(self,stage,python,module,args):
        self.state(stage)
        with (OUT/(stage+'.log')).open('x') as log:
            self.child=subprocess.Popen([str(python),'-u','-m',module,*map(str,args)],cwd=ROOT,
                env=ENV,stdout=log,stderr=subprocess.STDOUT)
            self.state(stage)
            code=self.child.wait()
        self.child=None
        if code:raise RuntimeError(f'{stage} failed with exit{code}; see preserved log')

    def fit(self,folder,updates,resume=None,alignment=128):
        path=OUT/folder
        args=['--output',path,'--updates',updates,'--alignment-until',alignment,'--accumulation',16]
        if resume:args+=['--resume',resume]
        self.run(folder,PY,'VLA.evo_transfer_r80.train',args)
        reload=OUT/(folder+'_FRESH_RELOAD.json')
        self.run(folder+'_reload',PY,'VLA.evo_transfer_r80.reload',['--checkpoint',path,'--output',reload])
        if not read(reload)['passed']:raise ValueError('Fresh reload failed')
        return path,reload

    def stop_server(self):
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:self.server.wait(timeout=30)
            except subprocess.TimeoutExpired:self.server.kill();self.server.wait(timeout=10)
        if self.server_log:self.server_log.close()
        self.server=None;self.server_log=None

    def serve(self,checkpoint,reload,name):
        self.stop_server()
        self.server_log=(OUT/(name+'_server.log')).open('x')
        self.server=subprocess.Popen([str(PY),'-u','-m','VLA.evo_transfer_r80.server',
            '--checkpoint',str(checkpoint),'--reload-proof',str(reload)],cwd=ROOT,env=ENV,
            stdout=self.server_log,stderr=subprocess.STDOUT)
        self.state(name+'_server_loading')
        session=requests.Session();session.trust_env=False
        try:
            for _ in range(180):
                if self.server.poll() is not None:raise RuntimeError('Native server exited during load')
                try:
                    response=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
                    if response.status_code==200:return response.json()
                except requests.RequestException:pass
                time.sleep(2)
            raise TimeoutError('Native server did not become ready')
        finally:session.close()

    def evaluate(self,name,cohort,prefix=10,smoke=False,freeze=None):
        folder=OUT/name
        args=['--cohort',cohort,'--output',folder,'--prefix',prefix]
        if smoke:args+=['--smoke-controls',2]
        if freeze:args+=['--frozen-method',freeze]
        self.run(name,SIM,'VLA.evo_transfer_r80.evaluate',args)
        result=read(folder/'progress.json')
        expected=4*len(read(cohort)['seeds'])
        if result['completed_trials']!=expected:raise ValueError('Incomplete neural evaluation')
        from .report import main as report
        report()
        return result


def score(result):
    rows=result['rows']
    return (sum(r['success'] for r in rows),min(v['successes'] for v in result['per_action'].values()),
        -sum(bool(r['safety_violation']) for r in rows))


def main():
    previous=read(OUT/'owner_status.json') if (OUT/'owner_status.json').exists() else {}
    if alive(previous.get('pid')):raise ValueError('Existing owner is alive')
    if previous:raise FileExistsError('Preserve this finite attempt; do not overwrite previous owner')
    owner=Owner()
    try:
        bound=bindings();verify(bound)
        declaration={'model':'official_Evo1_LIBERO_with_disclosed_privileged_PEFT',
            'code':bound,'same_frozen_R62_plant_and_success':True,'TRAIN_episodes':2185,'TRAIN_frames':119495,
            'finite_actual_optimizer_targets':[2,128,1024,4096],'accumulation':16,
            'stage1_head_alignment_until_update':128,'stage2':'rank8 VLM LoRA + native full flow head',
            'prefix_candidates_on_pilot8':[5,10],'DEV_checkpoint_rule':'total successes, worst action successes, fewer safety aborts',
            'early_DEV_ready':'at least24/40 and at least3/10 each action',
            'confirmation_not_used_for_selection_or_further_training':True,
            'minimum_confirmation_to_start800':'at least20/40 and at least2/10 each action',
            'final800_episodes':800,'final800_episodes_per_action':200,'final800_used_for_selection':False,
            'promised_success_rate':None,'no_metric_or_denominator_changes':True}
        atomic_json(OUT/'PREDECLARATION.json',declaration)
        atomic_json(OUT/'pilot8.json',{'role':'development_selection','seeds':[20268001,20268002]})
        atomic_json(OUT/'smoke4.json',{'role':'development_selection','seeds':[20268001]})
        atomic_json(OUT/'final800_cohort.json',read(ROOT/'VLA/audits/flower800_r77_20261019/COHORT800.json'))
        development=ROOT/'VLA/audits/recovery_r56_20261017/automatic_v1/development40.json'
        confirmation=development.with_name('confirmation40.json')
        for _ in range(5760):
            out=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            with Path('/proc/meminfo').open() as f:mem=dict((r.split(':')[0],int(r.split()[1])) for r in f)
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if int(out.strip())>=13500 and mem['MemAvailable']>=6*1024**2 and not compute:break
            owner.state('waiting_safe_resources_not_interrupting_other_jobs');time.sleep(30)
        else:raise TimeoutError('Finite resource wait expired')
        gate,reload=owner.fit('gate2_v1',2,alignment=0)
        if not read(gate/'GATE2.json')['passed']:raise ValueError('Optimizer gate not passed')
        # The assistant must inspect actual native pixels before long fitting.
        # This is not an automatic image-success claim based only on shapes.
        from VLA.common.provenance import sha256_file
        for _ in range(240):
            review=OUT/'GATE_IMAGE_REVIEW.json'
            if review.exists():
                viewed=read(review)
                if not viewed.get('passed') or viewed.get('image_sha256')!=sha256_file(gate/'ACTUAL_NATIVE_INPUT.png'):
                    raise ValueError('Actual image review missing or does not match native pixel artifact')
                break
            owner.state('waiting_assistant_actual_native_image_review',image=str(gate/'ACTUAL_NATIVE_INPUT.png'))
            time.sleep(15)
        else:raise TimeoutError('No actual native image review within the finite admission window')
        owner.serve(gate,reload,'gate2')
        smoke=owner.evaluate('native8',OUT/'smoke4.json',1,smoke=True)
        owner.stop_server()
        if smoke['motor_controls']!=8 or any(r['safety_violation'] for r in smoke['rows']):
            raise ValueError('Actual8 motor safety gate failed; preserve integration diagnosis, not model SR')
        source,reload=owner.fit('alignment128',128,gate)
        source,reload=owner.fit('fit1024',1024,source)
        owner.serve(source,reload,'fit1024')
        candidates=[]
        for prefix in (5,10):
            result=owner.evaluate(f'prefix{prefix}_pilot8',OUT/'pilot8.json',prefix)
            candidates.append((score(result),prefix))
        prefix=max(candidates)[1]
        atomic_json(OUT/'PREFIX_SELECTION.json',{'prefix':prefix,'pilot_only':True,'candidates':candidates})
        dev=owner.evaluate('development40_fit1024',development,prefix)
        selected=(source,reload,dev)
        owner.stop_server()
        if score(dev)[0]<24 or score(dev)[1]<3:
            source,reload=owner.fit('fit4096',4096,source)
            owner.serve(source,reload,'fit4096')
            dev2=owner.evaluate('development40_fit4096',development,prefix)
            if score(dev2)>score(dev):selected=(source,reload,dev2)
            owner.stop_server()
        source,reload,dev=selected
        atomic_json(OUT/'DEVELOPMENT_SELECTION.json',{'checkpoint':str(source),'prefix':prefix,
            'per_action':dev['per_action'],'confirmation_and_final800_not_read':True})
        owner.serve(source,reload,'selected')
        conf=owner.evaluate('confirmation40',confirmation,prefix)
        if score(conf)[0]<20 or score(conf)[1]<2:
            owner.stop_server();owner.state('completed_finite_fit_and_confirmation_NOT800',status='completed',
                per_action=conf['per_action'],reason='Behavioral readiness threshold not met; no final800 consumed')
            return
        meta=read(source/'TRAINING.json')
        freeze={'checkpoint':str(source.resolve()),'checkpoint_sha256':meta['learned_sha256'],'prefix':prefix,
            'per_action_confirmation':conf['per_action'],'code':bound,'selected_before_final800':True}
        atomic_json(OUT/'FROZEN_BEFORE_800.json',freeze)
        final=owner.evaluate('final800',OUT/'final800_cohort.json',prefix,freeze=OUT/'FROZEN_BEFORE_800.json')
        owner.stop_server();owner.state('completed_actual_final800',status='completed',per_action=final['per_action'])
    except Exception as error:
        owner.stop_server();owner.state('integration_or_training_error_NOT_zero_model_SR',status='error',
            error=str(error),traceback=traceback.format_exc());raise
    finally:
        owner.stop_server()


if __name__=='__main__':main()
