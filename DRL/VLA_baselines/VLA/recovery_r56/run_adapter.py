"""One evidence-qualified native Adapter LR-stage continuation, then test.

Wait for the existing R56 owner; no concurrent GPU admission race. Model
selection reads TRAIN/development only, never old/new final outcomes.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests

from VLA.common.artifact_io import atomic_json
from VLA.family_transfer_r54.common import ROOT,DATA,SIM,read,verify
from VLA.common.provenance import sha256_file
from .run import ADAPTER, alive


def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    child=server=log=None;stage='wait_R56_owner_NO_preemption'
    frozen={str(p.resolve()):sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('adapter_fit.py'))}
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    def status(state='running',**kw):
        atomic_json(a.output/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'active_child_pid':None if child is None else child.pid,
            'active_server_pid':None if server is None else server.pid,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),**kw})
    def execute(py,module,opts,label,gpu=False):
        nonlocal child
        verify(frozen)
        if gpu:
            from VLA.new_vlas.train import wait_gpu
            admission=a.output/'GPU_admission';admission.mkdir(exist_ok=True);wait_gpu(admission)
        with (a.output/(label+'.log')).open('x') as stream:
            child=subprocess.Popen([str(py),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT);status()
            while child.poll() is None:time.sleep(10);status()
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label}: exit {code}; integration evidence preserved, NOT zero SR')
    def serve(method,label):
        nonlocal server,log
        from VLA.new_vlas.train import wait_gpu
        admission=a.output/'GPU_admission';admission.mkdir(exist_ok=True);wait_gpu(admission)
        module='VLA.privileged_visual_r53.server' if method['mode']=='map3' else 'VLA.stable_native_r44.server'
        log=(a.output/(label+'_server.log')).open('x')
        server=subprocess.Popen([str(ROOT/'VLA/.venvs/openvla/bin/python'),'-u','-m',module,
            '--checkpoint',method['checkpoint'],'--port','5899'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT)
        until=time.monotonic()+1800
        while True:
            status()
            if server.poll() is not None:raise RuntimeError('Own actual Adapter server exited')
            try:
                response=requests.get('http://127.0.0.1:5899/health',timeout=5);response.raise_for_status()
                if response.json()['checkpoint']!=method['checkpoint']:raise ValueError('Wrong actual Adapter checkpoint')
                return
            except requests.RequestException:
                if time.monotonic()>until:raise TimeoutError('Own Adapter server health timeout')
                time.sleep(5)
    def stop():
        nonlocal server,log
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        server=None
        if log is not None:log.close();log=None
    def evaluate(method,cohort,folder,label):
        execute(SIM,'VLA.autocycle_r55.evaluate',['--model','vla_adapter','--mode',method['mode'],
            '--port',5899,'--schedule',method['schedule'],'--cohort',cohort,'--output',folder],label)
        execute(SIM,'VLA.autocycle_r55.audit_video',['--model','vla_adapter','--progress',folder/'progress.json',
            '--output',folder/'video_audit'],label+'_decode_video')
    try:
        atomic_json(a.output/'plan.json',{'code':frozen,'wait_owner':str(a.wait_owner.resolve()),
            'parent':str(ADAPTER),'training_hypothesis_NOT_proven_cause':'old cumulative cosine LR was near floor; TRAIN-only head probe improved XY error34.5percent at fresh1e-4',
            'one_actual_gate2_and_fresh_reload_then1500_total_new_native_updates':True,
            'no_architecture_loss_camera_observation_action_controller_change':True,
            'selection':'matched development40 + confirmation40; preserve baseline on tie',
            'final_cohort':'same predeclared R56 20269601..30; never read predecessor final outcomes for selection',
            'guaranteed_improvement':False})
        until=time.monotonic()+30*3600;status()
        while True:
            prior=read(a.wait_owner/'status.json')
            if not alive(prior['pid']) and not any(alive(prior.get(k)) for k in ('active_child_pid','active_server_pid')):
                break
            if time.monotonic()>until:raise TimeoutError('Finite30h predecessor wait exceeded')
            time.sleep(20);status()
        proof=read(a.wait_owner/'adapter_TRAIN_probe/audit.json')
        if (not proof['passed'] or proof['checkpoint']!=str(ADAPTER)
            or proof['XY_train_error_reduction_fraction']<.25
            or any(proof['native_full_model_gradient'][k]['nonzero_elements']==0 for k in ('policy','head','projector'))):
            raise ValueError('Actual native TRAIN learnability evidence did not qualify this stage')
        origin=read(ADAPTER/'config.json')['actual_optimizer_updates']
        common=['--data',DATA,'--total-updates',origin+1500,'--input-proof',
            ROOT/'VLA/audits/privileged_visual_r53_20261016/adapter_input_v3/audit.json']
        py=ROOT/'VLA/.venvs/openvla/bin/python'
        stage='Adapter_actual_native_stage_relative_LR_gate2';status()
        gate=a.output/'gate2'
        execute(py,'VLA.recovery_r56.adapter_fit',[*common,'--mode','gate','--resume',ADAPTER,
            '--updates',origin+2,'--output',gate],'gate2',True)
        cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);reload=a.output/'reload2'
        execute(py,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--output',reload],'reload2',True)
        stage='Adapter_actual1500_total_added_updates_NATIVE_L1';status()
        fit=a.output/'fit1500'
        execute(py,'VLA.recovery_r56.adapter_fit',[*common,'--mode','long','--resume',cp,
            '--updates',origin+1500,'--reload-proof',reload/'audit.json','--output',fit],'fit1500',True)
        cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
        execute(py,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--completed-fit',
            '--output',a.output/'reload1500'],'reload1500',True)
        method={'checkpoint':str(cp.resolve()),'mode':'map3','preserved_baseline':False}
        stage='Adapter_actual_native_development_and_confirmation';status();serve(method,'new')
        pilots=[]
        for prefix in (4,6,8):
            folder=a.output/f'prefix{prefix}_pilot8'
            execute(SIM,'VLA.autocycle_r55.evaluate',['--model','vla_adapter','--port',5899,'--mode','map3',
                '--prefix',prefix,'--output',folder],f'prefix{prefix}')
            pilots.append(folder/'progress.json')
        from VLA.privileged_visual_r53.run import global_schedule
        schedule=a.output/'global_schedule.json';atomic_json(schedule,global_schedule(pilots,'vla_adapter'))
        method['schedule']=str(schedule)
        for cohort,field in (('development40','development_progress'),('confirmation40','confirmation_progress')):
            folder=a.output/cohort
            evaluate(method,a.wait_owner/(cohort+'.json'),folder,cohort)
            method[field]=str(folder/'progress.json')
        stop()
        from VLA.autocycle_r55.decision import candidate_key
        baseline=read(ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/vla_adapter_frozen_method.json')
        selected=max((method,baseline),key=candidate_key)
        atomic_json(a.output/'FROZEN_BEFORE_FINAL.json',{'selected':selected,'candidates':[method,baseline],
            'prior_or_current_final_outcomes_consumed':False})
        stage='Adapter_locked_predeclared_R56_final120';status()
        serve(selected,'locked_final')
        evaluate(selected,a.wait_owner/'final120.json',a.output/'final120','final120')
        stop();stage='completed_real_native_fit_and_frozen_test';status('completed',selected=selected)
    except Exception as error:
        status('failed_preserve_all_evidence_NOT_model_zero_SR',error=f'{type(error).__name__}: {error}')
        raise
    finally:stop()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-owner',type=Path,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.log').open('x') as stream:
            owner=subprocess.Popen([sys.executable,'-u','-m','VLA.recovery_r56.run_adapter','--output',str(a.output.resolve()),
                '--wait-owner',str(a.wait_owner.resolve())],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':owner.pid,'output':str(a.output.resolve())}))
    else:main(a)
