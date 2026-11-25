"""Finite serialized FLOWER qualification→calibration→fit, preserving old work.

No 800-cohort or SR claims; stop on a failed gate rather than forging results.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.cancel_fast_r66 import utc
from VLA.flower_r69.native import ROOT

PY=ROOT/'VLA/.venvs/flower_r69/bin/python'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
DEFAULT=ROOT/'VLA/audits/flower_privileged_r69_20261019'
ENV={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
     'TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled'}


def read(p):return json.loads(Path(p).read_text())


def calibration_decision(value, reload_value):
    if not value.get('completed') or value.get('actual_new_optimizer_updates')!=128:
        return False,'Actual128-update calibration not complete'
    if not reload_value.get('passed') or not reload_value.get('fresh_process'):
        return False,'Independent exact reload failed'
    first,last=value['loss_first32'],value['loss_last32']
    import math
    if not all(math.isfinite(x) and x>0 for x in (first,last)) or last>first*1.10:
        return False,'Native TRAIN loss is nonfinite or diverging; inspect before spending long-fit budget'
    counts=value['sampler_counts'];total=sum(counts['task_'+t] for t in ('touch','pick','move','pick_place'))
    if min(counts['task_'+t] for t in ('touch','pick','move','pick_place'))<total*.15:
        return False,'Actual sampler did not cover each task adequately'
    return True,'Numerical/input/optimizer gates passed; NOT evidence of robot success or convergence'


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'pipeline_status.json').exists():raise FileExistsError('Do not duplicate the new pipeline')
        child=server=log=None;stage='qualification_v2';cp=out/'gate2_v2'
        def status(state='running',**extra):
            atomic_json(out/'pipeline_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,'UTC':utc(),**extra})
        def execute(py,module,args,label):
            nonlocal child
            with (out/(label+'.log')).open('x') as stream:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,env=ENV,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:status();time.sleep(2)
                code=child.returncode;child=None
                if code:raise RuntimeError(f'{label} exited{code}; actual log preserved, not measured SR')
        def stop_server():
            nonlocal server,log
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
            if log is not None:log.close();log=None
        try:
            status()
            if not (cp/'GATE2.json').exists():
                execute(PY,'VLA.flower_r69.gate',['fit2','--output',cp],'gate2_v2')
            stage='fresh_process_reload_v2';status()
            fresh=out/'fresh_reload2_v2'
            execute(PY,'VLA.flower_r69.gate',['reload','--checkpoint',cp,'--output',fresh],'reload2_v2')
            stage='same_PyBullet_native8_v2';status()
            log=(out/'native8_server_v2.log').open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r69.server','--checkpoint',str(cp),
                '--reload-proof',str(fresh/'FRESH_RELOAD.json'),'--output',str(out/'smoke_server_v2')],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            session=requests.Session();session.trust_env=False
            until=time.monotonic()+180
            while True:
                if server.poll() is not None:raise RuntimeError('Own FLOWER neural server exited')
                try:
                    r=session.get('http://127.0.0.1:5927/health',timeout=2);r.raise_for_status()
                    if r.json()['checkpoint']!=str(cp):raise ValueError('Wrong native checkpoint on port5927')
                    break
                except requests.RequestException:
                    if time.monotonic()>until:raise TimeoutError('Own native server startup exceeded180s')
                    status();time.sleep(2)
            session.close()
            smoke=out/'native8_v2'
            execute(SIM,'VLA.flower_r69.smoke',['--output',smoke],'native8_v2');stop_server()
            common=['--initial',cp,'--reload-proof',fresh/'FRESH_RELOAD.json',
                    '--robot-proof',smoke/'SMOKE8_NOT_SR.json','--accumulation','16']
            stage='native_masked_loss_calibration128';status()
            calibrate=out/'calibration128'
            execute(PY,'VLA.flower_r69.fit',[*common,'--output',calibrate,'--updates',128],'calibration128')
            stage='fresh_process_reload_calibration';status()
            reload_cp=out/'CALIBRATION_FRESH_RELOAD.json'
            execute(PY,'VLA.flower_r69.reload_fit',['--checkpoint',calibrate,'--output',reload_cp],'reload_calibration128')
            ok,reason=calibration_decision(read(calibrate/'TRAINING.json'),read(reload_cp))
            atomic_json(out/'CALIBRATION_DECISION.json',{'admit_long_fit':ok,'reason':reason,'UTC':utc(),
                'robot_success_rate_established':False})
            if not ok:stage='calibration_requires_diagnosis_no_long_fit';status('gate_failed_preserve_actual_artifacts');return
            stage='native_finetune4096_resume_same_optimizer';status()
            fit=out/'fit4096'
            execute(PY,'VLA.flower_r69.fit',[*common,'--resume',calibrate,'--output',fit,'--updates',4096],'fit4096')
            stage='fresh_reload_final4096';status()
            execute(PY,'VLA.flower_r69.reload_fit',['--checkpoint',fit,'--output',out/'FINAL_FRESH_RELOAD.json'],
                    'reload_final4096')
            stage='fit_complete_NEEDS_development40_no_SR_claim';status('completed_finetune_and_reload_NOT_evaluated_SR')
        except Exception as error:
            status('integration_error_NOT_model_failure_SR',error=str(error));raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            stop_server()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'pipeline_owner.log').open('x') as stream:
            c=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r69.pipeline','--output',str(a.output.resolve())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pipeline_pid':c.pid,'status':str(a.output/'pipeline_status.json')}))
    else:main(a)
