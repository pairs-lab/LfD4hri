"""Automatically test a diagnostic snapshot, then the completed native fit.

Never changes/stops the trainer, never reads the final800 test outcomes.
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
from VLA.flower_r70.common import (ROOT,R69,OUT,PY,SIM,ENV,DEVELOPMENT,CONFIRMATION,
                                  read,freeze_test_code,verify,checkpoint)


def select_prefix(results):
    if any(v['completed_trials']!=8 for v in results.values()):
        raise ValueError('Every declared pilot8 must complete; do not select from partial trials')
    return max(results,key=lambda p:(sum(r['success'] for r in results[p]['rows']),
        -sum(bool(r['safety_violation']) for r in results[p]['rows']),-p))


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'watch_status.json').exists():raise FileExistsError('Refuse duplicate test owner')
        bound=freeze_test_code();stage='rolling_development40';child=server=server_log=None
        def status(state='running',**extra):
            atomic_json(out/'watch_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,**extra})
        def execute(py,module,args,label):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as stream:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,env=ENV,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:status();time.sleep(3)
                code=child.returncode;child=None
            if code:raise RuntimeError(f'{label} exited{code}; see actual integration log, NOT zero SR')
        def stop_server():
            nonlocal server,server_log
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
            if server_log is not None:server_log.close();server_log=None
        def serve(cp,label,rolling=False):
            nonlocal server,server_log
            verify(bound)
            # Allow only one new inference worker, retain >1GiB free memory
            # above its verified ~3GiB inference footprint.
            deadline=time.monotonic()+1800
            while True:
                import psutil
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(gpu.strip().splitlines()[0])>=4500 and psutil.virtual_memory().available>=3.5*2**30:break
                if time.monotonic()>deadline:raise TimeoutError('Safe inference resource window unavailable')
                status(waiting_resources=True);time.sleep(10)
            server_log=(out/(label+'_server.log')).open('x')
            args=[str(PY),'-u','-m','VLA.flower_r70.server','--checkpoint',str(cp),
                '--output',str(out/(label+'_runtime'))]
            if rolling:args+=['--allow-rolling']
            server=subprocess.Popen(args,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=server_log,stderr=subprocess.STDOUT)
            session=requests.Session();session.trust_env=False;deadline=time.monotonic()+180
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own native server exited during reload')
                    try:
                        r=session.get('http://127.0.0.1:5930/health',timeout=3);r.raise_for_status()
                        if r.json()['checkpoint']!=str(cp):raise ValueError('Wrong model/checkpoint on port5930')
                        return
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native startup exceeds180s')
                        status();time.sleep(2)
            finally:session.close()
        def evaluate(cohort,dest,prefix):
            execute(SIM,'VLA.flower_r70.evaluate',['--cohort',cohort,'--output',dest,'--prefix',prefix],dest.name)
        try:
            atomic_json(out/'PREDECLARATION.json',{'rolling40_is_diagnostic_only':True,
                'final_prefix_pilot':[1,5,10],'final_development_episodes':40,'final_confirmation_episodes':40,
                'checkpoint_selection_does_not_read_final800':True,'main_training_code_not_changed':True,
                'all_safety_aborts_and_timeouts_stay_in_denominator':True,'code':bound})
            cp=out/'rolling_development_checkpoint';checkpoint(cp,allow_rolling=True)
            status();serve(cp,'rolling2048',True)
            evaluate(DEVELOPMENT,out/'rolling_development40',5);stop_server()
            stage='waiting_unmodified_R69_fit_and_fresh_reload';status()
            deadline=time.monotonic()+12*3600
            while True:
                s=read(R69/'pipeline_status.json')
                if s['status']=='completed_finetune_and_reload_NOT_evaluated_SR':break
                if s['status']!='running':raise RuntimeError('R69 did not complete cleanly: '+str(s))
                if time.monotonic()>deadline:raise TimeoutError('Finite12h postfit wait expired')
                status(training_stage=s['stage']);time.sleep(15)
            final=R69/'fit4096';proof=checkpoint(final)
            reload_proof=read(R69/'FINAL_FRESH_RELOAD.json')
            if not reload_proof['passed'] or reload_proof['actual_new_optimizer_updates']!=4096:
                raise ValueError('Final own exact native fresh reload incomplete')
            stage='final_frozen_checkpoint_prefix_pilots';status();serve(final,'final4096')
            source=read(DEVELOPMENT.with_name('pilot8.json'))
            source.update(role='development_selection',original_role='development_prefix_pilot')
            pilot=out/'PILOT8.json';atomic_json(pilot,source)
            results={}
            for prefix in (1,5,10):
                dest=out/f'final_prefix{prefix}_pilot8';evaluate(pilot,dest,prefix)
                results[prefix]=read(dest/'progress.json')
            prefix=select_prefix(results)
            atomic_json(out/'PREFIX_SELECTION.json',{'prefix':prefix,'selected_only_from_declared_development_pilot8':True,
                'pilot_results':results,'no_final800_read':True})
            stage='final_development40';status();evaluate(DEVELOPMENT,out/'final_development40',prefix)
            stage='independent_confirmation40';status();evaluate(CONFIRMATION,out/'final_confirmation40',prefix)
            stop_server();stage='per_action_actual_results_and_failure_audit';status()
            execute(SIM,'VLA.flower_r70.report',['--output',out],'per_action_report')
            stage='completed_actual_development_and_confirmation_NOT800';status('completed')
        except Exception as error:
            status('integration_error_NOT_model_SR',error=str(error));raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            stop_server()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'watch_owner.log').open('x') as stream:
            c=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r70.watch','--output',str(a.output.resolve())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'watch_pid':c.pid,'status':str(a.output/'watch_status.json')}))
    else:main(a)
