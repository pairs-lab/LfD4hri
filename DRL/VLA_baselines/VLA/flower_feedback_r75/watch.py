"""One bounded, native-action-only feedback ablation and candidate review."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.flower_r70.common import ROOT,R69,PY,SIM,ENV,DEVELOPMENT,read,verify
from VLA.flower_r71.watch import candidate_wins
from VLA.flower_feedback_r75.evaluate import CONFIG,binding
from VLA.flower_feedback_r75.freeze import BASE,BALANCED,review


OUT=ROOT/'VLA/audits/flower_feedback_r75_20261019'


def parity(reference,candidate):
    import numpy as np
    actual={r['task']:r for r in candidate['rows']}
    ref={r['task']:r for r in reference['rows'] if r['seed']==20268011}
    if set(actual)!=set(ref) or len(actual)!=4:raise ValueError('Four-action parity cohort incomplete')
    maxima={'actions':0.,'pre_action_scene64':0.}
    for task,r in actual.items():
        b=ref[task]
        if (r['steps'],r['success'])!=(b['steps'],b['success']):
            raise ValueError('Static control altered episode steps/outcome: '+task)
        bt=read(b['trajectory'])[task];ct=read(r['trajectory'])[task]
        for field,key in [('action','actions'),('pre_action_scene64','pre_action_scene64')]:
            x=np.asarray([v[field] for v in bt]);y=np.asarray([v[field] for v in ct])
            difference=float(np.max(np.abs(x-y)));maxima[key]=max(maxima[key],difference)
            if not np.allclose(x,y,rtol=0,atol=1e-6):
                raise ValueError('Static control failed unchanged model/plant parity: '+task+'/'+field)
    return {'passed':True,'actual_robot_episodes':4,'all_four_actions':True,
            'same_steps_and_outcomes':True,'max_abs_differences':maxima,'tolerance':1e-6}


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise FileExistsError('Duplicate feedback test owner')
        bound=binding();child=server=stream=None;stage='qualification_and_resource_gate'
        def status(state='running',**extra):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,**extra})
        def run(py,module,args,label,limit=3600):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as log:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,
                    env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+limit
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite stage budget: '+label)
                    status();time.sleep(3)
                code=child.returncode;child=None
            if code:raise RuntimeError(label+f' exited {code}: integration error, not model SR0')
        def stop_server():
            nonlocal server,stream
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
            if stream is not None:stream.close();stream=None
        def evaluate(cohort,folder,static=False):
            args=['--cohort',cohort,'--output',folder,'--port',5950]
            if static:args+=['--static-control']
            run(SIM,'VLA.flower_feedback_r75.evaluate',args,folder.name)
        try:
            atomic_json(out/'PREDECLARATION.json',{'variant':'native FLOWER4096 feedback-cadence only',
                'configuration':CONFIG,'baseline_development':str(BASE/'final_development40/progress.json'),
                'static_control_episodes':4,'development_episodes':40,
                'conditional_independent_confirmation_episodes':40,'confirmation_seeds':list(range(20268601,20268611)),
                'selection_requires_successes_increase_pick_not_worse_safety_not_worse':True,
                'no_new_training_for_this_scheduler':True,'independent_R74_finetune_preserved':True,
                'no_other_jobs_stopped_or_sources_changed':True,'code':bound})
            status();run(SIM,'VLA.flower_feedback_r75.tests',[],'unit_tests',120)
            run(SIM,'VLA.flower_feedback_r75.audit',['--output',out/'SAFETY_TRACE_AUDIT.json'],'trace_audit',120)
            import psutil
            deadline=time.monotonic()+3600
            while True:
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(memory.strip().splitlines()[0])>=5000 and psutil.virtual_memory().available>=4.5*2**30:break
                if time.monotonic()>deadline:raise TimeoutError('No verified 5GB GPU/4.5GiB host inference window')
                status('waiting',resource_gate=True);time.sleep(10)
            stage='fresh_exact_native4096_server';status()
            stream=(out/'native_server.log').open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r70.server',
                '--checkpoint',str(R69/'fit4096'),'--output',str(out/'native_runtime'),'--port','5950'],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            s=requests.Session();s.trust_env=False;deadline=time.monotonic()+240
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own native FLOWER server failed')
                    try:
                        response=s.get('http://127.0.0.1:5950/health',timeout=3);response.raise_for_status();meta=response.json()
                        if meta['checkpoint']!=str(R69/'fit4096') or not meta['all_saved_FP32_masters_exact']:
                            raise ValueError('Checkpoint/native exact reload differs')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Own native reload exceeded 240s')
                        status();time.sleep(2)
            finally:s.close()
            cohort=out/'STATIC_PARITY4.json'
            atomic_json(cohort,{'role':'development_selection','seeds':[20268011]})
            stage='actual_static_control_four_action_parity';status()
            evaluate(cohort,out/'static_parity4',True)
            atomic_json(out/'STATIC_PARITY.json',parity(read(BASE/'final_development40/progress.json'),read(out/'static_parity4/progress.json')))
            stage='feedback_development40';status();evaluate(DEVELOPMENT,out/'final_development40')
            baseline=read(BASE/'final_development40/progress.json');candidate=read(out/'final_development40/progress.json')
            wins=candidate_wins(baseline,candidate)
            atomic_json(out/'DEVELOPMENT_SELECTION.json',{'candidate_selected':wins,
                'baseline_per_action':baseline['per_action'],'candidate_per_action':candidate['per_action'],
                'confirmation_not_used_for_selection':True,'no_checkpoint_or_metric_change':True})
            run(SIM,'VLA.flower_r70.report',['--output',out],'development_report',120)
            if wins:
                stage='feedback_new_independent_confirmation40';status()
                cohort=out/'CONFIRMATION40.json'
                atomic_json(cohort,{'role':'development_confirmation','seeds':list(range(20268601,20268611))})
                evaluate(cohort,out/'final_confirmation40')
                run(SIM,'VLA.flower_r70.report',['--output',out],'confirmation_report',120)
            stop_server()
            stage='wait_independent_R74_finetune_comparison_for_freeze_review';status('waiting')
            deadline=time.monotonic()+4*3600
            while True:
                state=read(BALANCED/'status.json')
                if state['status'] in ('completed','integration_error_NOT_model_SR'):break
                if time.monotonic()>deadline:raise TimeoutError('Finite 4h R74 review wait elapsed')
                verify(bound);status('waiting',R74_stage=state['stage']);time.sleep(15)
            stage='development_only_selection_and_reproducible_freeze_review';status()
            result=review(out)
            stage='completed_feedback_ablation_and_candidate_review_NOT800'
            status('completed',candidate=result['candidate'],ready_for_larger_evaluation=result['ready_for_larger_quantitative_evaluation'])
        except Exception as error:
            status('integration_error_NOT_model_SR',error=str(error));raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            stop_server()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as log:
            c=subprocess.Popen([str(SIM),'-u','-m','VLA.flower_feedback_r75.watch','--output',str(a.output.resolve())],
                cwd=ROOT,env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':c.pid,'status':str(a.output/'owner_status.json')})
    else:main(a)
