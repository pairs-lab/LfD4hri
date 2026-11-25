"""One controlled target-balanced FLOWER ablation, after R71 completes."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import ROOT,R69,OUT as R70,PY,SIM,ENV,DEVELOPMENT,read,verify,freeze_test_code
from VLA.flower_r71.watch import candidate_wins


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'status.json').exists():raise FileExistsError('Duplicate controlled sampler owner')
        bound=freeze_test_code();bound.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
        stage='wait_R69_R70_R71_before_target_sampler_ablation';child=server=stream=None
        def status(state='waiting',**extra):
            atomic_json(out/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,'active_server_pid':None if server is None else server.pid,**extra})
        def run(py,module,args,label):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as f:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,env=ENV,
                    stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+3*3600
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite3h stage budget: '+label)
                    status('running');time.sleep(3)
                code=child.returncode;child=None
            if code:raise RuntimeError(label+f' integration exited{code}, not model SR0%')
        def serve(cp):
            nonlocal server,stream
            verify(bound);stream=(out/'native_server.log').open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r70.server','--checkpoint',str(cp),
                '--output',str(out/'native_runtime')],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            s=requests.Session();s.trust_env=False;deadline=time.monotonic()+180
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own target-balanced native server exited')
                    try:
                        r=s.get('http://127.0.0.1:5930/health',timeout=3);r.raise_for_status()
                        if r.json()['checkpoint']!=str(cp):raise ValueError('Own checkpoint/port differs')
                        return
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Server reload exceeded180s')
                        time.sleep(2)
            finally:s.close()
        try:
            sampler=read(out.parent/'SAMPLER_AUDIT.json')
            if not sampler['passed'] or sampler['task_target_strata']!=76:raise ValueError('Target-first sampler audit incomplete')
            atomic_json(out/'PREDECLARATION.json',{'experiment':'one128-update target-first sampler ablation',
                'parent_weights':'unchanged R69 final4096','reference_training_budget_matches_R71':True,
                'no_model_input_loss_optimizer_action_camera_plant_metric_changes':True,
                'changes_only_TRAIN_sampling_with_source_and_recipe_binding':True,
                'selection_uses_development_only':True,'confirmation_seeds':list(range(20268501,20268511)),
                'sampler_audit_sha256':sha256_file(out.parent/'SAMPLER_AUDIT.json'),'code':bound})
            status();deadline=time.monotonic()+16*3600
            r71=ROOT/'VLA/audits/flower_grasp_rehearsal_r71_20261019'
            while True:
                state=read(r71/'status.json')
                if state['status'] in ('completed','integration_error_NOT_model_SR'):break
                if time.monotonic()>deadline:raise TimeoutError('Finite16h prerequisite wait elapsed')
                verify(bound);status(R71_stage=state['stage']);time.sleep(15)
            if not read(R69/'FINAL_FRESH_RELOAD.json')['passed']:raise ValueError('Untouched final4096 fresh reload incomplete')
            baseline=read(R70/'final_development40/progress.json')
            if (r71/'DEVELOPMENT_SELECTION.json').exists() and read(r71/'DEVELOPMENT_SELECTION.json')['candidate_selected']:
                baseline=read(r71/'final_development40/progress.json')
            if sum(r['success'] for r in baseline['rows'])==40:
                stage='skipped_perfect_development_NOT_algorithm_ceiling';status('completed');return
            prefix=int(read(R70/'PREFIX_SELECTION.json')['prefix'])
            import psutil
            deadline=time.monotonic()+1800
            while True:
                gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(gpu.strip().splitlines()[0])>=10500 and psutil.virtual_memory().available>=5*2**30:break
                if time.monotonic()>deadline:raise TimeoutError('No safe single128-update resource window')
                status(waiting_resources=True);time.sleep(15)
            stage='one_controlled_native128_target_balanced_updates';status('running');cp=out/'fit128'
            run(PY,'VLA.target_balance_r74.flower_fit',['--source',R69/'fit4096','--output',cp],'fit128')
            stage='actual_fresh_process_reload';status('running');proof=out/'FRESH_RELOAD.json'
            run(PY,'VLA.flower_r69.reload_fit',['--checkpoint',cp,'--output',proof],'fresh_reload')
            if not read(proof)['passed']:raise ValueError('Sampler ablation exact native reload failed')
            stage='same_development40_same_prefix';status('running');serve(cp)
            run(SIM,'VLA.flower_r70.evaluate',['--cohort',DEVELOPMENT,'--output',out/'final_development40','--prefix',prefix],'dev40')
            candidate=read(out/'final_development40/progress.json');wins=candidate_wins(baseline,candidate)
            atomic_json(out/'DEVELOPMENT_SELECTION.json',{'candidate_selected':wins,'prefix_unchanged':prefix,
                'prior_best_checkpoint':baseline['checkpoint'],'prior_best_per_action':baseline['per_action'],
                'candidate_per_action':candidate['per_action'],'confirmation_NOT_used_for_selection':True})
            if wins:
                stage='new_independent_confirmation40';status('running');cohort=out/'CONFIRMATION40.json'
                atomic_json(cohort,{'role':'development_confirmation','seeds':list(range(20268501,20268511))})
                run(SIM,'VLA.flower_r70.evaluate',['--cohort',cohort,'--output',out/'final_confirmation40','--prefix',prefix],'conf40')
            run(SIM,'VLA.flower_r70.report',['--output',out],'measured_report')
            stage='completed_one_controlled_target_sampler_ablation_NOT800';status('completed')
        except Exception as error:
            status('integration_error_NOT_model_SR',error=str(error));raise
        finally:
            for proc in (child,server):
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
            if stream is not None:stream.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'VLA/audits/target_balance_r74_20261019/flower')
    parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as f:
            c=subprocess.Popen([str(PY),'-u','-m','VLA.target_balance_r74.watch_flower','--output',str(a.output.resolve())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':c.pid,'status':str(a.output/'status.json')})
    else:main(a)
