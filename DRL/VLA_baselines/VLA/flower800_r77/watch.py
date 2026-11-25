"""Admit, freeze, parity-check and execute the actual FLOWER800 cohort."""
import argparse
import fcntl
from math import sqrt
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import ROOT,PY,SIM,ENV,read,verify
from VLA.flower_feedback_r75.watch import parity
from VLA.flower800_r77.evaluate import binding
from VLA.flower800_r77.queue_guard import OUT,resume
from VLA.flower_rotation_r76.queue_guard import identity
from VLA.cross_vla800_r62.common import SEEDS,TASKS,matrix


def measured_report(folder):
    p=read(folder/'progress.json');values=matrix(p['rows'])
    if p['status']!='completed_actual_final800':raise ValueError('All actual800 required before final report')
    per={}
    for i,task in enumerate(TASKS):
        k=int(values[:,i].sum());n=200;prob=k/n;z=1.959963984540054
        center=(prob+z*z/(2*n))/(1+z*z/n)
        radius=z*sqrt(prob*(1-prob)/n+z*z/(4*n*n))/(1+z*z/n)
        per[task]={'successes':k,'episodes':n,'SR_percent':100*prob,
                   'Wilson_95_CI_percent':[100*(center-radius),100*(center+radius)]}
    result={'model':'FLOWER','actual_completed_episodes':800,'per_action':per,
        'checkpoint':p['checkpoint'],'one_evaluation_cohort_NOT_five_training_seeds':True,
        'all_safety_aborts_and_timeouts_in_denominator':True,'no_video_saved':True,
        'progress_sha256':sha256_file(folder/'progress.json')}
    atomic_json(folder/'MEASURED_RESULTS.json',result)
    lines=['# FLOWER: one measured 800-episode evaluation','',
        '|Action|Successes / episodes|Success rate|95% Wilson interval|','|---|---:|---:|---:|']
    for task,v in per.items():
        lo,hi=v['Wilson_95_CI_percent']
        lines.append(f"|{task}|{v['successes']}/200|{v['SR_percent']:.1f}%|{lo:.1f}–{hi:.1f}%|")
    lines+=['',f"Total: {int(values.sum())}/800 ({100*float(values.mean()):.2f}%).",'',
        'These are actual trials, not five measured training seeds or an estimated five-seed mean ± standard deviation.',
        'The checkpoint was selected before this cohort using development outcomes only. No task, seed, failure or timeout was rerolled or removed.',
        'UR5e, Robotiq2F85, objects, camera profile, common contact-qualified grasp assist, low-level EEF IK and safety/success settings match R62.',
        'Actions are native FLOWER outputs, not Oracle/planner actions. PUT is goal-region first-hit, not release and settled placement.',
        'No videos/images were stored; full dynamics and actual native-query receipts remain.','']
    (folder/'REPORT.md').write_text('\n'.join(lines))


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise FileExistsError('Refuse duplicate800 owner')
        bound=binding();stage='frozen_checkpoint_admission';child=server=stream=None
        record=read(out/'INTERN_QUEUE_HOLD.json') if (out/'INTERN_QUEUE_HOLD.json').exists() else None
        def status(state='running',**extra):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,**extra})
        def run(py,module,args,label,limit=3600):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as log:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,
                    env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+limit
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite stage limit: '+label)
                    status();time.sleep(3)
                code=child.returncode;child=None
            if code:raise RuntimeError(label+f' integration exited{code}; preserve actual trials, not model SR0')
        try:
            status();proof=ROOT/'VLA/audits/flower_rotation_r76_20261019/FREEZE_REVIEW.json'
            selected=read(proof)
            if not selected['ready_for_larger_evaluation'] or not selected['new_rotation_finetune_selected']:
                raise ValueError('Latest explicitly qualified freeze candidate required')
            if selected['evaluation_entry_point']!='VLA.flower_r70.evaluate' or selected['prefix']!=10:
                raise ValueError('Unqualified runner/cadence')
            conf=selected['confirmation_per_action']
            if any(conf[t]['n']!=10 or conf[t]['SR_percent']<50 for t in TASKS):
                raise ValueError('Readiness gate: independent40 complete, no action below50%; do not blindly run800')
            cp=Path(selected['checkpoint']);training=read(cp/'TRAINING.json');verify(training['code'])
            if sha256_file(cp/'trainables.pt')!=selected['trainables_sha256'] or sha256_file(cp/'stats.json')!=selected['stats_sha256']:
                raise ValueError('Frozen checkpoint/normalizer changed')
            reload=read(cp.parent/'FRESH_RELOAD.json')
            if not reload['passed']:raise ValueError('Independent exact checkpoint reload not passed')
            method={**selected,'freeze_source':str(proof),'freeze_source_sha256':sha256_file(proof),
                'final800_outcomes_used_for_selection':False,'video_saved':False}
            atomic_json(out/'FROZEN_METHOD.json',method)
            bound[str((out/'FROZEN_METHOD.json').resolve())]=sha256_file(out/'FROZEN_METHOD.json')
            bound[str(proof)]=sha256_file(proof)
            cohort=out/'COHORT800.json';atomic_json(cohort,{'role':'final_fixed_evaluation800','seeds':SEEDS,
                'episodes_per_action':200,'same_locked_scene_seeds_as_R62_XVLA':True})
            bound[str(cohort.resolve())]=sha256_file(cohort)
            atomic_json(out/'PREDECLARATION.json',{'model':'native FLOWER','actual_plan_episodes':800,
                'episodes_per_action':200,'freeze_before_first_final_trial':True,
                'no_videos_or_images_saved_for800':True,'all_failures_retained':True,
                'no_hyperparameter_or_checkpoint_change_after800_starts':True,
                'implementation_parity_required_before_admission':True,'code':bound})
            stage='wait_current_Intern_evaluation_child_finishes_without_interrupt';status('waiting')
            if record is not None:
                import psutil
                original=record['prior_status']['active_child_pid'];deadline=time.monotonic()+3600
                while True:
                    try:
                        process=psutil.Process(original)
                        active=(process.ppid()==record['pid'] and process.status()!='zombie')
                    except psutil.NoSuchProcess:active=False
                    if not active:break
                    if time.monotonic()>deadline:raise TimeoutError('Intern current evaluation did not finish within1h')
                    status('waiting');time.sleep(5)
            run(SIM,'VLA.flower800_r77.tests',[],'unit_tests',120)
            import psutil,shutil
            stage='native_inference_resource_gate';status('waiting');deadline=time.monotonic()+1800
            while True:
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if (int(memory.strip().splitlines()[0])>=5000 and psutil.virtual_memory().available>=4.5*2**30
                    and shutil.disk_usage(ROOT).free>=10*2**30):break
                if time.monotonic()>deadline:raise TimeoutError('No safe native inference memory/disk window')
                status('waiting');time.sleep(10)
            stage='fresh_exact_native_FlowER_server';status();stream=(out/'native_server.log').open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r70.server','--checkpoint',str(cp),
                '--output',str(out/'native_runtime'),'--port','5980'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            session=requests.Session();session.trust_env=False;deadline=time.monotonic()+240
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own native server exited')
                    try:
                        r=session.get('http://127.0.0.1:5980/health',timeout=3);r.raise_for_status()
                        if r.json()['checkpoint']!=str(cp) or not r.json()['all_saved_FP32_masters_exact']:
                            raise ValueError('Native checkpoint reload differs')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native load exceeds240s')
                        status();time.sleep(2)
            finally:session.close()
            # These controls use an OLD development seed, never consume/relabel
            # the final800, and are excluded from all final denominators.
            old=out/'ORIGINAL_CONTROL4.json';new=out/'IMPLEMENTATION_CONTROL4.json'
            atomic_json(old,{'role':'development_selection','seeds':[20268011]})
            atomic_json(new,{'role':'implementation_parity_control4','seeds':[20268011]})
            stage='actual_four_action_original_runner_control';status()
            run(SIM,'VLA.flower_r70.evaluate',['--cohort',old,'--output',out/'original_control4',
                '--port',5980,'--prefix',10],'original_control4')
            stage='actual_no_media_runner_parity_control';status()
            run(SIM,'VLA.flower800_r77.evaluate',['--cohort',new,'--method',out/'FROZEN_METHOD.json',
                '--output',out/'new_control4','--control','--port',5980],'new_control4')
            atomic_json(out/'ACTUAL_PARITY.json',parity(read(out/'original_control4/progress.json'),read(out/'new_control4/progress.json')))
            stage='actual_final800_200_per_action_no_video';status()
            run(SIM,'VLA.flower800_r77.evaluate',['--cohort',cohort,'--method',out/'FROZEN_METHOD.json',
                '--output',out/'final800','--port',5980],'final800',12*3600)
            stage='verify_complete800_and_per_action_report';status();verify(bound);measured_report(out/'final800')
            stage='completed_measured800_NOT_five_training_seeds';status('completed')
        except Exception as error:
            status('integration_error_NOT_model_SR',error=str(error));raise
        finally:
            for process in (child,server):
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            if stream is not None:stream.close()
            if record is not None:resume(record,out/'INTERN_QUEUE_RESUMED.json','FLOWER800_finished_or_error')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as log:
            process=subprocess.Popen([str(SIM),'-u','-m','VLA.flower800_r77.watch','--output',str(a.output.resolve())],
                cwd=ROOT,env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':process.pid,'status':str(a.output/'owner_status.json')})
    else:main(a)
