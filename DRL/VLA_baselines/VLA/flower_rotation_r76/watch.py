"""Finish one native rotation/full-support rehearsal, then review freeze.

No frozen earlier source, normalizer, plant or checkpoint is edited. Only the
still-idle Intern queue owner is temporarily paused, with an independent finite
resume watchdog. Existing fine-tuning, evaluations and CPU data jobs continue.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.flower_r70.common import ROOT,R69,PY,SIM,ENV,DEVELOPMENT,read,verify,freeze_test_code
from VLA.flower_r71.watch import candidate_wins
from VLA.flower_feedback_r75.freeze import completed,BASE,BALANCED
from VLA.flower_rotation_r76.queue_guard import identity,resume

OUT=ROOT/'VLA/audits/flower_rotation_r76_20261019'
FEEDBACK=ROOT/'VLA/audits/flower_feedback_r75_20261019'


def options():
    result=[('FLOWER4096_static10',BASE,'VLA.flower_r70.evaluate')]
    for name,folder,entry in [('FLOWER4096_balanced128',BALANCED,'VLA.flower_r70.evaluate'),
        ('FLOWER4096_feedback_r75',FEEDBACK,'VLA.flower_feedback_r75.evaluate')]:
        path=folder/'DEVELOPMENT_SELECTION.json'
        if path.exists() and read(path)['candidate_selected']:result.append((name,folder,entry))
    return result


def select(items):
    def score(item):
        d=completed(item[1]/'final_development40/progress.json')
        if d is None:raise ValueError('Incomplete candidate cannot be selected')
        return sum(r['success'] for r in d['rows']),-sum(bool(r.get('safety_violation')) for r in d['rows'])
    return max(items,key=score)


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise FileExistsError('Duplicate rotation owner')
        bound=freeze_test_code();bound.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
        stage='TRAIN_sampling_qualification';child=server=stream=None;paused=None;gpu_finished=False
        def status(state='running',**extra):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,'GPU_work_finished':gpu_finished,**extra})
        def run(py,module,args,label,limit=3600):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as log:
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,
                    env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+limit
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite stage budget: '+label)
                    status();time.sleep(3)
                code=child.returncode;child=None
            if code:raise RuntimeError(label+f' integration exited{code}; not model SR0')
        def stop_server():
            nonlocal server,stream
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
            if stream is not None:stream.close();stream=None
        try:
            atomic_json(out/'PREDECLARATION.json',{'experiment':'one128-update full-support/rotation-start sampler repair',
                'same_parent_FLOWER4096_and_native_R71_trainer_as_R74':True,
                'fixed_budget_additional_updates':128,'no_model_input_label_camera_action_loss_optimizer_or_plant_change':True,
                'selection':'development40 successes up, Pick not worse, safety aborts not more',
                'new_confirmation_seeds':list(range(20268701,20268711)),
                'no_final800_or_confirmation_score_selection':True,'code':bound})
            status();run(SIM,'VLA.flower_rotation_r76.tests',[],'unit_tests',120)
            run(SIM,'VLA.flower_rotation_r76.qualify',['--output',out/'SAMPLER_AUDIT.json'],'sampler_qualification',300)
            if not read(out/'SAMPLER_AUDIT.json')['passed']:raise ValueError('Sampler not qualified')
            # Parent is still running: this identifies an idle queue, NOT a live
            # model training worker. If it has already advanced, do not stop it.
            queued=ROOT/'VLA/audits/internvla_native_r72_20261019/balanced_v2/owner_status.json'
            q=read(queued);parent=read(BALANCED/'status.json')
            if (q['status']=='waiting' and q['stage']=='wait_FLOWER_R70_R71_R74_and_safe_resources'
                and not q['active_child_pid'] and not q['active_server_pid'] and parent['status']=='running'):
                pid=int(q['pid']);live=identity(pid)
                command=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
                if live is None or 'VLA.intern_r72.watch' not in command or 'balanced_v2' not in command:
                    raise ValueError('Queued process identity differs; do not stop it')
                paused={'pid':pid,'start_ticks':live['start_ticks'],'prior_status':q,
                    'reason':'Finish this FLOWER freeze audit before starting the unstarted native Intern GPU gate'}
                atomic_json(out/'QUEUED_INTERN_PAUSE.json',paused)
                with (out/'queue_resume_guard.log').open('x') as log:
                    subprocess.Popen([str(SIM),'-u','-m','VLA.flower_rotation_r76.queue_guard',
                        '--record',str(out/'QUEUED_INTERN_PAUSE.json'),'--owner',str(os.getpid()),
                        '--status',str(out/'owner_status.json'),'--output',str(out/'QUEUED_INTERN_WATCHDOG_RESUME.json')],
                        cwd=ROOT,env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                os.kill(pid,signal.SIGSTOP)
            import psutil
            stage='wait_safe_training_window';status('waiting');deadline=time.monotonic()+3600
            while True:
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(memory.strip().splitlines()[0])>=10500 and psutil.virtual_memory().available>=5*2**30:break
                if time.monotonic()>deadline:raise TimeoutError('No safe 10.5GB GPU/5GiB host native training window')
                verify(bound);status('waiting');time.sleep(10)
            stage='native128_rotation_full_support_finetune';status()
            cp=out/'fit128';run(PY,'VLA.flower_rotation_r76.fit',['--source',R69/'fit4096','--output',cp],'fit128')
            stage='fresh_process_exact_native_reload';status()
            proof=out/'FRESH_RELOAD.json';run(PY,'VLA.flower_r69.reload_fit',['--checkpoint',cp,'--output',proof],'fresh_reload')
            if not read(proof)['passed']:raise ValueError('Actual native reload did not pass')
            stage='own_native_server';status();stream=(out/'native_server.log').open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.flower_r70.server','--checkpoint',str(cp),
                '--output',str(out/'native_runtime'),'--port','5960'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            session=requests.Session();session.trust_env=False;deadline=time.monotonic()+240
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own native server exited')
                    try:
                        r=session.get('http://127.0.0.1:5960/health',timeout=3);r.raise_for_status()
                        if r.json()['checkpoint']!=str(cp):raise ValueError('Wrong checkpoint on5960')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native server load exceeded240s')
                        status();time.sleep(2)
            finally:session.close()
            stage='rotation_repair_development40';status()
            run(SIM,'VLA.flower_r70.evaluate',['--cohort',DEVELOPMENT,'--output',out/'final_development40',
                '--port',5960,'--prefix',10],'dev40')
            stage='wait_prior_development_comparisons';status('waiting');deadline=time.monotonic()+3600
            while True:
                states=[read(BALANCED/'status.json'),read(FEEDBACK/'owner_status.json')]
                if all(s['status'] in ('completed','integration_error_NOT_model_SR') for s in states):break
                if time.monotonic()>deadline:raise TimeoutError('Prior FLOWER comparisons did not finish in1h')
                verify(bound);status('waiting');time.sleep(10)
            prior=select(options());baseline=completed(prior[1]/'final_development40/progress.json')
            candidate=completed(out/'final_development40/progress.json');wins=candidate_wins(baseline,candidate)
            atomic_json(out/'DEVELOPMENT_SELECTION.json',{'candidate_selected':wins,'reference':prior[0],
                'prior_best_per_action':baseline['per_action'],'candidate_per_action':candidate['per_action'],
                'confirmation_not_used':True})
            if wins:
                stage='new_rotation_repair_confirmation40';status();cohort=out/'CONFIRMATION40.json'
                atomic_json(cohort,{'role':'development_confirmation','seeds':list(range(20268701,20268711))})
                run(SIM,'VLA.flower_r70.evaluate',['--cohort',cohort,'--output',out/'final_confirmation40',
                    '--port',5960,'--prefix',10],'conf40')
            run(SIM,'VLA.flower_r70.report',['--output',out],'measured_report',120)
            stop_server();gpu_finished=True
            if paused is not None:resume(paused,out/'QUEUED_INTERN_RESUMED.json')
            stage='candidate_freeze_review';status();chosen=('FLOWER4096_rotation_full_support128',out,'VLA.flower_r70.evaluate') if wins else prior
            dev=completed(chosen[1]/'final_development40/progress.json');conf=completed(chosen[1]/'final_confirmation40/progress.json')
            chosen_cp=Path(dev['checkpoint']);training=read(chosen_cp/'TRAINING.json');verify(training['code']);verify(bound)
            if sha256_file(chosen_cp/'trainables.pt')!=training['trainables_sha256']:raise ValueError('Candidate weights changed')
            result={'candidate':chosen[0],'checkpoint':str(chosen_cp),'evaluation_entry_point':chosen[2],
                'prefix':10,'trainables_sha256':training['trainables_sha256'],'stats_sha256':sha256_file(chosen_cp/'stats.json'),
                'development_per_action':dev['per_action'],'confirmation_per_action':None if conf is None else conf['per_action'],
                'ready_for_larger_evaluation':conf is not None,'native_model_not_Oracle':True,
                'new_rotation_finetune_selected':wins,'selection_scope':'development only',
                'performance_ceiling_reached':False,'not_final800_or_five_measured_training_seeds':True,
                'PUT_requires_release_settle':False,'all_safety_aborts_counted':True,'code':bound}
            atomic_json(out/'FREEZE_REVIEW.json',result)
            lines=['# FLOWER — measured candidate for larger evaluation','',f"Candidate: **{chosen[0]}**",f"Checkpoint: `{chosen_cp}`",'',
                '|Action|Development|Independent confirmation|','|---|---:|---:|']
            for task,d in dev['per_action'].items():
                c=None if conf is None else conf['per_action'][task]
                value='not completed' if c is None else f"{c['successes']}/{c['n']} ({c['SR_percent']:.1f}%)"
                lines.append(f"|{task}|{d['successes']}/{d['n']} ({d['SR_percent']:.1f}%)|{value}|")
            lines+=['','The extra rotation/full-support rehearsal was selected: '+str(wins)+'.',
                'Selection used development scores only. Confirmation, final800 and five-seed projections were not used.',
                'This is not a claim of statistical superiority, a model ceiling, zero safety failures, or completed paper-grade evaluation.',
                'PUT retains goal-region first-hit, not release/settle. No geometry-generated model action was used.','']
            (out/'FREEZE_REVIEW.md').write_text('\n'.join(lines))
            stage='completed_rotation_repair_and_freeze_candidate_NOT800';status('completed',candidate=chosen[0])
        except Exception as error:
            status('integration_error_NOT_model_SR',error=str(error));raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
            stop_server()
            if paused is not None:resume(paused,out/'QUEUED_INTERN_FINAL_RESUME.json')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as log:
            c=subprocess.Popen([str(SIM),'-u','-m','VLA.flower_rotation_r76.watch','--output',str(a.output.resolve())],
                cwd=ROOT,env={**ENV,'PYTHONPATH':str(ROOT)},stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':c.pid,'status':str(a.output/'owner_status.json')})
    else:main(a)
