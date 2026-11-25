"""Finite automatic native qualification/fit/development/confirmation owner."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.flower_r70.watch import select_prefix
from VLA.flower_r71.watch import candidate_wins, needs_rehearsal
from VLA.intern_r72.common import ROOT,OUT,PY,SIM,ENV,read,verify
from VLA.intern_r72.runtime_binding import bindings


def main(a):
    out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise FileExistsError('Duplicate native Intern owner')
        bound=bindings();stage='wait_FLOWER_R70_R71_R74_and_safe_resources';child=server=stream=None
        def status(state='waiting',**extra):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,'active_server_pid':None if server is None else server.pid,**extra})
        def run(py,module,args,label,allow_failure=False):
            nonlocal child
            verify(bound)
            with (out/(label+'.log')).open('x') as log:
                env={**ENV,'PYTHONPATH':str(ROOT)} if py==SIM else ENV
                child=subprocess.Popen([str(py),'-u','-m',module,*map(str,args)],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+24*3600
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite24h per-stage limit: '+label)
                    status('running');time.sleep(3)
                code=child.returncode;child=None
            if code and not allow_failure:raise RuntimeError(label+f' integration/gate exited{code}; not model SR0')
            return code
        def stop_server():
            nonlocal server,stream
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=10)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
            server=None
            if stream is not None:stream.close();stream=None
        def serve(cp,label):
            nonlocal server,stream
            verify(bound);stream=(out/(label+'_server.log')).open('x')
            server=subprocess.Popen([str(PY),'-u','-m','VLA.intern_r72.server','--checkpoint',str(cp),
                '--output',str(out/(label+'_runtime'))],cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT)
            s=requests.Session();s.trust_env=False;deadline=time.monotonic()+300
            try:
                while True:
                    if server.poll() is not None:raise RuntimeError('Own native server exited during load')
                    try:
                        r=s.get('http://127.0.0.1:5940/health',timeout=3);r.raise_for_status()
                        if r.json()['checkpoint']!=str(cp):raise ValueError('Different model checkpoint on own port5940')
                        return
                    except requests.RequestException:
                        if time.monotonic()>deadline:raise TimeoutError('Native server startup exceeds300s')
                        status('running');time.sleep(2)
            finally:s.close()
        def evaluate(cp,cohort,folder,prefix,smoke=False):
            args=['--cohort',cohort,'--output',folder,'--prefix',prefix]
            if smoke:args+=['--smoke-controls',2]
            run(SIM,'VLA.intern_r72.evaluate',args,folder.name)
        def reload(cp,label):
            dest=out/(label+'.json')
            run(PY,'VLA.intern_r72.reload',['--checkpoint',cp,'--output',dest],label)
            if not read(dest)['passed']:raise ValueError('Exact independent native reload failed')
        try:
            budget=read(OUT/'TOKEN_BUDGET_AUDIT.json')
            if not budget['passed'] or budget['episodes_covered']!=2185 or budget['actual_processor_reference_cases_verified']!=8:
                raise ValueError('Complete declared all-episode/actual-reference token qualification required')
            atomic_json(out/'PREDECLARATION.json',{'new_model':'InternVLA-A1.5 native',
                'order':'wait Flower owners; strict gate2; fresh reload; actual8 controls; calibrate128; fresh reload; fit1024; prefix pilots; development40; optional fit2048; fresh reload; independent confirmation40',
                'full_expert_first_explicit_LoRA_expert_fallback_only_on_actual_CUDA_OOM':True,
                'all_four_actions_same_R62_plant_success_safety':True,'no_final800_selection':True,
                'native_model_not_small_random_surrogate':True,'no_WAN_aux_or_full_paper_reproduction_claim':True,
                'native_shape':[50,32],'active_action_dimensions':7,'full_numeric67_continuous_and_native_bins':True,
                'extra_sensor_text_is_measured_facts_not_Oracle_phase_or_action_advice':True,
                'target_first_sampler_avoids_R52_five_object_overweighting':True,
                'FAST_labels_only_existing_time_steps_and_attention_blocked_from_flow':True,
                'confirmation_seeds':list(range(20268301,20268311)),'code':bound})
            status();deadline=time.monotonic()+16*3600
            parent=ROOT/'VLA/audits/target_balance_r74_20261019/flower/status.json'
            while True:
                state=read(parent)
                if state['status'] in ('completed','integration_error_NOT_model_SR'):break
                if time.monotonic()>deadline:raise TimeoutError('Finite16h FLOWER resource wait elapsed')
                verify(bound);status(FLOWER_stage=state['stage']);time.sleep(15)
            atomic_json(out/'UPSTREAM_FLOWER_FINISHED_STATUS.json',state)
            import psutil
            deadline=time.monotonic()+1800
            while True:
                memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
                if int(memory.strip().splitlines()[0])>=14000 and psutil.virtual_memory().available>=5.5*2**30:break
                if time.monotonic()>deadline:raise TimeoutError('No safe14GB GPU/5.5GiB host window')
                status(waiting_resources=True);time.sleep(15)
            stage='actual_native_2update_gradient_and_memory_gate';status('running')
            adaptation='expert_full';gate=out/'gate2_expert_full'
            code=run(PY,'VLA.intern_r72.train',['--mode','gate','--updates',2,'--accumulation',4,
                '--adaptation',adaptation,'--output',gate],'gate2_full',True)
            if code:
                error=(out/'gate2_full.log').read_text()
                if 'OutOfMemoryError' not in error and 'CUDA out of memory' not in error:
                    raise RuntimeError('Full native gate has a non-OOM integration error; inspect rather than changing model')
                adaptation='expert_lora';gate=out/'gate2_expert_lora'
                atomic_json(out/'MEMORY_FALLBACK.json',{'actual_full_expert_gate_OOM':True,
                    'native_pretrained_expert_preserved':True,'fallback':adaptation,'gate_must_repeat':True})
                run(PY,'VLA.intern_r72.train',['--mode','gate','--updates',2,'--accumulation',4,
                    '--adaptation',adaptation,'--output',gate],'gate2_lora')
            reload(gate,'FRESH_GATE_RELOAD')
            stage='native8_real_motor_controls_NOT_SR';status('running');serve(gate,'gate')
            smoke=out/'SMOKE_COHORT.json'
            atomic_json(smoke,{'role':'development_selection','seeds':[20268401]})
            evaluate(gate,smoke,out/'native8',1,True);stop_server()
            stage='native_calibration128';status('running');cal=out/'calibration128'
            run(PY,'VLA.intern_r72.train',['--mode','fit','--updates',128,'--resume',gate,
                '--adaptation',adaptation,'--output',cal],'calibration128')
            reload(cal,'FRESH_CALIBRATION_RELOAD');p=read(cal/'TRAINING.json')
            if not p['loss_last32']<p['loss_first32']:raise ValueError('Calibration loss did not decrease; do not blindly train long')
            stage='native_finetune1024';status('running');cp=out/'fit1024'
            run(PY,'VLA.intern_r72.train',['--mode','fit','--updates',1024,'--resume',cal,
                '--adaptation',adaptation,'--output',cp],'fit1024');reload(cp,'FRESH_FIT1024_RELOAD')
            stage='development_prefix_pilots';status('running');serve(cp,'fit1024')
            parent_dev=ROOT/'VLA/audits/recovery_r56_20261017/automatic_v1/development40.json'
            source=read(parent_dev.with_name('pilot8.json'));source['role']='development_selection'
            pilot=out/'PILOT8.json';atomic_json(pilot,source);results={}
            for prefix in (1,5,10):
                dest=out/f'prefix{prefix}_pilot8';evaluate(cp,pilot,dest,prefix);results[prefix]=read(dest/'progress.json')
            prefix=select_prefix(results);atomic_json(out/'PREFIX_SELECTION.json',{'prefix':prefix,'development_only':True,'results':results})
            stage='development40_fit1024';status('running');dev=out/'development40_fit1024'
            evaluate(cp,parent_dev,dev,prefix);baseline=read(dev/'progress.json');stop_server()
            selected=cp;selected_dev=baseline
            if needs_rehearsal(baseline):
                stage='single_bounded_extension2048';status('running');next_cp=out/'fit2048'
                run(PY,'VLA.intern_r72.train',['--mode','fit','--updates',2048,'--resume',cp,
                    '--adaptation',adaptation,'--output',next_cp],'fit2048');reload(next_cp,'FRESH_FIT2048_RELOAD')
                serve(next_cp,'fit2048');dest=out/'development40_fit2048';evaluate(next_cp,parent_dev,dest,prefix)
                candidate=read(dest/'progress.json');stop_server()
                if candidate_wins(baseline,candidate):selected=next_cp;selected_dev=candidate
            atomic_json(out/'DEVELOPMENT_SELECTION.json',{'checkpoint':str(selected),'prefix':prefix,
                'per_action':selected_dev['per_action'],'confirmation_and_final800_not_read':True})
            stage='independent_confirmation40';status('running');serve(selected,'selected')
            conf=out/'CONFIRMATION40.json'
            atomic_json(conf,{'role':'development_confirmation','seeds':list(range(20268301,20268311))})
            evaluate(selected,conf,out/'confirmation40',prefix);stop_server()
            run(SIM,'VLA.intern_r72.report',['--output',out],'measured_report')
            stage='completed_finite_native_fit_and_actual_per_action_confirmation_NOT800';status('completed')
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
        with (a.output/'owner.log').open('x') as f:
            c=subprocess.Popen([str(PY),'-u','-m','VLA.intern_r72.watch','--output',str(a.output.resolve())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':c.pid,'status':str(a.output/'owner_status.json')})
    else:main(a)
