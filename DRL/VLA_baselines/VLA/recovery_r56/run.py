"""Finite, real GPU repair -> reload -> fit -> development -> frozen test.

Uses prior TRAIN and DEVELOPMENT evidence only. Prior final outcomes are not
consumed by selection. No other owner's process is killed or preempted.
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
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import ROOT, DATA, SIM, PI, SMOL, BASE as R54_BASE, read, verify

BASE = ROOT/'VLA/audits/recovery_r56_20261017'
REPAIRED_PI = BASE/'pi0_metadata_gate2/checkpoint'
PI_RELOAD = BASE/'pi0_metadata_gate2/fresh_reload8.json'
ADAPTER = ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/vla_adapter_round1/fit1500/checkpoints/step-009000'


def alive(pid):
    try:
        return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = {str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
    env = {**os.environ, 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
        'XLA_PYTHON_CLIENT_PREALLOCATE':'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION':'.82'}
    child = server = server_log = None
    stage = 'qualification'
    selected, errors = {}, {}
    def status(state='running', **kw):
        value = {'status':state, 'stage':stage, 'pid':os.getpid(),
            'active_child_pid':None if child is None else child.pid,
            'active_server_pid':None if server is None else server.pid,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'selected_methods':selected, 'errors':errors, **kw}
        atomic_json(a.output/'status.json', value)
        lines = ['# R56 — integration repairs and real native fine-tuning', '',
            'Stage: '+stage, 'Updated UTC: '+value['updated_at_utc'], '',
            'Smol: explicit concrete configuration registration in every clean process.',
            'pi0: separately copied missing schema metadata; original gate2 weights and FP32/counters proof preserved.',
            'Adapter: actual TRAIN gradient and native-head overfit diagnostic, not another blind long fit.',
            'X-VLA: prior selected weights stay frozen; no old final outcome enters training.', '',
            'Same two real cameras, UR5e/Robotiq, objects, physics, motor interface, safety and success rules.',
            'Privileged simulator observations are disclosed. Neural VLA chooses actions; no new Oracle/IK policy.',
            'Place primary metric remains goal-region entry, not release/settling.', '',
            'Selected methods: `'+json.dumps(selected, ensure_ascii=False)+'`',
            'Integration errors (not zero success rates): `'+json.dumps(errors, ensure_ascii=False)+'`', '',
            'Training loss and GPU gates do not establish task success. No >50% or global maximum is guaranteed.']
        (a.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def execute(py, module, opts, label, gpu=False):
        nonlocal child
        verify(frozen)
        if gpu:
            from VLA.new_vlas.train import wait_gpu
            folder=a.output/'GPU_admission';folder.mkdir(exist_ok=True);wait_gpu(folder)
        with (a.output/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(py), '-u', '-m', module, *map(str, opts)], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            status()
            while child.poll() is None:
                time.sleep(10);status()
            code=child.returncode
        child=None
        if code:
            raise RuntimeError(f'{label}: exit {code}; retained integration evidence, NOT zero model SR')
    def entry(py, op, opts, label, gpu=True):
        execute(py, 'VLA.recovery_r56.entry', [op, *opts], label, gpu)
    def stop():
        nonlocal server, server_log
        if server is not None and server.poll() is None:
            server.terminate()
            try: server.wait(timeout=30)
            except subprocess.TimeoutExpired: server.kill();server.wait(timeout=10)
        server=None
        if server_log is not None:server_log.close();server_log=None
    def serve(model, method, port, label):
        nonlocal server, server_log
        from VLA.new_vlas.train import wait_gpu
        admission=a.output/'GPU_admission';admission.mkdir(exist_ok=True);wait_gpu(admission)
        cp=Path(method['checkpoint']).resolve()
        if model in ('smolvla', 'pi0'):
            py=SIM if model=='smolvla' else PI
            module='VLA.recovery_r56.entry'
            opts=['server','--model',model,'--checkpoint',cp,'--port',port,'--output',a.output/(label+'_runtime')]
            if method['mode']=='baseline':opts+=['--baseline']
        else:
            py=ROOT/'VLA/.venvs'/('xvla' if model=='xvla' else 'openvla')/'bin/python'
            module='VLA.privileged_visual_r53.server' if method['mode']=='map3' else 'VLA.stable_native_r44.server'
            opts=['--checkpoint',cp,'--port',port]
        verify(frozen)
        server_log=(a.output/(label+'_server.log')).open('x')
        server=subprocess.Popen([str(py),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=server_log,stderr=subprocess.STDOUT)
        until=time.monotonic()+1800
        while True:
            status()
            if server.poll() is not None:raise RuntimeError('Own server exited before native health')
            try:
                result=requests.get(f'http://127.0.0.1:{port}/health',timeout=5)
                result.raise_for_status()
                if result.json()['checkpoint']!=str(cp):raise ValueError('Wrong checkpoint server')
                return
            except requests.RequestException:
                if time.monotonic()>until:raise TimeoutError('Native health wait exceeded30min')
                time.sleep(5)
    def evaluate(model, method, cohort, folder, port, label):
        opts=['--model',model,'--port',port,'--mode',method['mode'],'--cohort',cohort,'--output',folder]
        opts+=['--prefix',method['prefix']] if model in ('smolvla','pi0') else ['--schedule',method['schedule']]
        execute(SIM,'VLA.autocycle_r55.evaluate',opts,label)
        execute(SIM,'VLA.autocycle_r55.audit_video',['--model',model,'--progress',folder/'progress.json',
            '--output',folder/'video_audit'],label+'_video_decode')
    def assess(model, cp, port, label):
        method={'checkpoint':str(cp.resolve()),'mode':'map3','preserved_baseline':False}
        serve(model,method,port,label)
        paths=[]
        for prefix in (2,4,5):
            folder=a.output/f'{label}_prefix{prefix}_pilot8'
            execute(SIM,'VLA.autocycle_r55.evaluate',['--model',model,'--port',port,'--mode','map3',
                '--prefix',prefix,'--cohort',a.output/'pilot8.json','--output',folder],f'{label}_prefix{prefix}')
            paths.append(folder/'progress.json')
        from VLA.family_transfer_r54.run import choose_prefix
        method['prefix']=choose_prefix(paths)
        for name in ('development40','confirmation40'):
            folder=a.output/(label+'_'+name)
            evaluate(model,method,a.output/(name+'.json'),folder,port,label+'_'+name)
            method['development_progress' if name=='development40' else 'confirmation_progress']=str(folder/'progress.json')
        stop()
        return method
    try:
        atomic_json(a.output/'plan.json', {'code':frozen,'data':str(DATA),
            'data_audit_sha256':sha256_file(DATA/'audit.json'),
            'smol_budget':'actual2 gate + fresh8 then2500 TOTAL added updates, optimizer resumed from gate',
            'pi0_budget':'reuse verified2 update gate qualification; actual1500 independent official-foundation LoRA fit; if not development-stable one3000 fit',
            'adapter_diagnostic':'8 actual TRAIN maxXY anchors, all4 tasks; actual full-model gradient then64 cached-feature native-head-only steps; NO production CP exported',
            'old_checkpoint_source_or_simulator_edits':False,'old_final_outcomes_used_for_tuning':False,
            'new_final_seeds':list(range(20269601,20269631)),
            'all_four_methods_freeze_before_new_final':True,'guaranteed_SR':None})
        for name,role,seeds in (
            ('pilot8','development_prefix_pilot',[20268001,20268002]),
            ('development40','development_selection',list(range(20268011,20268021))),
            ('confirmation40','development_confirmation',list(range(20268101,20268111))),
            ('final120','untouched_final',list(range(20269601,20269631)))):
            atomic_json(a.output/(name+'.json'),{'role':role,'scope':'R56_'+role,'seeds':seeds})
        stage='waiting_own_existing_pi0_reload8_NO_GPU_preemption';status()
        until=time.monotonic()+1800
        while a.predecessor_pid and alive(a.predecessor_pid):
            if time.monotonic()>until:raise TimeoutError('Existing fresh pi0 reload exceeded30min')
            time.sleep(10);status()
        if not PI_RELOAD.exists():
            entry(PI,'reload_pi0',['--checkpoint',REPAIRED_PI,'--output',PI_RELOAD],'pi0_reload8_retry')
        if not read(PI_RELOAD).get('passed'):raise ValueError('pi0 native reload qualification failed')
        stage='smolvla_clean_process_actual_gate2';status()
        entry(SIM,'smol-config',[],'smol_clean_config',False)
        gate=a.output/'smol_gate2'
        entry(SIM,'fit_smol',['--mode','gate','--updates',2,'--output',gate],'smol_gate2')
        cp=Path(read(gate/'latest_checkpoint.json')['checkpoint']);proof=a.output/'smol_reload2.json'
        entry(SIM,'reload_smol',['--checkpoint',cp,'--output',proof],'smol_reload2')
        stage='adapter_TRAIN_gradient_and_native_head_overfit_DIAGNOSTIC';status()
        try:
            execute(ROOT/'VLA/.venvs/openvla/bin/python','VLA.recovery_r56.probe_adapter',
                ['--checkpoint',ADAPTER,'--output',a.output/'adapter_TRAIN_probe'],'adapter_TRAIN_probe',True)
        except Exception as error:
            errors['adapter_TRAIN_probe']=f'{type(error).__name__}: {error}';status()
        stage='smolvla_actual2500_added_optimizer_updates';status()
        fit=a.output/'smol_fit2500'
        entry(SIM,'fit_smol',['--mode','long','--updates',2500,'--resume',cp,
            '--reload-proof',proof,'--output',fit],'smol_fit2500')
        cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
        entry(SIM,'reload_smol',['--checkpoint',cp,'--output',a.output/'smol_reload2500.json'],'smol_reload2500')
        stage='smolvla_actual_native_DEVELOPMENT80';status()
        candidate=assess('smolvla',cp,5895,'smol_new2500')
        # Compare against the preserved baseline on the SAME two development
        # cohorts. Never compare training loss or different final seeds.
        baseline={'checkpoint':str(SMOL),'mode':'baseline','prefix':5,'preserved_baseline':True}
        prior=R54_BASE/'automatic_four_family_v1/smol_preserved_baseline_development40/progress.json'
        old=read(prior)
        baseline['prefix']=old['prefix'];baseline['development_progress']=str(prior)
        serve('smolvla',baseline,5895,'smol_baseline_confirmation')
        folder=a.output/'smol_baseline_confirmation40'
        evaluate('smolvla',baseline,a.output/'confirmation40.json',folder,5895,'smol_baseline_confirmation40')
        stop();baseline['confirmation_progress']=str(folder/'progress.json')
        from VLA.autocycle_r55.decision import candidate_key, stable
        selected['smolvla']=max((candidate,baseline),key=candidate_key)
        atomic_json(a.output/'smolvla_selection.json',{'selected':selected['smolvla'],
            'candidates':[candidate,baseline],'used_final_outcomes':False});status()
        for budget in (1500,3000):
            stage=f'pi0_actual{budget}_native_EEF_LoRA_updates';status()
            fit=a.output/f'pi0_fit{budget}'
            entry(PI,'fit_pi0',['--updates',budget,'--qualification',PI_RELOAD,
                '--input-proof',R54_BASE/'pi0_input_final/audit.json','--output',fit],f'pi0_fit{budget}')
            cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
            entry(PI,'reload_pi0',['--checkpoint',cp,'--output',a.output/f'pi0_reload{budget}.json'],f'pi0_reload{budget}')
            stage=f'pi0_{budget}_actual_native_DEVELOPMENT80';status()
            candidate=assess('pi0',cp,5896,f'pi0_new{budget}')
            if 'pi0' not in selected or candidate_key(candidate)>candidate_key(selected['pi0']):selected['pi0']=candidate
            atomic_json(a.output/f'pi0_candidate{budget}.json',candidate);status()
            if stable(selected['pi0']):break
        # All four final methods freeze before reading ANY new final outcome.
        prior=read(ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/status.json')['selected_methods']
        selected.update({name:prior[name] for name in ('xvla','vla_adapter')})
        from VLA.run_feedback_refinement_r52 import qualify_final_seeds
        geometry=qualify_final_seeds(list(range(20269601,20269631)))
        new={r['physical_scene_sha256'] for r in geometry['cases']}
        old=set()
        for path, key in ((ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/geometry_exclusion.json','final'),):
            old.update(r['physical_scene_sha256'] for r in read(path)[key]['cases'])
        for path in (ROOT/'VLA/audits/privileged_visual_r53_20261016/automatic_ablation_v2/final_geometry_exclusion.json',
            ROOT/'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/final_cohort_exclusion_audit.json'):
            old.update(r['physical_scene_sha256'] for r in read(path)['cases'])
        if new & old:raise ValueError('New final physical geometry overlaps old final')
        atomic_json(a.output/'final_geometry_exclusion.json',geometry)
        atomic_json(a.output/'ALL_METHODS_FROZEN_BEFORE_FINAL.json',{'methods':selected,
            'new_final_definition_sha256':sha256_file(a.output/'final120.json'),
            'development_only_selection':True,'all_prior_final_outcomes_not_used':True})
        for index, model in enumerate(('smolvla','pi0','xvla','vla_adapter')):
            stage=model+'_LOCKED_final120_NO_subsequent_tuning';status()
            method=selected[model];port=5895+index
            serve(model,method,port,model+'_locked_final')
            evaluate(model,method,a.output/'final120.json',a.output/(model+'_final120'),port,model+'_final120')
            stop()
        stage='completed_actual_fits_and_matched_final_cohorts_REVIEW_each_model';status('completed')
    except Exception as error:
        status('failed_preserve_all_evidence_NOT_model_zero_SR',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        stop()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--predecessor-pid',type=int)
    p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            command=[sys.executable,'-u','-m','VLA.recovery_r56.run','--output',str(a.output.resolve())]
            if a.predecessor_pid:command+=['--predecessor-pid',str(a.predecessor_pid)]
            owner=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':owner.pid,'output':str(a.output.resolve())}))
    else:
        main(a)
