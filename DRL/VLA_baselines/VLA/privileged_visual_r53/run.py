"""Finite two-model GT-visual ablation, only after the current owner exits.

Predeclare candidates before running. R52 FINAL outcomes/trajectories are
never read or used to select/refit. Only its already-frozen DEVELOPMENT
method/checkpoint is a parent/baseline. A separate R53 final120 is untouched.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .contract import source_hashes
from .scene_map import contract

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4'
BASE=ROOT/'VLA/audits/privileged_visual_r53_20261016'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
TASKS=('touch','pick','move','pick_place')


def read(path):return json.loads(Path(path).read_text())


def score(path):
    d=read(path)
    if d['completed_trials']!=d['planned_trials']:raise ValueError('Only complete cohorts may select a method')
    return d['successes'],-sum(bool(r.get('safety_violation')) for r in d['rows'])


def global_schedule(paths,family):
    from VLA.evaluate_task_chunks_r52 import choose_schedule
    result=choose_schedule(paths,family) # strict identity/complete matched8 checks
    native=30 if family=='xvla' else 8
    options=[]
    for path in paths:
        p=read(path);rows=p['rows']
        options.append({'prefix':p['native_execution_chunk'],'successes':sum(r['success'] for r in rows),
            'aborts':sum(bool(r.get('safety_violation')) for r in rows)})
    best=max(options,key=lambda o:(o['successes'],-o['aborts'],o['prefix']==native,o['prefix']))
    result.update(tasks={t:{'selected_prefix':best['prefix']} for t in TASKS},
        rule='ONE_global_prefix_most_complete8_success_then_fewer_aborts_then_native_then_longer',
        global_options=options,task_specific_overfit_avoidance=True)
    return result


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    child=server=server_log=None
    stage='validating_predeclared_CPU_inputs';results=[]
    def status(state='running',**kw):
        atomic_json(out/'status.json',{'status':state,'stage':stage,'pid':os.getpid(),
            'active_child_pid':None if child is None else child.pid,
            'active_server_pid':None if server is None else server.pid,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'results':results,**kw})
    def run(python,module,options,label):
        nonlocal child
        with (out/(label+'.log')).open('x') as log:
            child=subprocess.Popen([str(python),'-u','-m',module,*map(str,options)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
                    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
            status()
            while child.poll() is None:time.sleep(10);status()
            code=child.returncode
        child=None
        if code:raise RuntimeError(f'{label} integration/gate exit{code}; NOT a zero-SR outcome')
    def stop():
        nonlocal server,server_log
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        server=None
        if server_log is not None:server_log.close();server_log=None
    def serve(family,checkpoint,port,mode,label):
        nonlocal server,server_log
        from VLA.new_vlas.train import wait_gpu
        admission=out/'GPU_admission';admission.mkdir(exist_ok=True)
        wait_gpu(admission)
        python=ROOT/'VLA/.venvs'/('xvla' if family=='xvla' else 'openvla')/'bin/python'
        server_log=(out/(label+'_server.log')).open('x')
        module='VLA.privileged_visual_r53.server' if mode=='map3' else 'VLA.stable_native_r44.server'
        server=subprocess.Popen([str(python),'-u','-m',module,'--checkpoint',str(checkpoint),'--port',str(port)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=server_log,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
        until=time.monotonic()+1200
        while True:
            if server.poll() is not None:raise RuntimeError('Own server exited before health')
            try:
                r=requests.get(f'http://127.0.0.1:{port}/health',timeout=3);r.raise_for_status()
                if r.json()['checkpoint']!=str(checkpoint.resolve()):raise ValueError('Wrong actual model server')
                return
            except requests.RequestException:
                if time.monotonic()>until:raise TimeoutError('Own server not ready')
                time.sleep(3);status()
    try:
        data=PRIOR/'qualified_native_data'
        proofs={'xvla':BASE/'xvla_input_v3/audit.json','vla_adapter':BASE/'adapter_input_v3/audit.json'}
        for family,p in proofs.items():
            q=read(p)
            if not q['passed'] or q['family']!=family or q['profile']!=contract() or q['data_audit_sha256']!=sha256_file(data/'audit.json'):
                raise ValueError('Own current CPU3-image input proof missing')
        if [r['GT_map_RGB_sha256'] for r in read(proofs['xvla'])['rows']]!=[r['GT_map_RGB_sha256'] for r in read(proofs['vla_adapter'])['rows']]:
            raise ValueError('Map renderer differs across real model environments')
        for path in (BASE/'physical_world_geometry_v3/audit.json',BASE/'official_adapter_sequence_v1/audit.json'):
            if not read(path)['passed']:raise ValueError('Physical coordinate or official structural proof missing')
        from VLA.run_feedback_refinement_r52 import qualify_final_seeds
        final=list(range(20269801,20269831))
        if set(final)&set(range(20269901,20269931)):raise ValueError('Final cohorts overlap')
        exclusion=qualify_final_seeds(final)
        previous_geometry={r['physical_scene_sha256'] for r in read(PRIOR/'final_cohort_exclusion_audit.json')['cases']}
        if previous_geometry & {r['physical_scene_sha256'] for r in exclusion['cases']}:
            raise ValueError('R53 final geometry overlaps R52 final definition')
        exclusion['R52_final_definition_geometry_overlap']=0
        atomic_json(out/'final_geometry_exclusion.json',exclusion)
        atomic_json(out/'development_cohort.json',{'scope':'R53_DEVELOPMENT40_for_model_selection_NOT_final','seeds':list(range(20268011,20268021))})
        atomic_json(out/'final_cohort.json',{'scope':'R53_LOCKED_FINAL120_NOT_TRAIN_or_tuning','seeds':final,'episodes_per_action':30})
        frozen=source_hashes()
        atomic_json(out/'plan.json',{'models':['xvla','vla_adapter'],'observation':contract(),
            'prior_owner':str(PRIOR),'never_preempt_prior_trainer_or_evaluator':True,
            'input_proofs':{f:str(p) for f,p in proofs.items()},'frozen_source_hashes':frozen,
            'candidate_fit_budget':'1500_added_then3000_total_if_development_below_min40_max24_or_baselinePlus2',
            'fit_gates':'real2_Adam_updates_exact_fresh_reload_then_long_fit_with14GB_process_cap',
            'prefix_selection':'ONE_global_4_6_native_prefix_on_matched8_NOT_per_task_selection',
            'variant_selection':'highest_complete40_then_fewer_aborts_then_preserved_baseline_on_ties',
            'R52_final_outcomes_trajectories_or_scores_used':False,'R53_final_tunes_any_model':False,
            'maximum_new_fit_updates_each_family':3000,'guaranteed_SR':None,
            'motor_action_space_robot_physics_success_horizons_and_two_RGB_unchanged':True,
            'geometric_action_override_or_oracle_fallback':False})
        stage='waiting_R52_FINITE_owner_complete_NO_preemption_NO_final_outcome_reads';status()
        until=time.monotonic()+36*3600
        while True:
            prior=read(PRIOR/'status.json')
            # Deliberately never inspect or copy prior['results'] (FINAL SR).
            if prior['status']=='completed':
                if prior['active_child_pid'] is not None or prior['active_server_pid'] is not None:
                    raise ValueError('Prior still owns a child/server')
                break
            if prior['status'].startswith('failed'):raise RuntimeError('Prior R52 needs audited repair; no duplicate GPU jobs')
            if time.monotonic()>until:raise TimeoutError('Prior owner not completed within36h')
            time.sleep(20);status()
        if source_hashes()!=frozen:raise ValueError('Predeclared R53 code changed while queued')
        for family,port in [('xvla',5865),('vla_adapter',5866)]:
            folder=out/family;folder.mkdir()
            prior_model=PRIOR/family;baseline=read(prior_model/'frozen_final_method.json')
            parent=Path(baseline['checkpoint']);baseline_schedule=Path(baseline['schedule'])
            dev=(prior_model/'new_development40' if baseline['selected']=='new_recovery3000' else prior_model/'old_development40')
            if not dev.exists():dev=Path(read(prior_model/'reused_completed_original_development40.json')['source'])
            baseline_score=score(dev/'progress.json')
            prior_progress=read(dev/'progress.json')
            if (prior_progress['checkpoint']!=str(parent) or prior_progress['completed_trials']!=40
                    or read(prior_progress['cohort'])['seeds']!=read(out/'development_cohort.json')['seeds']):
                raise ValueError('Baseline must have exact same complete DEVELOPMENT40')
            atomic_json(folder/'baseline_DEVELOPMENT_ONLY_reference.json',{'checkpoint':str(parent),
                'schedule':str(baseline_schedule),'progress':str(dev/'progress.json'),
                'progress_sha256':sha256_file(dev/'progress.json'),'final_scores_read':False})
            python=ROOT/'VLA/.venvs'/('xvla' if family=='xvla' else 'openvla')/'bin/python'
            stage=family+'_explicit_observation_warm_migration';status()
            warm=folder/'WARM_START_ONLY_NOT_SERVE'
            run(SIM,'VLA.privileged_visual_r53.migrate',['--parent',parent,'--data',data,'--output',warm],family+'_warm')
            origin=read(warm/'config.json')['actual_optimizer_updates']
            common=['--family',family,'--data',data,'--total-updates',origin+3000,'--input-proof',proofs[family]]
            stage=family+'_real_GPU_three_image_gradient_memory_gate2';status()
            gate=folder/'gate2'
            run(python,'VLA.privileged_visual_r53.fit',[*common,'--mode','gate','--updates',origin+2,'--resume',warm,'--output',gate],family+'_gate2')
            cp=Path(read(gate/'latest_checkpoint.json')['checkpoint'])
            stage=family+'_fresh_process_three_image_reload';status()
            reload=folder/'reload2'
            run(python,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--output',reload],family+'_reload2')
            candidates=[];resume,proof=cp,reload/'audit.json'
            for budget in (1500,3000):
                stage=family+f'_actual_map_fit_{budget}_added_updates';status()
                fit=folder/f'fit{budget}'
                run(python,'VLA.privileged_visual_r53.fit',[*common,'--mode','long','--updates',origin+budget,
                    '--resume',resume,'--reload-proof',proof,'--output',fit],family+f'_fit{budget}')
                cp=Path(read(fit/'latest_checkpoint.json')['checkpoint'])
                stage=family+f'_map{budget}_native_prefix8_development_sweep';status()
                serve(family,cp,port,'map3',family+f'_map{budget}')
                paths=[]
                for prefix in (4,6,30 if family=='xvla' else 8):
                    pilot=folder/f'map{budget}_prefix{prefix}_pilot8'
                    run(SIM,'VLA.privileged_visual_r53.evaluate',['--family',family,'--port',port,
                        '--execution-steps',prefix,'--seed-count',2,'--output',pilot],family+f'_map{budget}_prefix{prefix}')
                    paths.append(pilot/'progress.json')
                schedule=folder/f'map{budget}_global_schedule.json';atomic_json(schedule,global_schedule(paths,family))
                stage=family+f'_map{budget}_complete_DEVELOPMENT40';status()
                current=folder/f'map{budget}_development40'
                run(SIM,'VLA.privileged_visual_r53.evaluate',['--family',family,'--port',port,'--schedule',schedule,
                    '--cohort',out/'development_cohort.json','--output',current],family+f'_map{budget}_dev40')
                current_score=score(current/'progress.json');stop()
                candidates.append({'budget':budget,'checkpoint':str(cp),'schedule':str(schedule),
                    'score_successes_negative_aborts':current_score,'progress':str(current/'progress.json')})
                atomic_json(folder/'development_candidates.json',candidates)
                if budget==3000 or current_score[0]>=min(40,max(24,baseline_score[0]+2)):break
                stage=family+'_completed1500_fresh_reload_before_predeclared_extension';status()
                reload=folder/'reload1500'
                run(python,'VLA.privileged_visual_r53.reload_gate',['--checkpoint',cp,'--completed-fit','--output',reload],family+'_reload1500')
                resume,proof=cp,reload/'audit.json'
            best=max(candidates,key=lambda c:tuple(c['score_successes_negative_aborts']))
            use_map=tuple(best['score_successes_negative_aborts'])>baseline_score
            chosen_cp=Path(best['checkpoint']) if use_map else parent
            chosen_schedule=Path(best['schedule']) if use_map else baseline_schedule
            mode='map3' if use_map else 'native2'
            atomic_json(folder/'frozen_final_method.json',{'checkpoint':str(chosen_cp),'schedule':str(chosen_schedule),
                'visual_mode':mode,'baseline_DEVELOPMENT40_score':baseline_score,'map_candidates':candidates,
                'selection_used_any_FINAL_outcome':False})
            stage=family+'_R53_LOCKED_final120_NO_tuning';status()
            serve(family,chosen_cp,port,mode,family+'_selected_final')
            final_dir=folder/'final120'
            run(SIM,'VLA.privileged_visual_r53.evaluate',['--family',family,'--port',port,'--visual-mode',mode,
                '--schedule',chosen_schedule,'--cohort',out/'final_cohort.json','--output',final_dir],family+'_final120')
            stop();final_score=score(final_dir/'progress.json')
            results.append({'family':family,'visual_mode':mode,'successes':final_score[0],'trials':120,
                'progress':str(final_dir/'progress.json'),'SR_above50percent':final_score[0]>60});status()
        stage='finite_GT_visual_ablation_and_independent_final120_completed';status('completed')
    except Exception as error:
        status('failed_preserve_evidence_NO_blind_retry',error=f'{type(error).__name__}: {error}');raise
    finally:stop()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--launch',action='store_true')
    a=parser.parse_args()
    if a.launch:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.with_suffix('.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-m','VLA.privileged_visual_r53.run','--output',str(a.output.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':child.pid,'output':str(a.output.resolve())}),flush=True)
    else:main(a)
