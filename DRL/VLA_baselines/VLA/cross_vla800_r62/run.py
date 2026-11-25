"""Detached, serial admission after existing real fits; no other job preemption."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT,OUT,R59,MAIN,ADAPTER,SIM,PI,MODELS,SEEDS,TASKS,ENV,read,utc,blockers,resources,admissible,method_for,checkpoint_hashes,verify,matrix


def dependencies():
    paths=list(Path(__file__).parent.glob('*.py'))
    for name in ('evaluate_task_chunks_r52.py','evaluate_frozen_cohort_r52.py',
        'family_transfer_r54/evaluate.py','family_transfer_r54/server.py','family_transfer_r54/smol.py',
        'family_transfer_r54/runtime_pi0.py','recovery_r56/entry.py','autocycle_r55/evaluate.py',
        'autocycle_r55/decision.py','privileged_visual_r53/evaluate.py','privileged_visual_r53/server.py',
        'privileged_visual_r53/scene_map.py','stable_native_r44/server.py',
        'new_vlas/policy_outputs.py','new_vlas/action_contracts.py','benchmark.py'):
        paths.append(ROOT/'VLA'/name)
    old=read(R59/'PREDECLARATION.json')
    bound={str(p.resolve()):sha256_file(p) for p in paths}
    for p,h in old['environment_fingerprint']['sources_sha256'].items():bound[str(ROOT/p)]=h
    bound[old['protocol']]=old['protocol_sha256']
    return bound


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise ValueError('Owner already declared; refuse duplicate800 runs')
        with (out/'UNIT_TESTS.log').open('x') as stream:
            tests=subprocess.run([sys.executable,'-m','VLA.cross_vla800_r62.tests','-v'],
                cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        if tests.returncode:raise ValueError('CPU regression tests failed; no model evaluation launched')
        bound=dependencies();verify(bound);atomic_json(out/'CODE_AND_PLANT_BINDING.json',bound)
        cohort=read(R59/'cohort.json')
        if cohort['seeds']!=SEEDS or not read(R59/'current_TRAIN_geometry_exclusion.json')['passed']:
            raise ValueError('Wrong shared800scene cohort or missing actual TRAIN exclusion')
        atomic_json(out/'cohort.json',{**cohort,'scope':'R62_SHARED_R59_800_ONE_EVALUATION_COHORT_PER_MODEL',
            'previous_XVLA_outcomes_NEVER_used_for_selection':True})
        atomic_json(out/'PREDECLARATION.json',{'UTC':utc(),'selected_models':['xvla',*MODELS],
            'already_completed_XVLA_cohort_reused_not_rerun':str(R59),'per_action':200,'per_model':800,
            'seeds':SEEDS,'tasks':TASKS,'fit_owners_waited_without_preemption':[str(MAIN),str(ADAPTER)],
            'checkpoint_selection':'existing restricted40+40development ONLY',
            'same_robot_objects_profile_physics_safety_success':True,
            'primary_PUT':'legacy target-region first hit, NOT release/settle',
            'native_VLA_action_source_no_new_Oracle_or_geometric_repair':True,
            'original_render_mask_cadence_retained_without_writing_media':True,
            'four_additional_environment_cohorts_forecast_NOT_measured_seeds':True,
            'source_sha256':bound,'geometry_receipt':str(R59/'current_TRAIN_geometry_exclusion.json'),
            'geometry_receipt_sha256':sha256_file(R59/'current_TRAIN_geometry_exclusion.json')})
        from .report import measured,refresh
        p=read(R59/'episodes/progress.json');matrix(p['rows'])
        if not read(R59/'FINAL_AUDIT.json')['passed'] or not read(R59/'INDEPENDENT_FINAL_CERTIFICATE.json')['passed']:
            raise ValueError('Original X-VLA800 certificate failed')
        folder=out/'xvla';folder.mkdir(exist_ok=False)
        atomic_json(folder/'PRESERVED_ACTUAL_SOURCE.json',{'source_progress':str(R59/'episodes/progress.json'),
            'progress_sha256':sha256_file(R59/'episodes/progress.json'),
            'independent_certificate_sha256':sha256_file(R59/'INDEPENDENT_FINAL_CERTIFICATE.json'),
            'not_new_evaluation_or_retraining':True})
        measured('xvla',p['rows'],folder,R59/'episodes/progress.json');refresh(out)
        models={m:{'status':'waiting_existing_fits_and_gates','completed':0,'planned':800} for m in MODELS}
        stage='waiting_existing_R56_fits_and_frozen_tests_NO_preemption';child=server=log=None
        def status(state='running',**kw):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,
                'active_server_pid':None if server is None else server.pid,'models':models,'updated_UTC':utc(),**kw})
        def execute(module,opts,target,label):
            nonlocal child
            verify(bound)
            with (target/(label+'.log')).open('x') as stream:
                child=subprocess.Popen([str(SIM),'-u','-m',module,*map(str,opts)],cwd=ROOT,env=ENV,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                while child.poll() is None:
                    for model in MODELS:
                        f=out/model/'episodes/progress.json'
                        if f.exists():models[model]['completed']=read(f)['completed_trials']
                    status();refresh(out);time.sleep(10)
                code=child.returncode
            child=None
            if code:raise RuntimeError(label+f' exited{code}; integration evidence preserved, NOT a task failure')
        def stop_server():
            nonlocal server,log
            if server is not None and server.poll() is None:
                server.terminate()
                try:server.wait(timeout=30)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=15)
            server=None
            if log is not None:log.close();log=None
        try:
            deadline=time.monotonic()+a.wait_hours*3600;status()
            while blockers():
                if time.monotonic()>deadline:raise TimeoutError('Existing fits have not released resources within declared wait')
                status(existing_job_blockers=blockers());time.sleep(15)
            # Freeze all admitted choices before ANY new800episode result.
            admitted={}
            for model in MODELS:
                try:
                    method,source,key=method_for(model);dest=out/model;dest.mkdir(exist_ok=False)
                    evidence={str(source):sha256_file(source)}
                    for k in ('development_progress','confirmation_progress','schedule'):
                        if k in method:evidence[method[k]]=sha256_file(method[k])
                    frozen={'model':model,'method':method,'selection_score_DEVELOPMENT_only':key,
                        'selection_source':str(source),'selection_evidence_sha256':evidence,
                        'checkpoint_files_sha256':checkpoint_hashes(method['checkpoint']),
                        'new800outcomes_consumed_for_selection':False,'frozen_UTC':utc()}
                    atomic_json(dest/'FROZEN_BEFORE_800.json',frozen)
                    admitted[model]=method;models[model]['status']='qualified_frozen_waiting_resource_slot'
                except Exception as e:
                    models[model].update(status='not_admitted_integration_NOT_zero_SR',error=f'{type(e).__name__}: {e}')
                status()
            for index,(model,method) in enumerate(admitted.items()):
                dest=out/model;port=5912+index
                try:
                    stage=model+'_exclusive_resource_admission';status()
                    deadline=time.monotonic()+a.wait_hours*3600
                    while True:
                        r=resources()
                        if admissible(r) and not blockers():break
                        if time.monotonic()>deadline:raise TimeoutError('No safe slot; no other job stopped')
                        status(resource_admission=r);time.sleep(15)
                    stage=model+'_fixed_DEVELOPMENT_actions_no_media_noninterference';status()
                    execute('VLA.cross_vla800_r62.preflight',['--method',dest/'FROZEN_BEFORE_800.json',
                        '--output',dest/'no_media_qualification'],dest,'no_media_qualification')
                    if not read(dest/'no_media_qualification/audit.json')['passed']:raise ValueError('Media shim not qualified')
                    stage=model+'_native_server_load';status();verify(bound)
                    if model in ('smolvla','pi0'):
                        py=SIM if model=='smolvla' else PI
                        opts=['-m','VLA.recovery_r56.entry','server','--model',model,
                            '--checkpoint',method['checkpoint'],'--output',dest/'server_runtime','--port',port]
                        if method['mode']=='baseline':opts+=['--baseline']
                    else:
                        py=ROOT/'VLA/.venvs/openvla/bin/python'
                        module='VLA.privileged_visual_r53.server' if method['mode']=='map3' else 'VLA.stable_native_r44.server'
                        opts=['-m',module,'--checkpoint',method['checkpoint'],'--port',port]
                    log=(dest/'server.log').open('x')
                    server=subprocess.Popen([str(py),'-u',*map(str,opts)],cwd=ROOT,env=ENV,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    deadline=time.monotonic()+1800
                    while True:
                        if server.poll() is not None:raise RuntimeError('Own actual model server exited; inspect its log')
                        try:
                            reply=requests.get(f'http://127.0.0.1:{port}/health',timeout=5);reply.raise_for_status()
                            meta=reply.json()
                            if meta['checkpoint']!=method['checkpoint'] or meta.get('model',meta.get('family'))!=model:
                                raise ValueError('Wrong actual model/checkpoint health')
                            atomic_json(dest/'server_health.json',meta);break
                        except requests.RequestException:
                            if time.monotonic()>deadline:raise TimeoutError('Own native server health exceeded30min')
                            status();time.sleep(5)
                    stage=model+'_locked_actual800_no_video';models[model]['status']='running_actual_neural_800';status()
                    execute('VLA.cross_vla800_r62.evaluate',['--model',model,'--method',dest/'FROZEN_BEFORE_800.json',
                        '--port',port,'--cohort',out/'cohort.json','--output',dest/'episodes'],dest,'evaluation800')
                    stop_server();stage=model+'_independent_all_native_chunk_audit';status();verify(bound)
                    from .audit import audit
                    progress=audit(model,dest);measured(model,progress['rows'],dest,dest/'episodes/progress.json')
                    models[model].update(status='completed800_and_audit_and_labelled_forecast',completed=800)
                    refresh(out);status()
                except Exception as e:
                    models[model].update(status='integration_or_resource_error_NOT_zero_SR',error=f'{type(e).__name__}: {e}')
                    atomic_json(dest/'INTEGRATION_ERROR_NOT_MODEL_SR.json',{'error':str(e),'traceback':traceback.format_exc(),
                        'completed_trials_preserved':models[model]['completed'],'no_seed_reroll_or_fake_failures':True,'UTC':utc()})
                    stop_server();status();refresh(out)
            stage='finished_admitted_models_review_pending_any_errors'
            complete=all(s['status']=='completed800_and_audit_and_labelled_forecast' for s in models.values())
            status('completed_all_four_actual800' if complete else 'finished_with_unmeasured_integration_models');refresh(out)
        except Exception as e:
            status('orchestration_error_NO_fake_model_outcomes',error=f'{type(e).__name__}: {e}');raise
        finally:
            if child is not None and child.poll() is None:child.terminate();child.wait(timeout=30)
            stop_server()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--wait-hours',type=float,default=48);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        with (a.output/'owner.log').open('x') as stream:
            proc=subprocess.Popen([sys.executable,'-u','-m','VLA.cross_vla800_r62.run',
                '--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'owner_pid':proc.pid,'output':str(a.output.resolve())}))
    else:main(a)
