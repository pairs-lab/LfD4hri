"""Detached owner: safe own-process admission, native800, audit, uncertainty."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import time
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import ROOT, OUT, CP, SIM, MODEL_PY, PORT, read, resources


def verify():
    plan=read(OUT/'PREDECLARATION.json')
    sources=dict(plan['source_sha256'])
    review_sha=plan['qualitative_review_sha256']
    amendment=OUT/'PRE_EPISODE_DIAGNOSTIC_AMENDMENT.json'
    if amendment.exists():
        amended=read(amendment)
        if (amended['predeclaration_sha256']!=sha256_file(OUT/'PREDECLARATION.json')
            or amended['actual_new_evaluation_episodes_before_amendment']!=0):
            raise ValueError('Invalid pre-episode source amendment')
        for path,change in amended['source_amendments'].items():
            if sources[path]!=change['original_sha256']:raise ValueError('Amendment did not bind original source')
            sources[path]=change['new_sha256']
        review_sha=amended['clarified_qualitative_review_sha256']
    for path,digest in sources.items():
        if sha256_file(path)!=digest:raise ValueError('Declared benchmark source changed: '+path)
    if sha256_file(Path(plan['qualitative_review']))!=review_sha:
        raise ValueError('Bound qualitative review changed')
    for filename,key in (('config.json','checkpoint_config_sha256'),('trainable_masters.pt','checkpoint_master_sha256')):
        if sha256_file(CP/filename)!=plan[key]:raise ValueError('Frozen checkpoint changed')
    if sha256_file(Path(plan['schedule']))!=plan['schedule_sha256']:raise ValueError('Native schedule changed')
    proof=read(OUT/'NO_VIDEO_NONINTERFERENCE.json')
    if not proof['passed']:raise ValueError('No-video noninterference not qualified')
    if not read(OUT/'current_TRAIN_geometry_exclusion.json')['passed']:
        raise ValueError('Actual current TRAIN geometry exclusion missing')
    binding=OUT/'OWNER_SOURCE_BINDING.json'
    if binding.exists():
        for path,digest in read(binding).items():
            if sha256_file(path)!=digest:raise ValueError('Owner/report source changed during evaluation: '+path)


def main():
    if (OUT/'owner_status.json').exists():raise ValueError('Own output already has an owner; no duplicate800')
    verify()
    atomic_json(OUT/'OWNER_SOURCE_BINDING.json',
        {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    server=child=None;stage='resource_admission';attempt=0
    def status(state='running',**kw):
        p=OUT/'episodes/progress.json';progress=read(p) if p.exists() else {}
        atomic_json(OUT/'owner_status.json', {'status':state,'stage':stage,'pid':os.getpid(),
            'active_server_pid':None if server is None else server.pid,
            'active_child_pid':None if child is None else child.pid,
            'completed_trials':progress.get('completed_trials',0),'planned_trials':800,
            'UTC':datetime.now(timezone.utc).isoformat(),**kw})
    def stop_own_server():
        nonlocal server
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=10)
        server=None
    until=time.monotonic()+30*3600
    exclusive_only=False
    try:
        base_proof=read(ROOT/'VLA/audits/new_vlas_r43_20261016/xvla_download/source.json')
        expected=next(r['lfs_sha256'] for r in base_proof['files'] if r['name']=='model.safetensors')
        actual=sha256_file(ROOT/'VLA/checkpoints/xvla-base/model.safetensors')
        if actual!=expected:raise ValueError('Official base checkpoint changed')
        atomic_json(OUT/'base_weights_verified.json', {'passed':True,'sha256':actual,'revision':base_proof['revision']})
        while True:
            r=resources()
            can_start=(r['free_GPU_MiB']>=6000 and r['available_RAM_MiB']>=4500)
            if exclusive_only:can_start=can_start and r['free_GPU_MiB']>=14000
            if not can_start:
                status(resources=r,exclusive_fallback=exclusive_only)
                if time.monotonic()>until:raise TimeoutError('No safe resource slot within30h; no other job interrupted')
                time.sleep(20);continue
            attempt+=1;stage='bounded_native_server_load_and_fresh_prediction_parity'
            margin=1800 if attempt==1 else 1100
            fraction=min(10500,r['free_GPU_MiB']-margin)/r['total_GPU_MiB']
            proof=OUT/f'server_reload_attempt{attempt}.json'
            log=(OUT/f'server_attempt{attempt}.log').open('x')
            server=subprocess.Popen([str(MODEL_PY),'-u','-m','VLA.quantitative_r59.server',
                '--port',str(PORT),'--memory-fraction',str(fraction),'--proof',str(proof)],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status(resources=r,allocator_fraction=fraction)
            start=time.monotonic();resource_reject=False
            while True:
                if server.poll() is not None:break
                try:
                    response=requests.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
                    response.raise_for_status()
                    if response.json()['checkpoint']!=str(CP) or not proof.exists() or not read(proof)['passed']:
                        raise ValueError('Wrong weights or fresh parity proof')
                    break
                except requests.RequestException:pass
                now=resources();status(resources=now)
                if now['available_RAM_MiB']<900 or now['free_GPU_MiB']<700:
                    resource_reject=True;stop_own_server();break
                if time.monotonic()-start>900:raise TimeoutError('Own native server failed health within15min')
                time.sleep(3)
            log.close()
            if server is not None and server.poll() is None:
                break
            logfile=(OUT/f'server_attempt{attempt}.log').read_text()
            if not resource_reject and not any(word in logfile for word in ('out of memory','OutOfMemory','bad_alloc')):
                raise RuntimeError('Native server integration/parity error; inspect log, not a task failure')
            atomic_json(OUT/f'resource_rejection{attempt}.json', {'pre_episode_resource_rejection':True,
                '800_trials_started':False,'other_processes_interrupted':False})
            stage='waiting_safe_slot_after_bounded_resource_probe'
            # At most two own allocator-only startup probes; never an episode
            # reroll, never change precision/weights or kill another job.
            exclusive_only=attempt>=2
        verify();stage='actual_neural_800_no_video';status()
        with (OUT/'evaluation.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-u','-m','VLA.quantitative_r59.evaluate'],
                cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            last=-1
            while child.poll() is None:
                status()
                p=OUT/'episodes/progress.json'
                if p.exists():
                    n=read(p)['completed_trials']
                    if n!=last:
                        # Lightweight partial report; final audit is separate.
                        from .report import main as report
                        if n<800:report()
                        last=n
                time.sleep(10)
            if child.returncode:raise RuntimeError(f'Neural evaluator integration exit{child.returncode}; no fake failures')
        child=None;stop_own_server();verify();stage='completed800_independent_audit_and_uncertainty';status()
        from .report import main as report
        report();status('completed')
    except Exception as error:
        status('integration_or_resource_error_NOT_task_failure',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate();child.wait(timeout=20)
        stop_own_server()


if __name__=='__main__':main()
