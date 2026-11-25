"""Finite CPU-only paired-data preparation, no premature FALCON finetune."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import ROOT,SIM,read


def main(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'owner_status.json').exists():raise FileExistsError('Do not duplicate full data owner')
        child=None;stage='wait_recovery12_exact_physical_pairing';env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
        def status(state='running',**extra):
            atomic_json(out/'owner_status.json',{'status':state,'stage':stage,'pid':os.getpid(),
                'active_child_pid':None if child is None else child.pid,**extra})
        def download():
            from huggingface_hub import snapshot_download
            folder=ROOT/'VLA/checkpoints/candidates_r68/falcon_pcd'
            file='falcon-3dpc-fc-calvin-abc/ckpts/falcon_3dpc_fc_calvin_abc.pt'
            snapshot_download('FALCON-VLA/FALCON-series',revision='902eec199093478b500edd2cde2d8728c556d5de',
                local_dir=folder,max_workers=1,allow_patterns=[file,'falcon-3dpc-fc-calvin-abc/logs/logs.json'])
            digest=sha256_file(folder/file)
            if digest!='6884c55d4d99797bc11fee026b96b70125b95c4a57542fae20afe682f8424dd6':
                raise ValueError('Official PCD checkpoint SHA differs')
            result={'downloaded_and_SHA_verified':True,'model':str(folder/file),'sha256':digest,
                'variant':'FALCON repository 3DPC / paper Kosmos-VLA RGB-D, NOT FALCON-ESM',
                'native_optimizer_step_verified':False,'robot_SR_measured':False}
            atomic_json(out/'MODEL_DOWNLOAD.json',result);return result
        try:
            status();deadline=time.monotonic()+1800
            pilot=out/'recovery12_r73_v2'
            while not (pilot/'AUDIT.json').exists():
                if (pilot/'ERROR.json').exists():raise ValueError('Recovery replay candidate failed; do not render full bank')
                if time.monotonic()>deadline:raise TimeoutError('Recovery pairing wait exceeded30min')
                time.sleep(5)
            if not read(pilot/'AUDIT.json')['passed_pairing_replay']:raise ValueError('Recovery replay not qualified')
            stage='all2185_TRAIN_metric_depth_PCD_candidate_and_pinned_model_download';status()
            atomic_json(out/'FULL_PREPARATION_PLAN.json',{'all_qualified_TRAIN_episodes':2185,'frames':119495,
                'reuses_RGB_not_duplicate_video_OR_RGBA_source':True,'depth_from_actual_camera_not_AABB_meshes':True,
                'only_CPU_data_preparation_no_new_GPU_model_process':True,
                'native_FALCON_admission_and_finetune_NOT_started':True,
                'replay_script_sha256':sha256_file(ROOT/'VLA/falcon_depth_bank_r73.py')})
            with ThreadPoolExecutor(max_workers=1) as pool, (out/'full_bank.log').open('x') as log:
                future=pool.submit(download)
                child=subprocess.Popen([str(SIM),'-u','-m','VLA.falcon_depth_bank_r73','--mode','all',
                    '--output',str(out/'full_TRAIN_depth_r73')],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+16*3600
                while child.poll() is None:
                    if time.monotonic()>deadline:raise TimeoutError('Finite16h full pairing limit exceeded')
                    status(model_download_finished=future.done());time.sleep(10)
                code=child.returncode;child=None
                if code:raise RuntimeError(f'Exact full TRAIN pairing exited{code}; preserve data/error, no silent filtering')
                model=future.result(timeout=3600)
            result=read(out/'full_TRAIN_depth_r73/AUDIT.json')
            if result['episodes']!=2185 or result['frames']!=119495 or not result['passed_pairing_replay']:
                raise ValueError('Incomplete paired bank')
            stage='paired_TRAIN_complete_requires_native_FALCON_qualification_and_finetune';status('completed',model=model)
        except Exception as error:
            status('preparation_error_NOT_model_SR',error=str(error));raise
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'VLA/audits/falcon_depth_pairing_r72_20261019')
    parser.add_argument('--launch',action='store_true');a=parser.parse_args()
    if a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        py=ROOT/'VLA/.venvs/flower_r69/bin/python'
        with (a.output/'owner.log').open('x') as f:
            child=subprocess.Popen([str(py),'-u','-m','VLA.falcon_prepare_owner_r73','--output',str(a.output.resolve())],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'status':str(a.output/'owner_status.json')})
    else:main(a)
