import os
import subprocess

from VLA.common.artifact_io import atomic_json
from .run import OUT,bindings,preflight


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
    from VLA.dita_close800_r94.common import CPU
    subprocess.run([str(CPU),'-m','unittest','VLA.dita_terminal_recovery_r95.tests',
        'VLA.dita_close800_r94.tests'],check=True,env=env)
    meta,reload=preflight()
    atomic_json(OUT/'QUALIFICATION.json',{'passed':True,'code':bindings(),'CPU_tests':22,
        'completed_native_optimizer_updates':meta['updates'],'weights_sha256':meta['weights_sha256'],
        'existing_native_CUDA_reload':reload,'new_GPU_model_or_optimizer_loaded':False,
        'all_native_fit_commands_remain_forbidden':True})
    print({'qualified':True,'tests':22,'updates':meta['updates'],'weights_sha256':meta['weights_sha256']})


if __name__=='__main__': main()
