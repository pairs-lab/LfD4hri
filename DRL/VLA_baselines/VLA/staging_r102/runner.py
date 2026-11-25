"""Bounded owned staging workers, preserving the independent live trainer."""
import argparse,fcntl,os,signal,subprocess,time
from pathlib import Path
from .common import OUT,R100,ROOT,read,utc,atomic_json,gpu_admissible,stop_reason
from VLA.admission_r100.common import ENV,CPU
from VLA.admission_r100.guard import resources,same_owned_process
from VLA.continuous_r86.owner import identity
from VLA.dita_terminal_recovery_r95.identity import await_identity

MODULES={'VLA.staging_r102.esm_probe','VLA.staging_r102.depth_scan','VLA.staging_r102.geometry_scan'}


def stop_owned(expected):
    if not same_owned_process(expected):raise ValueError('Staging worker identity changed; refusing signal')
    cmd=expected['command']
    if '-m' not in cmd or cmd[cmd.index('-m')+1] not in MODULES:
        raise ValueError('NEVER signal a trainer or the R100 workflow')
    os.kill(expected['pid'],signal.SIGTERM)


def worker(module,python,*,GPU=False):
    command=list(map(str,[python,'-u','-m',module]));name=module.rsplit('.',1)[-1]
    with (OUT/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=ROOT,env={**ENV,**({} if GPU else {'CUDA_VISIBLE_DEVICES':''})},
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    expected,_=await_identity(child,command,os.getpid(),identity)
    atomic_json(OUT/(name+'_IDENTITY.json'),expected);started=time.monotonic();reason=None
    try:
        while child.poll() is None:
            r=resources()
            current=read(R100/'WORKFLOW_STATUS.json')
            phase=current.get('stage','')
            train=R100/'gr00t_n15'/phase/'status.json'
            heartbeat_age=time.time()-train.stat().st_mtime if train.exists() else float('inf')
            reason=stop_reason(r,current,heartbeat_age,GPU)
            if reason:break
            if time.monotonic()-started>(1800 if GPU else 8*3600):
                reason='finite staging worker deadline';break
            atomic_json(OUT/'RUNNER_STATUS.json',{'UTC':utc(),'status':'running','worker':module,
                'child_pid':child.pid,'resources':r,'elapsed_s':time.monotonic()-started,
                'primary_processes_signalled':0});time.sleep(.5 if GPU else 3)
        if reason:
            stop_owned(expected)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if same_owned_process(expected):os.kill(expected['pid'],signal.SIGKILL)
                child.wait(timeout=15)
        value={'UTC':utc(),'returncode':child.returncode,'reason':reason,
            'status':'deferred_resources' if reason else ('completed' if child.returncode==0 else 'requires_diagnosis'),
            'primary_processes_signalled':0,'NOT_model_SR':True}
        atomic_json(OUT/(name+'_EXIT.json'),value);return value
    finally:
        if child.poll() is None:stop_owned(expected);child.wait(timeout=15)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        r=resources()
        if not (OUT/'esm_probe_EXIT.json').exists():
            if gpu_admissible(r):worker('VLA.staging_r102.esm_probe',ROOT/'VLA/.venvs/falcon_esm_r91/bin/python',GPU=True)
            else:atomic_json(OUT/'ESM_STATUS.json',{'UTC':utc(),'status':'deferred_safe_resource_gate','resources':r,
                'full_native_admission_previously_passed':True,'NOT_model_failure':True})
        # The full TRAIN scan adds missing exhaustive payload checks even if
        # a component job cannot safely share the current GPU.
        if not (OUT/'FULL_DEPTH_SCAN.json').exists():worker('VLA.staging_r102.depth_scan',CPU)
        if (OUT/'FULL_DEPTH_SCAN.json').exists() and not (OUT/'ALL_FRAME_GEOMETRY.json').exists():
            worker('VLA.staging_r102.geometry_scan',CPU)
        atomic_json(OUT/'RUNNER_STATUS.json',{'UTC':utc(),'status':'finite_staging_finished',
            'full_depth_scan_passed':(OUT/'FULL_DEPTH_SCAN.json').exists(),
            'ESM_component_passed':(OUT/'ESM_COMPONENT_PROOF.json').exists(),
            'all_frame_geometry_passed':(OUT/'ALL_FRAME_GEOMETRY.json').exists(),
            'primary_processes_signalled':0})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True,exist_ok=True)
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            child=subprocess.Popen([str(CPU),'-u','-m','VLA.staging_r102.runner'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'RUNNER_STATUS.json')},flush=True)
    else:main()
