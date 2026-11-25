"""Priority frozen Dita4096 benchmark; never kill a running training child."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import qualified
from VLA.dita_transfer_r84.common import PY
from VLA.cross_vla800_r62.common import resources
from .common import ROOT,OUT,R86,CP,SIM,PORT,SEEDS,bindings,identity,alive,same,read,verify,utc

ENV={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
    'TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled'}
MODULE='VLA.dita4096_benchmark_r92.run'


def proc_state(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
    except (FileNotFoundError,ProcessLookupError):return None


def owned_signal(expected,sig):
    if not same(expected):return False
    command=expected['command']
    own_eval='VLA.dita4096_benchmark_r92.evaluate' in command
    own_server=('VLA.dita_transfer_r84.server' in command and str(OUT/'FROZEN_METHOD.json') in command
        and str(PORT) in command)
    if not (own_eval or own_server):
        raise ValueError('Only this benchmark server/evaluator may be stopped')
    os.kill(expected['pid'],sig)
    return True


def resume_parent(expected):
    if same(expected) and proc_state(expected['pid']) in ('T','t'):
        if 'VLA.continuous_r86.owner' not in expected['command']:
            raise ValueError('Not the reserved queue parent')
        os.kill(expected['pid'],signal.SIGCONT)
        return True
    return False


def cleanup(reservation,children):
    stopped=[]
    for expected in reversed(children):
        if not same(expected) or not alive(expected['pid']):continue
        owned_signal(expected,signal.SIGTERM)
        deadline=time.monotonic()+20
        while same(expected) and alive(expected['pid']) and time.monotonic()<deadline:time.sleep(.25)
        if same(expected) and alive(expected['pid']):owned_signal(expected,signal.SIGKILL)
        deadline=time.monotonic()+10
        while same(expected) and alive(expected['pid']) and time.monotonic()<deadline:time.sleep(.25)
        if same(expected) and alive(expected['pid']):
            raise RuntimeError('Own GPU child did not exit; do not resume a competing queue yet')
        stopped.append(expected['pid'])
    resumed=resume_parent(reservation['parent'])
    return {'UTC':utc(),'only_benchmark_children_stopped':stopped,'R86_parent_resumed':resumed,
        'R86_optimizer_stop_or_kill_signals_sent':0}


def watchdog():
    with (OUT/'.watchdog.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reservation=read(OUT/'RESERVATION.json')
        atomic_json(OUT/'WATCHDOG_READY.json',{'UTC':utc(),'pid':os.getpid(),'owner':reservation['owner']})
        limit=time.monotonic()+72*3600
        while True:
            if (OUT/'CLEANUP_DONE.json').exists():return
            if not same(reservation['owner']) or not alive(reservation['owner']['pid']) or time.monotonic()>limit:
                children=read(OUT/'CHILDREN.json')['identities'] if (OUT/'CHILDREN.json').exists() else []
                result=cleanup(reservation,children)
                atomic_json(OUT/'WATCHDOG_RECOVERY.json',{**result,'owner_exited_or_watchdog_limit':True})
                return
            time.sleep(10)


class Owner:
    def __init__(self):
        self.stage='preflight';self.children=[];self.held=None;self.server=self.evaluator=None
        self.code=bindings()

    def status(self,status='running',**extra):
        atomic_json(OUT/'status.json',{'UTC':utc(),'pid':os.getpid(),'stage':self.stage,'status':status,
            'checkpoint':str(CP),'planned_trials':800,'episodes_per_action':200,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,
            'evaluator_pid':self.evaluator.pid if self.evaluator and self.evaluator.poll() is None else None,**extra})

    def child(self,cmd,label):
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        ident=identity(child.pid)
        if ident is None or ident['ppid']!=os.getpid():
            raise RuntimeError('Could not register own child identity')
        if 'VLA.dita4096_benchmark_r92.evaluate' in ident['command'] or 'VLA.dita_transfer_r84.server' in ident['command']:
            self.children.append(ident)
            atomic_json(OUT/'CHILDREN.json',{'identities':self.children})
        return child

    def preflight(self):
        subprocess.run([str(SIM),'-m','VLA.dita4096_benchmark_r92.tests'],cwd=ROOT,env=ENV,check=True)
        meta=qualified(CP)
        if meta['family']!='dita' or meta['updates']!=4096 or not read(CP/'FRESH_RELOAD_cuda.json')['passed']:
            raise ValueError('Require completed native CUDA4096 and actual fresh reload')
        method=read(R86/'dita/method4096.json')
        verify(method['code']);verify(method['weights_sha256'])
        if Path(method['checkpoint']).resolve()!=CP:
            raise ValueError('Historical method points at a different checkpoint')
        for python,original,name in ((PY,True,'DITA_NATIVE_MAP'),(SIM,False,'SIM_REFERENCE_MAP')):
            args=[str(python),'-u','-m','VLA.recovery_quality_r90.transport','--family','dita',
                '--output',str(OUT/(name+'.json'))]
            if original:args+=['--original']
            with (OUT/(name+'.log')).open('x') as log:
                subprocess.run(args,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        a,b=[read(OUT/(n+'.json')) for n in ('DITA_NATIVE_MAP','SIM_REFERENCE_MAP')]
        if a['TRAIN_cases']!=228 or a['case_hashes']!=b['case_hashes'] or a['combined_sha256']!=b['combined_sha256']:
            raise ValueError('Native/reference transport mismatch; no checksum tolerance')
        method.update(code={**method['code'],**self.code},actual_checkpoint_optimizer_updates=4096,
            frozen_after_confirmation=False,frozen_for_user_requested_benchmark=True,
            benchmark_authorization={'user_requested_800_despite_failed_pilot_gate':True,
                'request':'ủa sao dita mới mất không chạy 800 luôn đi','prior_DEV_successes':19,'prior_DEV_trials':40,
                'independent_confirmation_NOT_passed':True},
            no_final_data_used_to_select_or_update_this_frozen_method=True,
            paired_scene_blocks_with_Dita1024_not_five_empirical_seeds=True,
            metric='R62 semantic first-hit; NOT strict1cm; PUT release/settle not required')
        atomic_json(OUT/'FROZEN_METHOD.json',method)
        atomic_json(OUT/'COHORT.json',{'role':'fresh_secondary_final800','seeds':SEEDS,
            'paired_scene_seeds_with_Dita1024':True,'user_requested_benchmark_override':True})
        atomic_json(OUT/'PREFLIGHT.json',{'passed':True,'code':self.code,'checkpoint':str(CP),
            'native_checkpoint_updates':4096,'fresh_cuda_reload_passed':True,'transport_TRAIN_cases':228,
            'native_reference_exact':True,'no_new_GPU_model_loaded':True,
            'frozen_method_sha256':sha256_file(OUT/'FROZEN_METHOD.json')})

    def reserve(self):
        state=read(R86/'LIVE.json')
        if state['status']!='running' or state['family']!='vlanext_s' or state['stage']!='fit8192':
            raise ValueError('Expected exact S8192 phase changed; do not signal another stage blindly')
        parent,child=identity(state['pid']),identity(state['child_pid'])
        if (not parent or 'VLA.continuous_r86.owner' not in parent['command'] or not child
            or child['ppid']!=parent['pid'] or 'VLA.replacement_r84.train' not in child['command']
            or str(R86/'vlanext_s/fit8192') not in child['command']):
            raise ValueError('Exact queue/optimizer identity qualification failed')
        reservation={'UTC':utc(),'parent':parent,'optimizer_child':child,'owner':identity(os.getpid()),
            'signal_scope':'queue parent only; optimizer must continue to completion'}
        atomic_json(OUT/'RESERVATION.json',reservation)
        with (OUT/'watchdog.log').open('x') as log:
            self.watchdog=subprocess.Popen([str(SIM),'-u','-m',MODULE,'--watchdog'],cwd=ROOT,
                env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        limit=time.monotonic()+30
        while not (OUT/'WATCHDOG_READY.json').exists():
            if self.watchdog.poll() is not None or time.monotonic()>limit:raise RuntimeError('Resume watchdog not ready')
            time.sleep(.2)
        if not same(parent) or not same(child):raise ValueError('Process changed before reservation')
        fresh=read(R86/'LIVE.json')
        if fresh['stage']!='fit8192' or fresh.get('child_pid')!=child['pid']:
            raise ValueError('Phase changed before parent pause')
        self.held=reservation
        os.kill(parent['pid'],signal.SIGSTOP)
        if not same(child) or proc_state(child['pid']) in ('T','t'):
            raise RuntimeError('Optimizer must remain running; release queue reservation')
        atomic_json(OUT/'PARENT_HELD_CHILD_RUNNING.json',{'UTC':utc(),'reservation':reservation,
            'parent_state':proc_state(parent['pid']),'optimizer_state':proc_state(child['pid']),
            'training_interrupted_or_cancelled':False})
        self.stage='waiting_only_for_current_S8192_optimizer_no_new_stage_ahead_of_Dita800'
        limit=time.monotonic()+24*3600
        while same(child) and alive(child['pid']):
            progress=R86/'vlanext_s/fit8192/status.json'
            self.status(actual_other_optimizer_progress=read(progress) if progress.exists() else None,
                priority='Dita4096 fixed800 is next GPU workload',R86_queue_parent_only_paused=True)
            if time.monotonic()>limit:raise TimeoutError('Finite wait; resume queue, never kill optimizer')
            time.sleep(15)
        proof=qualified(R86/'vlanext_s/fit8192')
        if proof['updates']!=8192:raise ValueError('S8192 did not commit a complete optimizer checkpoint')
        self.stage='GPU_resource_admission_after_complete_S_checkpoint'
        limit=time.monotonic()+2*3600
        while True:
            r=resources()
            if r['GPU_free_MiB']>=6500 and r['RAM_available_MiB']>=4500 and r['disk_free_GiB']>=15:break
            self.status(resources=r)
            if time.monotonic()>limit:raise TimeoutError('No safe inference margin; preserve checkpoint and release queue')
            time.sleep(15)
        atomic_json(OUT/'RESOURCE_ADMISSION.json',{'UTC':utc(),'resources':r,'S8192_completed_before_inference':True,
            'no_optimizer_process_stopped':True})

    def benchmark(self):
        from VLA.benchmark_guardian_r89.audit import certify
        verify(self.code);verify(read(OUT/'FROZEN_METHOD.json')['weights_sha256'])
        self.stage='loading_actual_Dita4096_server';self.status()
        self.server=self.child([PY,'-u','-m','VLA.dita_transfer_r84.server','--method',OUT/'FROZEN_METHOD.json',
            '--port',PORT],'server')
        session=requests.Session();session.trust_env=False
        try:
            deadline=time.monotonic()+1200
            while True:
                if self.server.poll() is not None:raise RuntimeError('Actual native server failed; retain log, not SR0')
                try:
                    r=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2);r.raise_for_status();health=r.json()
                    if health['method_sha256']!=sha256_file(OUT/'FROZEN_METHOD.json') or health['checkpoint']!=str(CP):
                        raise ValueError('Wrong actual checkpoint at policy endpoint')
                    atomic_json(OUT/'NATIVE_HEALTH.json',health);break
                except requests.RequestException:
                    if time.monotonic()>deadline:raise TimeoutError('Native server startup limit')
                    self.status();time.sleep(2)
            self.stage='actual_Dita4096_frozen800_no_video';self.status()
            self.evaluator=self.child([SIM,'-u','-m','VLA.dita4096_benchmark_r92.evaluate',
                '--method',OUT/'FROZEN_METHOD.json','--cohort',OUT/'COHORT.json',
                '--output',OUT/'final800','--port',PORT],'evaluate')
            limit=time.monotonic()+36*3600
            while self.evaluator.poll() is None:
                if self.server.poll() is not None:raise RuntimeError('Native inference server exited during cohort')
                if time.monotonic()>limit:raise TimeoutError('Finite36h actual800 evaluation budget')
                path=OUT/'final800/progress.json';progress=read(path) if path.exists() else None
                if progress:progress={k:v for k,v in progress.items() if k!='rows'}
                self.status(actual_progress=progress);time.sleep(15)
            if self.evaluator.returncode:raise RuntimeError('Technical evaluator error; retain all completed outcomes')
            audit=certify(OUT/'final800/progress.json')
            if audit['actual_trials']!=800 or any(v['n']!=200 for v in audit['per_action'].values()):
                raise ValueError('800/200 per-action actual coverage missing')
            atomic_json(OUT/'CERTIFIED_RESULT.json',{**audit,'checkpoint':str(CP),'updates':4096})
            self.stage='completed_actual_Dita4096_800';self.status('completed',per_action=audit['per_action'])
        finally:session.close()


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'OWNER_STARTED.json').exists():raise FileExistsError('Never duplicate this fixed800')
        owner=Owner();atomic_json(OUT/'OWNER_STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':owner.code})
        try:
            owner.status();owner.preflight();owner.reserve();owner.benchmark()
        except BaseException as error:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'stage':owner.stage,'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('technical_interruption_NOT_zero_model_SR',error=str(error));raise
        finally:
            if owner.held:
                atomic_json(OUT/'CLEANUP_DONE.json',cleanup(owner.held,owner.children))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--watchdog',action='store_true')
    a=p.parse_args()
    if a.watchdog:watchdog()
    elif a.launch:
        OUT.mkdir(exist_ok=True,parents=True)
        with (OUT/'owner.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-u','-m',MODULE],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'owner_pid':child.pid,'status':str(OUT/'status.json'),'actual800_progress':str(OUT/'final800/progress.json')})
    else:main()
