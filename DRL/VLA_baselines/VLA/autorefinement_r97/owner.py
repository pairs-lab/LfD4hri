"""Bounded, restartable audit/FT/DEV/CONF/final800 coordinator."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import requests

from .common import (ROOT,OUT,PARENT,R95,CPU,PY,SIM,PORT,ENV,TASKS,DEV,CONF,FINAL,
    read,utc,verify,bindings,metric_guard,score,atomic_json)
from .audit import audit
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import identity, alive, qualified
from VLA.benchmark_guardian_r89.common import live_children
from VLA.dita_terminal_recovery_r95.identity import await_identity

MODULE = 'VLA.autorefinement_r97.owner'


def module_name(ident):
    command = ident.get('command',[]) if ident else []
    return command[command.index('-m')+1] if '-m' in command else None


def ram_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024
    raise RuntimeError('Cannot read memory availability')


def exact_signal(ident, sig, idle_queue=False):
    if not ident or not alive(ident['pid']):
        return
    if identity(ident['pid'])!=ident:
        raise ValueError('Process identity changed; no signal sent')
    module = module_name(ident)
    if idle_queue:
        if module!='VLA.dita_terminal_recovery_r95.others' or sig not in (signal.SIGSTOP,signal.SIGCONT):
            raise ValueError('Only idle S scheduler pause/resume is authorized')
    elif module not in ('VLA.autorefinement_r97.train','VLA.autorefinement_r97.evaluate',
        'VLA.dita_transfer_r84.server') or sig not in (signal.SIGTERM,signal.SIGKILL):
        raise ValueError('Not an owned native worker')
    os.kill(ident['pid'],sig)


def cleanup():
    record = read(OUT/'RESERVATION.json') if (OUT/'RESERVATION.json').exists() else {}
    children = read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:
            exact_signal(ident,signal.SIGTERM)
    deadline = time.monotonic()+20
    while any(alive(i['pid']) and identity(i['pid'])==i for i in children) and time.monotonic()<deadline:
        time.sleep(.2)
    for ident in reversed(children):
        if alive(ident['pid']) and identity(ident['pid'])==ident:
            exact_signal(ident,signal.SIGKILL)
    queued = record.get('idle_S_scheduler')
    if queued and alive(queued['pid']) and identity(queued['pid'])==queued:
        exact_signal(queued,signal.SIGCONT,idle_queue=True)
    return {'UTC':utc(),'owned_workers_cleaned':True,'idle_S_scheduler_resumed':bool(queued)}


class Owner:
    def __init__(self):
        self.children = read(OUT/'CHILDREN.json')['children'] if (OUT/'CHILDREN.json').exists() else []
        self.child=self.server=None
        self.stage='preflight'
        self.bound=bindings()

    def status(self,status='running',**extra):
        atomic_json(OUT/'status.json',{'UTC':utc(),'pid':os.getpid(),'status':status,'stage':self.stage,
            'family':'dita','original_thresholds_preserved':True,'planned_final_trials':800,
            'episodes_per_action':200,'child_pid':self.child.pid if self.child and self.child.poll() is None else None,
            'server_pid':self.server.pid if self.server and self.server.poll() is None else None,**extra})

    def launch_child(self,python,module,args,label):
        verify(self.bound);metric_guard()
        command = list(map(str,[python,'-u','-m',module,*args]))
        with (OUT/(label+'.log')).open('x') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        try:
            ident,observations = await_identity(child,command,os.getpid(),identity)
        except BaseException:
            if child.poll() is None:
                child.terminate();child.wait(timeout=20)
            raise
        self.children.append(ident)
        atomic_json(OUT/'CHILDREN.json',{'children':self.children})
        atomic_json(OUT/(label+'_IDENTITY.json'),{'identity':ident,'startup':observations})
        return child

    def reserve(self):
        previous = read(OUT/'RESERVATION.json') if (OUT/'RESERVATION.json').exists() else {}
        held = previous.get('idle_S_scheduler')
        live = R95/'other_models/LIVE.json'
        if held is None and live.exists():
            state = read(live)
            ident = identity(state.get('pid',0))
            if (module_name(ident)=='VLA.dita_terminal_recovery_r95.others'
                and not state.get('child_pid') and not state.get('server_pid')
                and state.get('resource_wait') and not live_children(ident['pid'])):
                held=ident
        record = {'UTC':utc(),'owner':identity(os.getpid()),'idle_S_scheduler':held,
            'never_signal_foreign_optimizer':True}
        atomic_json(OUT/'RESERVATION.json',record)
        # Start cleanup before any scheduler pause. It can also recover an
        # unexpectedly dead coordinator from committed checkpoints/rows.
        with (OUT/f'watchdog_{os.getpid()}.log').open('x') as log:
            subprocess.Popen([str(CPU),'-u','-m',MODULE,'--watchdog','--watch-owner',str(os.getpid())],
                cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ready=OUT/f'WATCHDOG_READY_{os.getpid()}.json'
        deadline=time.monotonic()+20
        while not ready.exists():
            if time.monotonic()>deadline:
                raise RuntimeError('Cleanup watchdog not ready')
            time.sleep(.1)
        if held:
            if live_children(held['pid']):
                raise ValueError('S scheduler acquired a child; do not pause active training')
            exact_signal(held,signal.SIGSTOP,idle_queue=True)
            deadline=time.monotonic()+3
            while (Path('/proc')/str(held['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('T','t'):
                if time.monotonic()>deadline:
                    raise RuntimeError('Idle scheduler did not acknowledge pause')
                time.sleep(.02)
            if live_children(held['pid']):
                exact_signal(held,signal.SIGCONT,idle_queue=True)
                raise ValueError('Scheduler launch raced; no optimizer interrupted')
        self.status(idle_S_scheduler_reserved=held)

    def resources(self, minimum_ram=4400):
        from VLA.cross_vla800_r62.common import resources
        measured=OUT/'MEASURED_NATIVE_GATE.json'
        if measured.exists():
            minimum_ram=max(minimum_ram,read(measured)['measured_host_peak_RSS_MiB']+768)
        deadline=time.monotonic()+12*3600
        while True:
            value=resources()
            compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
            if (not compute and value['GPU_free_MiB']>=7500 and value['RAM_available_MiB']>=minimum_ram
                and value['disk_free_GiB']>=25):
                return value
            self.status(status='waiting_resources',resource_wait=value,minimum_RAM_MiB=minimum_ram)
            if time.monotonic()>deadline:
                raise TimeoutError('No safe native resource window; unrelated applications untouched')
            time.sleep(10)

    def wait(self,child,label,hours):
        deadline=time.monotonic()+hours*3600
        low=0
        while child.poll() is None:
            available=ram_available()
            low=low+1 if available<640 else 0
            if low>=3 or time.monotonic()>deadline:
                ident=next(i for i in self.children if i['pid']==child.pid)
                exact_signal(ident,signal.SIGTERM)
                try:child.wait(timeout=20)
                except subprocess.TimeoutExpired:exact_signal(ident,signal.SIGKILL);child.wait(timeout=10)
                raise RuntimeError('Resource interruption; resume only from committed evidence')
            detail=None
            for name in ('status.json','progress.json'):
                p=OUT/label/name
                if p.exists():
                    detail=read(p)
                    if 'rows' in detail:detail={k:v for k,v in detail.items() if k!='rows'}
            self.status(log=str(OUT/(label+'.log')),actual_child_progress=detail,RAM_available_MiB=available)
            time.sleep(3)
        self.child=None
        if child.returncode:
            raise RuntimeError(f'{label} exit{child.returncode}; retained log; NOT model SR0')

    def train(self,label,parent,additional,accumulation):
        completed=OUT/(label+'_COMPLETE.json')
        if completed.exists():
            cp=Path(read(completed)['checkpoint']);qualified(cp)
            return cp
        origin=qualified(parent)['updates'];resume=parent;target=origin+additional
        for attempt in range(3):
            name=f'{label}_attempt{attempt:02d}';folder=OUT/name
            done=folder/'COMPLETED.json'
            if done.exists():
                cp=Path(read(done)['checkpoint']);qualified(cp)
                atomic_json(completed,{'checkpoint':str(cp)})
                return cp
            if folder.exists():
                latest=folder/'LATEST_COMMITTED.json'
                if latest.exists():
                    resume=Path(read(latest)['checkpoint']);qualified(resume)
                    if qualified(resume)['updates']==target:
                        m=qualified(resume)
                        if len(m['counts'])!=4 or not all(m['changed_tensors'].values()):
                            raise ValueError('Committed terminal gate evidence incomplete')
                        atomic_json(completed,{'checkpoint':str(resume)})
                        return resume
                continue
            self.stage=label
            self.resources()
            args=['fit','--output',folder,'--resume',resume,'--additional-updates',target-qualified(resume)['updates'],
                '--accumulation',accumulation,'--phase-origin',origin,'--phase-horizon',additional]
            self.child=self.launch_child(PY,'VLA.autorefinement_r97.train',args,name)
            try:
                self.wait(self.child,name,4)
            except RuntimeError as error:
                if (folder/'LATEST_COMMITTED.json').exists():
                    resume=Path(read(folder/'LATEST_COMMITTED.json')['checkpoint']);qualified(resume)
                # Never retry a non-resource/data/model error blindly.
                tail=(OUT/(name+'.log')).read_text(errors='replace')[-12000:]
                if ('Resource interruption' not in str(error)
                    and not any(x in tail for x in ('out of memory','Killed')) and ram_available()>1200):
                    raise
                continue
            cp=Path(read(done)['checkpoint']);qualified(cp)
            atomic_json(completed,{'checkpoint':str(cp)})
            return cp
        raise RuntimeError('Bounded native continuation retries exhausted; all evidence retained')

    def reload(self,cp,label):
        receipt=cp/'FRESH_RELOAD_cuda.json'
        if receipt.exists() and read(receipt)['passed']:
            qualified(cp)
            return
        self.stage=label;self.resources()
        self.child=self.launch_child(PY,'VLA.autorefinement_r97.train',['reload','--checkpoint',cp],label)
        self.wait(self.child,label,1)
        if not read(receipt)['passed']:
            raise ValueError('Fresh exact-device native reload failed')

    def method(self,cp,label,final=False,confirmation_passed=False):
        m=qualified(cp)
        if not read(cp/'FRESH_RELOAD_cuda.json')['passed']:
            raise ValueError('Missing real CUDA reload')
        original=read(R95/'FROZEN_METHOD.json')
        weights={p:h for p,h in original['weights_sha256'].items() if not p.endswith('/weights.safetensors')}
        weights[str(cp/'weights.safetensors')]=m['weights_sha256']
        path=OUT/(label+'.json')
        value={'family':'dita','checkpoint':str(cp),'variant':'native_Dita_rgb3_sensor74_R97_low_LR_recovery',
            'weights_sha256':weights,'code':{**self.bound,**m['code']},
            'prefixes':{t:1 for t in TASKS},'actual_checkpoint_optimizer_updates':m['updates'],
            'timing':'10Hz actual observations/queries; native10; execute1; real2-frame context',
            'no_Oracle_or_geometric_target_action_residual':True,
            'frozen_after_confirmation':bool(final and confirmation_passed),
            'frozen_for_user_requested_benchmark':final,'R97_locked_before_final':final,
            'benchmark_authorization':{'user_requested_800_after_refinement':final,
                'readiness_gate_passed':bool(confirmation_passed),'no_invented_confirmation_pass':True},
            'metric':'original R62 semantic first-hit; NOT strict1cm; put release/settle not required',
            'original_closed_Dita10k_result_unchanged':True,
            'fresh_final_seed_block_NOT_paired_with_previous800':True}
        if path.exists() and read(path)!=value:
            raise ValueError('Refuse to change an existing method binding')
        atomic_json(path,value)
        return path

    def evaluate(self,method,label,role,seeds):
        finish=OUT/(label+'_COMPLETE.json')
        if finish.exists():
            p=Path(read(finish)['progress']);audit(p)
            return p
        cohort=OUT/(label+'_COHORT.json')
        value={'role':role,'seeds':seeds}
        if cohort.exists() and read(cohort)!=value:
            raise ValueError('Cannot change reserved evaluation instances')
        atomic_json(cohort,value)
        previous=None
        for attempt in range(4):
            name=f'{label}_attempt{attempt:02d}';folder=OUT/name
            progress=folder/'progress.json'
            if folder.exists():
                if progress.exists():
                    if read(progress)['status'].startswith('completed_actual_'):
                        result=audit(progress);atomic_json(folder/'AUDIT.json',result)
                        atomic_json(finish,{'progress':str(progress)})
                        return progress
                    previous=progress
                continue
            self.stage=label;self.resources()
            server_label=name+'_server'
            self.server=self.launch_child(PY,'VLA.dita_transfer_r84.server',
                ['--method',method,'--port',PORT],server_label)
            session=requests.Session();session.trust_env=False
            try:
                deadline=time.monotonic()+300
                while True:
                    if self.server.poll() is not None:
                        raise RuntimeError('Native server failed to load; inspect retained log')
                    try:
                        reply=session.get(f'http://127.0.0.1:{PORT}/health',timeout=2)
                        reply.raise_for_status()
                        if reply.json()['method_sha256']!=sha256_file(method):
                            raise ValueError('Another method occupies native server port')
                        break
                    except requests.RequestException:
                        if time.monotonic()>deadline:
                            raise TimeoutError('Native server readiness timeout')
                        self.status();time.sleep(2)
                args=['--method',method,'--cohort',cohort,'--output',folder,'--port',PORT]
                if previous:
                    args.extend(['--resume-progress',previous])
                self.child=self.launch_child(SIM,'VLA.autorefinement_r97.evaluate',args,name)
                try:
                    self.wait(self.child,name,20 if len(seeds)==200 else 2)
                except RuntimeError as error:
                    tail=(OUT/(name+'.log')).read_text(errors='replace')[-12000:]
                    if ('Resource interruption' not in str(error) and not any(t in tail
                        for t in ('ConnectionError','ReadTimeout','Connection reset','out of memory'))):
                        raise
                    if progress.exists():previous=progress
                    continue
                result=audit(progress)
                atomic_json(folder/'AUDIT.json',result)
                atomic_json(finish,{'progress':str(progress)})
                return progress
            finally:
                session.close()
                if self.server and self.server.poll() is None:
                    ident=next(i for i in self.children if i['pid']==self.server.pid)
                    exact_signal(ident,signal.SIGTERM)
                    try:self.server.wait(timeout=20)
                    except subprocess.TimeoutExpired:exact_signal(ident,signal.SIGKILL);self.server.wait(timeout=10)
                self.server=None
        raise RuntimeError('Finite evaluation technical retries exhausted; no fabricated trials')

    def run(self):
        self.reserve()
        gate=self.train('GPU_gate16',PARENT,16,4)
        self.reload(gate,'GPU_gate16_reload')
        # The actual short gate is now the resource-footprint qualification.
        atomic_json(OUT/'MEASURED_NATIVE_GATE.json',{'passed':True,
            'checkpoint':str(gate),'actual_updates':16,'exact_CUDA_reload':read(gate/'FRESH_RELOAD_cuda.json'),
            'measured_host_peak_RSS_MiB':qualified(gate)['host_peak_RSS_MiB'],
            'measured_peak_GPU_MiB':qualified(gate)['peak_GPU_MiB'],
            'not_a_robot_success_rate':True})
        cp=self.train('recovery512',gate,512,32)
        self.reload(cp,'recovery512_reload')
        base_method=self.method(PARENT,'baseline10k_method')
        base_progress=self.evaluate(base_method,'baseline_DEV40','development_selection',DEV)
        candidate_method=self.method(cp,'candidate512_method')
        candidate_progress=self.evaluate(candidate_method,'candidate512_DEV40','development_selection',DEV)
        options=[{'checkpoint':str(PARENT),'progress':str(base_progress),'score':list(score(read(base_progress)))},
            {'checkpoint':str(cp),'progress':str(candidate_progress),'score':list(score(read(candidate_progress)))}]
        if options[1]['score']>options[0]['score'] and (options[1]['score'][0]<34 or options[1]['score'][1]<7):
            cp2=self.train('recovery1024',cp,512,32)
            self.reload(cp2,'recovery1024_reload')
            method2=self.method(cp2,'candidate1024_method')
            p2=self.evaluate(method2,'candidate1024_DEV40','development_selection',DEV)
            options.append({'checkpoint':str(cp2),'progress':str(p2),'score':list(score(read(p2)))})
        chosen=max(options,key=lambda v:tuple(v['score']))
        atomic_json(OUT/'DEV_SELECTION.json',{'options':options,'selected':chosen,
            'tie_rule':'retain earlier checkpoint','final_and_confirmation_NOT_used_to_select':True})
        selected=Path(chosen['checkpoint'])
        method=self.method(selected,'selected_before_confirmation')
        conf=self.evaluate(method,'CONF40','development_confirmation',CONF)
        conf_score=score(read(conf));passed=conf_score[0]>=24 and conf_score[1]>=4
        frozen=self.method(selected,'FROZEN_METHOD',True,passed)
        atomic_json(OUT/'FROZEN_BEFORE_FINAL.json',{'checkpoint':str(selected),
            'method':str(frozen),'method_sha256':sha256_file(frozen),'confirmation_score':conf_score,
            'confirmation_readiness_passed':passed,'final_per_explicit_user_request_regardless_SR_gate':True,
            'seed_block':FINAL,'no_final_outcome_based_model_switch':True})
        final=self.evaluate(frozen,'final800','fresh_secondary_final800',FINAL)
        result=audit(final)
        atomic_json(OUT/'CERTIFIED_RESULT.json',result)
        atomic_json(ROOT/'VLA/frozen_models/dita_r97.json',{'status':'closed_after_new_actual800',
            'result':str(OUT/'CERTIFIED_RESULT.json'),'result_sha256':sha256_file(OUT/'CERTIFIED_RESULT.json'),
            'checkpoint':str(selected),'method':str(frozen),'original_dita_manifest_preserved':True})
        text='# Dita R97 — actual final800\n\n'
        text+='Original R62 semantic first-hit thresholds; NOT strict1cm; put release/settle not required.\n\n'
        text+='|Action|Successes / episodes|SR|Wilson95|\n|---|---:|---:|---|\n'
        for task,v in result['per_action'].items():
            lo,hi=v['Wilson95_percent']
            text+=f"|{task}|{v['successes']}/{v['n']}|{v['SR_percent']:.2f}%|{lo:.2f}–{hi:.2f}%|\n"
        text+='\nOne real cohort, not five measured seeds. New block is not paired with older benchmark blocks. All failures and safety aborts retained.\n'
        (OUT/'RESULTS.md').write_text(text)
        self.stage='completed_new_actual800_certified'
        self.status('completed',checkpoint=str(selected),per_action=result['per_action'])


def main():
    OUT.mkdir(exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        qualification=read(OUT/'QUALIFICATION.json')
        if not qualification['passed'] or qualification['code']!=bindings():
            raise ValueError('Current source/data qualification required')
        if (OUT/'CERTIFIED_RESULT.json').exists():
            return
        owner=Owner()
        atomic_json(OUT/'STARTED.json',{'UTC':utc(),'pid':os.getpid(),'code':owner.bound,
            'user_reauthorized_new_Dita_refinement':True,'preserve_old_closed800':True})
        try:
            owner.run()
        except BaseException as error:
            atomic_json(OUT/'ERROR.json',{'UTC':utc(),'error':str(error),'traceback':traceback.format_exc(),
                'NOT_model_SR_or_completed800':True})
            owner.status('technical_error_evidence_retained',error=str(error))
            raise
        finally:
            result=cleanup()
            atomic_json(OUT/f'CLEANUP_{os.getpid()}.json',result)


def watchdog(pid):
    reservation=read(OUT/'RESERVATION.json')
    expected=reservation['owner']
    if expected['pid']!=pid:
        raise ValueError('Watchdog owner reservation mismatch')
    atomic_json(OUT/f'WATCHDOG_READY_{pid}.json',{'UTC':utc(),'owner':expected})
    while alive(pid) and identity(pid)==expected:
        if (OUT/f'CLEANUP_{pid}.json').exists():
            return
        time.sleep(5)
    if not (OUT/f'CLEANUP_{pid}.json').exists():
        atomic_json(OUT/f'WATCHDOG_CLEANUP_{pid}.json',cleanup())
        # Abrupt death only, not a diagnosed integrity/data error. Bounded
        # restarts continue committed stages; do not restart model failures.
        starts=list(OUT.glob('WATCHDOG_RESTART_*.json'))
        if len(starts)<2 and not (OUT/'ERROR.json').exists() and not (OUT/'CERTIFIED_RESULT.json').exists():
            verify(read(OUT/'QUALIFICATION.json')['code'])
            with (OUT/f'restart_{len(starts)}.log').open('x') as log:
                p=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            atomic_json(OUT/f'WATCHDOG_RESTART_{len(starts)}.json',{'UTC':utc(),'pid':p.pid})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--launch',action='store_true')
    p.add_argument('--watchdog',action='store_true')
    p.add_argument('--watch-owner',type=int)
    args=p.parse_args()
    if args.watchdog:
        watchdog(args.watch_owner)
    elif args.launch:
        OUT.mkdir(exist_ok=True)
        if (OUT/'status.json').exists() and alive(read(OUT/'status.json')['pid']):
            raise ValueError('Refuse duplicate live coordinator')
        with (OUT/f'launch_{time.time_ns()}.log').open('x') as log:
            process=subprocess.Popen([str(CPU),'-u','-m',MODULE],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':process.pid,'status':str(OUT/'status.json')})
    else:
        main()
