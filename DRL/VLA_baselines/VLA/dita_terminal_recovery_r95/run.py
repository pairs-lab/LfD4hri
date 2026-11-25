"""Continue only reload-qualified Dita10k -> actual800 -> model closure."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import traceback

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
import VLA.dita_close800_r94.common as common
import VLA.dita_close800_r94.run as original
from VLA.continuous_r86.owner import qualified
from .identity import await_identity

ROOT=common.ROOT
OUT=ROOT/'VLA/audits/dita_terminal800_r95_20261023'
OLD=ROOT/'VLA/audits/dita_close800_r94_20261023'
MODULE='VLA.dita_terminal_recovery_r95.run'
ORIGINAL_BINDINGS=common.bindings


def bindings():
    return {**ORIGINAL_BINDINGS(),**{str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}}


def configure():
    # Process-local scheduling namespaces only; no native/model/source edits.
    common.OUT=OUT
    original.OUT=OUT


def preflight():
    meta=qualified(common.CP); common.validate_checkpoint_metadata(meta)
    reload_path=common.CP/'FRESH_RELOAD_cuda.json'; reload=common.read(reload_path)
    if (not reload['passed'] or reload['device']!='cuda' or reload['actual_anchors']!=8
        or len(reload['max_errors'])!=8 or max(reload['max_errors'])>1e-6
        or reload_path.stat().st_mtime<(common.CP/'weights.safetensors').stat().st_mtime):
        raise ValueError('Require already completed exact-device native reload')
    old=common.read(OLD/'status.json')
    if common.alive(old['pid']): raise ValueError('Prior terminal coordinator is still alive')
    common.verify(common.read(OLD/'QUALIFICATION.json')['code'])
    return meta,reload


class Owner(original.Owner):
    def __init__(self):
        super().__init__(); self.code=bindings()

    def child(self,cmd,label,role):
        common.verify(self.code); requested=list(map(str,cmd))
        # Validate requested scope BEFORE launching anything. Do not weaken the
        # original signal allowlist to accept an unknown/transient identity.
        original.authorize_signal({'command':requested},signal.SIGTERM,role)
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(requested,cwd=ROOT,env=original.ENV,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT)
        try:
            expected,observations=await_identity(child,requested,os.getpid(),original.identity)
            original.authorize_signal(expected,signal.SIGTERM,role)
        except BaseException:
            # A just-created, unreaped Popen handle is ours, never a discovered
            # foreign PID. Only preauthorized non-fit commands can get here.
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            raise
        self.children.append({'identity':expected,'role':role})
        atomic_json(OUT/'CHILDREN.json',{'children':self.children})
        atomic_json(OUT/(label+'_POST_EXEC_IDENTITY.json'),{'requested_argv':requested,
            'registered_identity':expected,'startup_observations':observations,
            'full_command_parent_start_ticks_verified':True})
        return child

    def freeze_qualified(self,meta,reload):
        method=common.read(common.R86/'dita/method4096.json')
        common.verify(method['code']); common.verify(method['weights_sha256'])
        weights={p:h for p,h in method['weights_sha256'].items()
            if Path(p).resolve()!=common.R86/'dita/fit4096/weights.safetensors'}
        weights[str(common.CP/'weights.safetensors')]=meta['weights_sha256']
        method.update(checkpoint=str(common.CP),weights_sha256=weights,
            code={**method['code'],**meta['code'],**self.code},actual_checkpoint_optimizer_updates=10000,
            frozen_after_confirmation=False,frozen_for_user_requested_benchmark=True,
            terminal_checkpoint_no_further_Dita_finetune=True,
            benchmark_authorization={'user_requested_800_despite_failed_pilot_gate':True,
                'request':'finish current fit, test800, close Dita without further fine-tuning',
                'no_new_DEV_or_CONF_promotion_claim':True,'fixed_checkpoint_chosen_before_final_outcomes':True},
            metric='R62 semantic first-hit; NOT strict1cm; PUT release/settle not required',
            paired_scene_block_reused_from_earlier_Dita_NOT_new_independent_block=True,
            R95_only_process_registration_repair_native_model_and_physical_loop_unchanged=True)
        cohort={'role':'fresh_secondary_final800','seeds':common.SEEDS,
            'user_requested_benchmark_override':True,'scene_block_reused_for_checkpoint_comparison':True,
            'not_five_measured_seeds':True}
        from VLA.dita_close800_r94.evaluate import validate
        validate(method,cohort); common.verify(method['code']); common.verify(weights)
        atomic_json(OUT/'FROZEN_METHOD.json',method); atomic_json(OUT/'COHORT.json',cohort)
        atomic_json(OUT/'REUSED_NATIVE_RELOAD.json',{'receipt':str(common.CP/'FRESH_RELOAD_cuda.json'),
            'receipt_sha256':sha256_file(common.CP/'FRESH_RELOAD_cuda.json'),'actual_result':reload,
            'native_weight_hash_verified':meta['weights_sha256'],'no_new_optimizer_updates':True})
        atomic_json(ROOT/'VLA/frozen_models/dita.json',{'status':'terminal_checkpoint_frozen_benchmark_pending',
            'UTC':common.utc(),'checkpoint':str(common.CP),'updates':10000,'weights_sha256':meta['weights_sha256'],
            'method':str(OUT/'FROZEN_METHOD.json'),'method_sha256':sha256_file(OUT/'FROZEN_METHOD.json'),
            'further_finetune_authorized':False,'result_is_not_available_yet':True,'controller':str(OUT/'status.json')})


def main():
    configure(); OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'STARTED.json').exists(): raise FileExistsError('No duplicate recovered terminal benchmark')
        proof=common.read(OUT/'QUALIFICATION.json')
        if not proof['passed'] or proof['code']!=bindings(): raise ValueError('Repair CPU qualification required')
        meta,reload=preflight(); owner=Owner()
        atomic_json(OUT/'STARTED.json',{'UTC':common.utc(),'pid':os.getpid(),'code':owner.code,
            'completed_optimizer_updates':10000,'previous_technical_error':str(OLD/'ERROR.json')})
        owner.held={'owner':original.identity(os.getpid())}
        atomic_json(OUT/'RESERVATION.json',owner.held)
        with (OUT/'watchdog.log').open('x') as log:
            subprocess.Popen([str(common.CPU),'-u','-m',MODULE,'--watchdog'],cwd=ROOT,env=original.ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        import time
        deadline=time.monotonic()+30
        while not (OUT/'WATCHDOG_READY.json').exists():
            if time.monotonic()>deadline: raise RuntimeError('Recovered cleanup watchdog not ready')
            time.sleep(.1)
        atomic_json(OLD/'SUCCESSOR.json',{'UTC':common.utc(),'status':str(OUT/'status.json'),
            'prior_error_evidence_preserved':True,'same_checkpoint_no_new_finetune':True})
        old_status=common.read(OLD/'status.json')
        atomic_json(OLD/'status.json',{**old_status,'status':'superseded_by_terminal800_recovery',
            'successor_status':str(OUT/'status.json'),'previous_error':old_status.get('error'),
            'UTC':common.utc()})
        try:
            owner.stage='reuse_completed_native10k_and_passed_reload'; owner.status()
            owner.freeze_qualified(meta,reload); owner.benchmark()
        except BaseException as error:
            atomic_json(OUT/'ERROR.json',{'UTC':common.utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR':True})
            owner.status('technical_error_evidence_retained_NOT_completed800',error=str(error)); raise
        finally:
            atomic_json(OUT/'CLEANUP_DONE.json',original.cleanup({},owner.children))
        with (OUT/'other_models_launch.log').open('x') as log:
            process=subprocess.Popen([str(common.CPU),'-u','-m','VLA.dita_terminal_recovery_r95.others'],
                cwd=ROOT,env=original.ENV,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(OUT/'OTHER_MODELS_QUEUE.json',{'UTC':common.utc(),'pid':process.pid,
            'families':['vlanext_s'],'Dita_excluded':True})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--launch',action='store_true'); p.add_argument('--watchdog',action='store_true')
    args=p.parse_args(); configure()
    if args.watchdog: original.watchdog()
    elif args.launch:
        with (OUT/'owner.log').open('x') as log:
            process=subprocess.Popen([str(common.CPU),'-u','-m',MODULE],cwd=ROOT,env=original.ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':process.pid,'status':str(OUT/'status.json')})
    else: main()
