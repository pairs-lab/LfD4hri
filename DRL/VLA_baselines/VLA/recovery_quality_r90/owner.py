"""Finite automatic TRAIN-only data qualification, native FT and real tests.

Never signal an existing owner/optimizer, never hot-patch its source or add
expert actions to a VLA rollout. Reuse completed weights only after R86 ends.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .audit import ROOT, OUT, read, development_audit

R86 = ROOT/'VLA/audits/continuous_r86_20261021'
SIM = ROOT/'VLA/third_party/lerobot/.venv/bin/python'
CPU = Path('/home/uet/miniconda3/envs/robot/bin/python')
ENV = {**os.environ, 'PYTHONPATH': str(ROOT), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
    'MKL_NUM_THREADS': '2', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false', 'WANDB_MODE': 'disabled'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def bindings():
    return {str(p.resolve()): sha256_file(p) for p in Path(__file__).parent.glob('*.py')}


def status(stage, **extra):
    atomic_json(OUT/'LIVE.json', {'stage': stage, 'UTC': utc(), 'pid': os.getpid(), **extra})


def run_cpu(module, args, label, hours=8):
    with (OUT/(label+'.log')).open('x') as log:
        child = subprocess.Popen([str(CPU), '-u', '-m', module, *map(str,args)], cwd=ROOT,
            env=ENV, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+hours*3600
        try:
            while child.poll() is None:
                if time.monotonic() > deadline: raise TimeoutError('Finite CPU phase limit: '+label)
                status(label, child_pid=child.pid, no_running_GPU_job_interrupted=True)
                time.sleep(15)
            if child.returncode: raise RuntimeError(label+' failed; retained log, not a model SR')
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=15)
                except subprocess.TimeoutExpired: child.kill(); child.wait()


def await_r86():
    deadline = time.monotonic()+72*3600
    while not (R86/'WORKFLOW_COMPLETE.json').exists():
        if (R86/'WORKFLOW_ERROR.json').exists():
            raise RuntimeError('R86 orchestration error requires diagnosis, no automatic takeover')
        if time.monotonic() > deadline: raise TimeoutError('Finite wait for current workflow')
        live = read(R86/'LIVE.json')
        status('waiting_for_current_R86_native_FT_and_DEV_to_finish_NO_preemption',
            prior_family=live.get('family'), prior_stage=live.get('stage'),
            actual_prior_progress=live.get('actual_child_progress'))
        time.sleep(20)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT/'OWNER_STARTED.json').exists(): raise FileExistsError('No duplicate continuation')
        qualification = read(OUT/'QUALIFICATION.json')
        if not qualification['passed'] or qualification['code'] != bindings():
            raise ValueError('Current code requires CPU regression and transport qualification')
        atomic_json(OUT/'OWNER_STARTED.json', {'UTC':utc(), 'pid':os.getpid(), 'code':bindings()})
        try:
            deadline = time.monotonic()+4*3600
            pilot = OUT/'near_miss_pilot/status.json'
            while not pilot.exists() or read(pilot)['status'] != 'pilot_complete_pending_data_admission_NOT_neural_SR':
                if time.monotonic() > deadline: raise TimeoutError('Pilot did not complete all declared attempts')
                status('waiting_for_actual_pilot_replay', pilot=read(pilot) if pilot.exists() else None)
                time.sleep(15)
            actual = read(pilot)
            if actual['completed'] != 16 or actual['actual_replay_passes'] != 16:
                raise ValueError('Two-target physical pilot not fully qualified; preserve and diagnose, no blind sweep')
            run_cpu('VLA.recovery_quality_r90.admit', ['--candidate',OUT/'near_miss_pilot',
                '--output',OUT/'qualified_pilot_ONLY','--minimum-targets',2], 'pilot_admission', 1)
            run_cpu('VLA.recovery_quality_r90.sweep', ['--output',OUT/'near_miss_sweep'], 'all_target_TRAIN_sweep', 12)
            run_cpu('VLA.recovery_quality_r90.admit', ['--candidate',OUT/'near_miss_sweep',
                '--output',OUT/'qualified_recovery','--minimum-targets',10], 'full_recovery_admission', 1)
            await_r86()
            # Change namespaces only in THIS new process. The original R86
            # process has finished; its source, dataset and artifacts stay intact.
            import VLA.continuous_r86.owner as old
            old.OUT = OUT
            old.prior.flow.OUT = OUT
            old.prior.dita.OUT = OUT/'dita'
            old.prior.flow.PORT = 5996
            old.prior.dita.PORT = 5997
            for family in ('dita','vlanext_s'): (OUT/family).mkdir(exist_ok=True)
            atomic_json(OUT/'dita/TEXT_FOUNDATION.json', read(R86/'dita/TEXT_FOUNDATION.json'))

            class Revision:
                def __init__(self):
                    super().__init__(); self.bound.update(bindings())
                def run(self, python, module, args, label, **kwargs):
                    if module in ('VLA.autonomy_r83.evaluate','VLA.recovery_r85.evaluate','VLA.continuous_r86.evaluate'):
                        module = 'VLA.recovery_quality_r90.evaluate'
                    return super().run(python, module, args, label, **kwargs)
                def method(self, checkpoint, name, **kwargs):
                    path = super().method(checkpoint, name, **kwargs)
                    value = read(path)
                    value['code'].update(bindings())
                    value['code'].update(read(checkpoint/'TRAINING.json')['code'])
                    value['R90_native_loss_and_action_timing_unchanged'] = True
                    value['R90_data_receipt'] = str(OUT/'qualified_recovery/ADMISSION.json')
                    atomic_json(path, value)
                    return path

            class Dita(Revision, old.Dita): pass
            class Flow(Revision, old.Flow): pass

            recipes = [
                ('dita', Dita, R86/'dita/fit4096', old.prior.dita.PY, 5120, 32),
                ('vlanext_s', Flow, R86/'vlanext_s/fit8192', old.prior.flow.PY, 9216, 16),
            ]
            for family, cls, parent, python, steps, accumulation in recipes:
                owner = cls()
                try:
                    selection = read(R86/family/'DEV_SELECTION.json')
                    if any(sum(v['successes'] for v in option['per_action'].values()) >= 24
                        and min(v['successes'] for v in option['per_action'].values()) >= 4
                        for option in selection['actual_options']):
                        owner.stage='existing_R86_DEV_already_qualified_preserve_checkpoint';
                        owner.status('no_unnecessary_recovery_finetune');continue
                    # DEV only, never use confirmation/final results to choose
                    # this new data/recipe or to select a different parent.
                    if not (parent/'TRAINING.json').exists():
                        raise ValueError('Expected completed native parent absent: '+str(parent))
                    old.qualified(parent)
                    owner.resource()
                    gate = OUT/family/'recovery_gate16'
                    parent_step = read(parent/'TRAINING.json')['updates']
                    owner.run(python, 'VLA.recovery_quality_r90.train', ['--family',family,'fit',
                        '--output',gate,'--updates',parent_step+16,'--accumulation',4,'--resume',parent],
                        'recovery_gate16', hours=1)
                    owner.run(python, 'VLA.recovery_quality_r90.train', ['--family',family,'reload',
                        '--checkpoint',gate], 'recovery_gate_reload', hours=1)
                    cp = OUT/family/f'recovery_fit{steps}'
                    owner.run(python, 'VLA.recovery_quality_r90.train', ['--family',family,'fit',
                        '--output',cp,'--updates',steps,'--accumulation',accumulation,'--resume',gate],
                        f'recovery_fit{steps}', hours=24)
                    owner.run(python, 'VLA.recovery_quality_r90.train', ['--family',family,'reload',
                        '--checkpoint',cp], f'recovery_fit{steps}_reload', hours=1)
                    method = owner.method(cp, 'recovery_method')
                    dev = owner.evaluate(method, 'recovery_DEV40', 'development_selection', list(range(20283001,20283011)))
                    atomic_json(OUT/family/'RECOVERY_DEV_DIAGNOSTIC.json', development_audit(OUT/family/'recovery_DEV40/progress.json'))
                    if not old.ready(dev):
                        owner.stage='recovery_DEV_not_promoted_NO_blind800';owner.status('needs_further_audit');continue
                    conf = owner.evaluate(method, 'recovery_CONF40', 'development_confirmation', list(range(20293101,20293111)))
                    if not old.ready(conf):
                        owner.stage='recovery_independent_confirmation_failed';owner.status('not_promoted');continue
                    method = owner.method(cp, 'FROZEN_RECOVERY_METHOD', frozen=True)
                    atomic_json(OUT/family/'FROZEN_FINAL.json', {'checkpoint':str(cp),'method':str(method),
                        'method_sha256':sha256_file(method),'confirmation_and_final_not_used_to_tune':True,
                        'fresh_secondary_final_seeds_NOT_old_XVLA_FLOWER_paired_cohort':True})
                    final = owner.evaluate(method, 'recovery_final800', 'fresh_secondary_final800', list(range(20294001,20294201)))
                    if final['completed_trials'] != 800 or any(v['n']!=200 for v in final['per_action'].values()):
                        raise ValueError('Actual800 coverage mismatch')
                    atomic_json(OUT/family/'FINAL_AUDIT.json', {'passed':True,'actual_trials':800,
                        'per_action':final['per_action'],'actual_cohorts_of800':1,'NOT_five_measured_seeds':True})
                    owner.stage='recovery_frozen800_completed';owner.status('completed')
                except Exception as error:
                    atomic_json(OUT/family/'INTEGRATION_ERROR.json', {'error':str(error),
                        'traceback':traceback.format_exc(),'NOT_model_SR':True})
                    owner.status('quarantined_evidence_retained_next_model_continues',error=str(error))
                finally:
                    owner.stop()
            status('finite_recovery_workflow_finished_read_actual_per_model_results')
        except Exception as error:
            atomic_json(OUT/'WORKFLOW_ERROR.json', {'UTC':utc(),'error':str(error),
                'traceback':traceback.format_exc(),'NOT_model_SR_or_training_success':True})
            status('needs_diagnosis_NO_blind_training', error=str(error))
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    if a.launch:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT/'owner.log').open('x') as log:
            child = subprocess.Popen([str(CPU),'-u','-m','VLA.recovery_quality_r90.owner'],cwd=ROOT,env=ENV,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print({'pid':child.pid,'status':str(OUT/'LIVE.json')})
    else:
        main()
