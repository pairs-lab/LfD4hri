"""One resource-aware owner for implemented, bounded recovery recipes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import requests
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.autonomy_r83.common import ROOT, OUT, R80, R82, SIM, EVO, ADAPTER, PORT, ENV, TASKS, DEV, CONF, FINAL, read, utc, code, verify, ready, score
from VLA.autonomy_r83.catalog import refresh


class Owner:
    def __init__(self):
        self.child = self.server = self.log = None
        self.stage = 'initialized'
        self.jobs = {name: {'status': 'queued'} for name in ('smolvla', 'vla_adapter', 'evo1')}
        self.bound = code()

    def state(self, status='running', **extra):
        state = {'pid': os.getpid(), 'updated_UTC': utc(), 'status': status, 'stage': self.stage,
            'active_child_pid': self.child.pid if self.child and self.child.poll() is None else None,
            'active_server_pid': self.server.pid if self.server and self.server.poll() is None else None,
            'jobs': self.jobs, **extra}
        atomic_json(OUT / 'owner_status.json', state)
        refresh(state)
        return state

    def event(self, stage, **extra):
        self.stage = stage
        with (OUT / 'EVENTS.jsonl').open('a') as journal:
            journal.write(json.dumps({'UTC': utc(), 'stage': stage, **extra}) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        self.state()

    def run(self, python, module, args, folder, label, hours=12):
        verify(self.bound)
        if module == 'VLA.autonomy_r83.train':
            self.resource_window()
        self.event(folder.name + '/' + label)
        with (folder / (label + '.log')).open('x') as log:
            self.child = subprocess.Popen([str(python), '-u', '-m', module, *map(str, args)],
                cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + hours * 3600
            while self.child.poll() is None:
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite stage budget exceeded: ' + label)
                self.state()
                time.sleep(10)
            result = self.child.returncode
        self.child = None
        if result:
            raise RuntimeError(f'{label} integration exit{result}; no fabricated model outcome')

    def resource_window(self):
        from VLA.cross_vla800_r62.common import resources
        deadline = time.monotonic() + 12 * 3600
        while True:
            r = resources()
            compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
            if not compute and r['GPU_free_MiB'] >= 13500 and r['RAM_available_MiB'] >= 6000 and r['disk_free_GiB'] >= 30:
                return
            if time.monotonic() > deadline:
                raise TimeoutError('No exclusive safe resource slot within12h; no other job stopped')
            self.state(waiting_resource_admission=r)
            time.sleep(20)

    def stop(self):
        for process in (self.child, self.server):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        if self.log:
            self.log.close()
        self.child = self.server = self.log = None

    def serve(self, method, folder, label):
        self.stop()
        self.resource_window()
        verify(self.bound)
        self.log = (folder / (label + '_server.log')).open('x')
        spec = read(method)
        python = EVO if spec['family'] == 'evo1' else SIM if spec['family'] == 'smolvla' else ADAPTER
        self.server = subprocess.Popen([str(python), '-u', '-m', 'VLA.autonomy_r83.server',
            '--method', str(method)], cwd=ROOT, env=ENV, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT)
        with requests.Session() as session:
            session.trust_env = False
            for _ in range(180):
                if self.server.poll() is not None:
                    raise RuntimeError('Own native model server failed; see preserved log')
                try:
                    r = session.get(f'http://127.0.0.1:{PORT}/health', timeout=2)
                    r.raise_for_status()
                    if r.json()['method_sha256'] != sha256_file(method):
                        raise ValueError('Port occupied by another model; never terminate it')
                    return
                except requests.RequestException:
                    self.state()
                    time.sleep(2)
        raise TimeoutError('Native server not ready within six minutes')

    def evaluate(self, method, folder, label, cohort):
        self.serve(method, folder, label)
        try:
            self.run(SIM, 'VLA.autonomy_r83.evaluate', ['--method', method, '--cohort', cohort,
                '--output', folder / label], folder, label, hours=16 if '800' in label else 6)
            p = read(folder / label / 'progress.json')
            if p['completed_trials'] != 4 * len(read(cohort)['seeds']):
                raise ValueError('Incomplete cohort cannot be a selection score')
            return p
        finally:
            self.stop()

    def method(self, folder, name, family, parent, variant, prefixes, checkpoint=None):
        from VLA.cross_vla800_r62.common import checkpoint_hashes
        if checkpoint:
            meta = read(checkpoint / 'ADAPTATION.json')
            stat = meta['stats']
            digests = {meta['weights']: meta['weights_sha256'],
                       str(checkpoint / 'ADAPTATION.json'): sha256_file(checkpoint / 'ADAPTATION.json')}
        elif family == 'evo1':
            stat = read(parent / 'stats.json')
            digests = {str(parent / 'learned.safetensors'): read(parent / 'TRAINING.json')['learned_sha256']}
        elif family == 'smolvla':
            stat = None
            digests = {str(parent / 'model.safetensors'): sha256_file(parent / 'model.safetensors')}
        else:
            stat = read(parent / 'config.json')['stats']
            digests = {str(parent / 'trainable_masters.pt'): sha256_file(parent / 'trainable_masters.pt')}
        path = folder / (name + '_METHOD.json')
        if path.exists():
            raise FileExistsError('Never overwrite a predeclared method')
        atomic_json(path, {'family': family, 'parent': str(parent), 'variant': variant,
            'checkpoint': str(checkpoint) if checkpoint else None, 'stats': stat,
            'prefixes': prefixes, 'code': self.bound, 'weights_sha256': digests,
            'no_controller_or_physics_or_success_changes': True})
        return path

    def promote(self, folder, selected, progress):
        if not ready(progress):
            self.jobs[folder.name] = {'status': 'completed_below_readiness_NOT800', 'reason': 'DEV threshold not met'}
            return
        conf = self.evaluate(selected, folder, 'confirmation40', OUT / 'CONFIRMATION40.json')
        if not ready(conf):
            self.jobs[folder.name] = {'status': 'completed_confirmation_below_readiness_NOT800',
                                      'reason': 'Preserve candidate; do not tune on confirmation'}
            return
        frozen = read(selected)
        frozen.update(frozen_after_confirmation=True, final_outcomes_used_for_selection=False,
            confirmation_per_action=conf['per_action'], fresh_secondary_final_not_original_R62=True)
        frozen_path = folder / 'FROZEN_FINAL.json'
        atomic_json(frozen_path, frozen)
        final = self.evaluate(frozen_path, folder, 'final800', OUT / 'FINAL800.json')
        from VLA.autonomy_r83.tests import audit_final
        audit = audit_final(folder / 'final800', frozen, FINAL)
        verify(self.bound)
        verify(frozen['weights_sha256'])
        atomic_json(folder / 'FINAL_AUDIT.json', audit)
        self.jobs[folder.name] = {'status': 'completed800_and_audited', 'per_action': final['per_action']}

    def adapt(self, family, existing):
        folder = OUT / family
        folder.mkdir(exist_ok=False)
        self.jobs[family] = {'status': 'running', 'stage': 'unchanged_baseline_DEV_then_balanced_masked_finetune'}
        parent = Path(existing['checkpoint'])
        if family == 'smolvla':
            prefixes = {t: int(existing['prefix']) for t in TASKS}
        else:
            schedule = read(existing['schedule'])
            prefixes = {t: int(schedule['tasks'][t]['selected_prefix']) for t in TASKS}
        baseline_method = self.method(folder, 'baseline', family, parent, 'baseline', prefixes)
        baseline = self.evaluate(baseline_method, folder, 'baseline_development40', OUT / 'DEVELOPMENT40.json')
        python = SIM if family == 'smolvla' else ADAPTER
        source, selected, best = None, None, baseline
        candidates = []
        for updates in (2, 512, 1536):
            fit = folder / f'fit{updates}'
            args = ['fit', '--family', family, '--parent', parent, '--output', fit, '--updates', updates]
            if source:
                args += ['--resume', source]
            self.run(python, 'VLA.autonomy_r83.train', args, folder, f'fit{updates}')
            proof = folder / f'RELOAD{updates}.json'
            self.run(python, 'VLA.autonomy_r83.train', ['reload', '--checkpoint', fit, '--output', proof],
                     folder, f'reload{updates}')
            if not read(proof)['passed']:
                raise ValueError('Actual fresh native reload failed')
            source = fit
            if updates == 2:
                continue
            # Use frequent re-observation, selected on DEV not a final cohort.
            method = self.method(folder, f'fit{updates}', family, parent, 'map3', {t: 1 for t in TASKS}, fit)
            dev = self.evaluate(method, folder, f'development40_fit{updates}', OUT / 'DEVELOPMENT40.json')
            candidates.append({'method': str(method), 'score': score(dev), 'per_action': dev['per_action']})
            if score(dev) > score(best):
                selected, best = method, dev
            if updates == 512 and score(dev)[0] == 0:
                # Do not consume another long fit merely because scalar loss drops.
                holds = sum(any(s['after'].get('correct_object_held', False)
                    for s in read(r['trajectory'])[r['task']]) for r in dev['rows'])
                if holds < 2:
                    break
        atomic_json(folder / 'SELECTION.json', {'baseline_score': score(baseline), 'candidates': candidates,
            'selected_method': str(selected) if selected else str(baseline_method),
            'selection_DEV_only_no_confirmation_or_final_read': True})
        if selected is None:
            self.jobs[family] = {'status': 'completed_no_DEV_improvement_baseline_preserved_NOT800'}
        else:
            self.promote(folder, selected, best)

    def evo_decode(self):
        family, folder = 'evo1', OUT / 'evo1'
        folder.mkdir(exist_ok=False)
        self.jobs[family] = {'status': 'running', 'stage': 'native_decoder_replan_ablation_NO_retraining'}
        cp = Path(read(R80 / 'DEVELOPMENT_SELECTION.json')['checkpoint'])
        baseline = self.method(folder, 'continuous10', family, cp, 'continuous', {t: 10 for t in TASKS})
        old = self.evaluate(baseline, folder, 'baseline_development40', OUT / 'DEVELOPMENT40.json')
        selected, best, evidence = None, old, []
        for prefix in (1, 5):
            method = self.method(folder, f'binary{prefix}', family, cp, 'binary', {t: prefix for t in TASKS})
            dev = self.evaluate(method, folder, f'binary{prefix}_development40', OUT / 'DEVELOPMENT40.json')
            evidence.append({'method': str(method), 'score': score(dev)})
            if score(dev) > score(best):
                selected, best = method, dev
            if ready(best):
                break
        atomic_json(folder / 'SELECTION.json', {'selected': str(selected) if selected else str(baseline),
            'baseline_score': score(old), 'candidates': evidence, 'no_new_optimizer_updates': True,
            'no_final800_outcomes_read_for_selection': True})
        if selected:
            self.promote(folder, selected, best)
        else:
            self.jobs[family] = {'status': 'completed_decoder_ablation_no_DEV_gain_NOT800'}


def main():
    from VLA.cross_vla800_r62.common import resources, method_for
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'owner_status.json').exists():
            raise FileExistsError('One owner/run only; never overwrite an existing attempt')
        owner = Owner()
        try:
            if not read(OUT / 'CPU_AUDIT.json')['passed'] or not read(OUT / 'IMAGE_REVIEW.json')['passed']:
                raise ValueError('Actual data audit and assistant image review required')
            verify(read(OUT / 'CPU_AUDIT.json')['code'])
            methods = {name: method_for(name)[0] for name in ('smolvla', 'vla_adapter')}
            atomic_json(OUT / 'PREDECLARATION.json', {'code': owner.bound, 'selected_parents': methods,
                'model_order': ['smolvla', 'vla_adapter', 'evo1'], 'max_new_updates_per_adapted_model': 1536,
                'checkpoints': [2, 512, 1536], 'DEV_minimum': '20/40 and >=2/10 each',
                'CONF_minimum': '20/40 and >=2/10 each', 'all_prior_results_and_best_weights_preserved': True,
                'final800_new_disjoint_secondary_cohort': FINAL,
                'current_R82_800_not_cancelled_or_used_for_model_selection': True,
                'all_recipe_changes_preimplemented_and_source_bound_NOT_arbitrary_auto_code_writes': True})
            for name, seeds in (('DEVELOPMENT40', DEV), ('CONFIRMATION40', CONF), ('FINAL800', FINAL)):
                role = 'development_selection' if name.startswith('DEV') else 'development_confirmation' if name.startswith('CONF') else 'fresh_secondary_final800'
                atomic_json(OUT / (name + '.json'), {'role': role, 'seeds': seeds})
            owner.event('waiting_current_R82_800_and_exclusive_GPU_NO_preemption')
            deadline = time.monotonic() + 36 * 3600
            while True:
                state = read(R82 / 'owner_status.json')
                current = resources()
                compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
                if state['status'] != 'running' and not compute and current['GPU_free_MiB'] >= 13500 and current['RAM_available_MiB'] >= 6000:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Finite36h prerequisite wait; existing worker never interrupted')
                owner.state(waiting_for_existing_R82_completed=state.get('completed_trials', 0))
                time.sleep(20)
            for family in ('smolvla', 'vla_adapter', 'evo1'):
                try:
                    verify(owner.bound)
                    owner.event('start_' + family)
                    owner.evo_decode() if family == 'evo1' else owner.adapt(family, methods[family])
                except Exception as error:
                    owner.stop()
                    folder = OUT / family
                    folder.mkdir(exist_ok=True)
                    atomic_json(folder / 'INTEGRATION_ERROR.json', {'UTC': utc(), 'error': str(error),
                        'traceback': traceback.format_exc(), 'not_model_SR0': True,
                        'all_partial_evidence_preserved_no_scene_rerolls': True})
                    owner.jobs[family] = {'status': 'quarantined_integration_error', 'reason': str(error)}
                finally:
                    owner.stop()
                    owner.state()
            owner.event('finite_implemented_queue_finished_review_recorded_evidence')
            owner.state('completed_with_quarantined_jobs' if any(v['status'].startswith('quarantined') for v in owner.jobs.values()) else 'completed')
        except Exception as error:
            owner.state('owner_integration_error', error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            owner.stop()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch', action='store_true')
    args = p.parse_args()
    if args.launch:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / 'owner.log').open('x') as log:
            process = subprocess.Popen([str(SIM), '-u', '-m', 'VLA.autonomy_r83.run'], cwd=ROOT,
                env=ENV, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': process.pid, 'output': str(OUT)}))
    else:
        main()
