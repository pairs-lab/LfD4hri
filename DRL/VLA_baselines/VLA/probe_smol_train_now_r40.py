"""Read-only actual Smol TRAIN20 during a safe inter-seed800 window.

No800 outcomes are read/used for learning. A tiny CUDA reservation makes the
existing800 admission guard wait between seeds, NEVER interrupts an episode.
Only AFTER that seed worker exits and RAM is measured free is one capped
model loaded for the existing predeclared TRAIN20 fixture. No simulator,
training, model-input repair, success change or other process signal.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA import audit_smol_fit_r38 as native
from VLA import evaluate_smol800_r39 as cohort
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.wait_render_then_resume_queues import identity,render_jobs

ROOT=cohort.ROOT;BASE=cohort.BASE;SIM=cohort.SIM


def stable_window(state,child_process,gpu,expected_pids,initial_admission=True):
    return bool(state.get('status')=='running' and state.get('stage')=='waiting_RAM_before_fresh_seed_worker'
        and state.get('active_child_pid') is None and child_process is None
        and {int(p) for p in gpu['compute_pids']}<=set(expected_pids)
        and gpu['available_RAM_MiB']>=(6144 if initial_admission else 1024)
        and gpu['free_mib']>=(5120 if initial_admission else 512))


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);stage='checking_original_TRAIN_fixture';error=None
    deadline=time.monotonic()+a.wait_hours*3600
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'simulator_controls':0,
            'cohort800_outcomes_read_for_fitting_or_selection':False,'other_worker_or_episode_signalled':False})
    save()
    try:
        proof=json.loads((native.FIXTURE/'audit.json').read_text());request_file=native.FIXTURE/'requests_with_LOCAL_gold.json'
        if not proof['passed'] or proof['observations']!=20 or sha256_file(request_file)!=proof['request_file_sha256']:
            raise ValueError('Exact predeclared original TRAIN20 required')
        for path,digest in proof['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Selected TRAIN source changed')
        pointer=json.loads((BASE/'Smol_800_r39_active.json').read_text());root=Path(pointer['cycle'])
        if root.parent!=ROOT/'VLA/smolvla_scene67' or not root.name.startswith('test800_200peraction_'):raise ValueError('Unknown800 owner')
        state=json.loads((root/'status.json').read_text());parent=identity(state['pid']);lease=cohort.held_priority_lease()
        if (not parent or b'VLA.resume_smol800_r40' not in parent['tokens'] or str(root).encode() not in parent['tokens']
                or not lease):raise ValueError('Exact process-isolated800 + existing resource lease required')
        gpu=cohort.resources()
        if gpu['available_RAM_MiB']<2048 or gpu['free_mib']<512:raise ValueError('No safe capacity even for tiny reservation')
        import torch
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        reservation=torch.empty(1,device='cuda')
        stage='waiting_current800_seed_to_finish_NO_episode_interrupted';save();ready_since=None
        def window(initial_admission=True):
            owner=identity(state['pid']);current=cohort.held_priority_lease()
            if not owner or not current:return False
            s=json.loads((root/'status.json').read_text());child=identity(s.get('active_child_pid') or 0)
            gpu=cohort.resources();expected={os.getpid(),state['pid'],current['trainer_pid']}
            return stable_window(s,child,gpu,expected,initial_admission)
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Finite safe diagnostic window')
            if window() and not render_jobs():
                if ready_since is None:ready_since=time.monotonic()
                if time.monotonic()-ready_since>=2:break
            else:ready_since=None
            save();time.sleep(2)
        atomic_json(out/'resource_admission.json',{'resources':cohort.resources(),'cohort800_parent':state['pid'],
            'actual_seed_worker_absent':True,'two_seconds_stable_inter_seed_window':True,'own_model_allocator_cap_MiB':3072,
            'at_most_one_Smol_NN_resident_at_a_time':True,'existing_RDT_training_continues':True})
        original=native.inference_owner
        native.inference_owner=lambda:state['pid'] if window(initial_admission=False) else None
        stage='actual_predeclared_TRAIN20_native10_predictions';save()
        # Gold is local diagnostic data, never a model input. The observation
        # transport/policy/noise/metrics are the exact existing audited helper.
        try:native.run_probe(cohort.CHECKPOINT,json.loads(request_file.read_text()),'two_raw_views_warm',out/'actual_TRAIN20',state['pid'])
        finally:native.inference_owner=original
        atomic_json(out/'scope.json',{'actual_scope':'baseline_TRAIN20_in_inter_seed800_window_NOT_GTcrop_phase_or_SR',
            'fixture_sha256':sha256_file(native.FIXTURE/'audit.json'),'model':str(cohort.CHECKPOINT),
            'shared_audited_helper_sha256':sha256_file(ROOT/'VLA/audit_smol_fit_r38.py'),
            'only_helper_window_predicate_overridden':True,'no800_scene_image_label_or_outcome_sent_to_policy':True,
            'actual_neural_predictions':20,'simulator_controls':0,'training_updates':0})
        stage='completed_actual_TRAIN20_NOT_success_rate';save('completed');del reservation
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';save('failed_preserve_diagnostics_NOT_zero_SR');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=4);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=6:raise ValueError('Finite auxiliary diagnostic only')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if marker.exists() or a.output.exists():raise FileExistsError('No duplicate TRAIN probe')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.probe_smol_train_now_r40','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','MALLOC_ARENA_MAX':'2',
                    'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'no_simulator_episode_interrupted':True});print(child.pid)
