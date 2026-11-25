"""Actual two-view warm / three-view fitted Smol TRAIN-action diagnostics.

Loads ONE additional capped model only during the exact GT-view120 inference
phase, never alongside a fit. No simulator stepping, oracle deployment,
training, test-seed selection or changes to the primary model's private RNG.
Gold actions are used only AFTER prediction to compute local diagnostics.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT
from VLA.run_flow20_parallel_r37 import resources
from VLA.wait_render_then_resume_queues import identity,render_jobs
from VLA.scene64_conditioning import encode_numeric,validate_scene
from VLA.audit_native_policy_fit_r37 import metrics

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
FIT=BASE/'Smol_GTcrop_motion_r36_v6_priority'
FIXTURE=BASE/'native_TRAIN_action_probe_r37_v2'
WARM=ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'


def inference_owner():
    try:
        state=json.loads((FIT/'status.json').read_text());child=json.loads((FIT/'active_child.json').read_text());p=identity(child['pid'])
        if (state['status']=='running' and state['stage']=='GTcrop_development120' and child['stage']=='GTcrop_development120'
                and p and p['cwd']==str(ROOT) and b'VLA.evaluate_target_crop_r36' in p['tokens']
                and str(FIT/'development120').encode() in p['tokens']):return child['pid']
    except (OSError,KeyError,ValueError):pass
    return None


def run_probe(checkpoint, items, mode, output, evaluator_pid):
    import torch
    from VLA.smolvla.scene67_adapter import Scene67SmolVLA
    from VLA.smolvla.target_crop import TargetCropSmolVLA
    from VLA.common.tasks import TaskSpec
    from VLA.smolvla.bounded_feedback_r38 import load_stats
    output.mkdir(parents=True,exist_ok=False);policy=None;rows=[];first=None
    allrows=json.loads((Path(json.loads((FIXTURE/'audit.json').read_text())['source'])/'index.json').read_text())
    bypath={r['file']:r for r in allrows};cache={};error=None
    mean,std=load_stats(checkpoint)
    def save(status):
        errors=[r['metrics']['XYZ_error_mm'] for r in rows]
        atomic_json(output/'audit.json',{'status':status,'checkpoint':str(checkpoint),'view_variant':mode,
            'rows':rows,'completed_predictions':len(rows),'planned_predictions':len(items),'error':error,
            'median_XYZ_error_mm':float(np.median(errors)) if errors else None,
            'gripper_sign_correct':sum(r['metrics']['gripper_sign_matches'] for r in rows),
            'scope':'20_selected_original_TRAIN_fit_checks_NOT_SR_or_generalization',
            'same_flow_steps':10,'simulator_controls':0,'teacher_actions_provided_to_policy':False,
            'one_extra_model_allocator_cap_MiB':3072,'primary_evaluator_pid_NOT_signalled':evaluator_pid})
    save('running_actual_neural_TRAIN_predictions')
    try:
        for item in items:
            if inference_owner()!=evaluator_pid:raise ValueError('Exact primary inference window ended; preserve partial diagnostics')
            path=Path(item['source_file']);fi=item['source_frame'];raw=validate_scene(item['payload']['scene64'])
            if path not in cache:
                with np.load(path,allow_pickle=False) as z:cache[path]={k:z[k].copy() for k in
                    ('front_rgb','wrist_rgb','front_pick_mask','wrist_pick_mask','joint_positions','gripper_closure','prompt')}
            data=cache[path];np.testing.assert_allclose(raw[:7],np.r_[data['joint_positions'][fi],data['gripper_closure'][fi]],rtol=0,atol=1e-6)
            task=TaskSpec.from_dict(bypath[path.name]['task'])
            if task.prompt!=str(data['prompt'].item()) or task.prompt!=item['payload']['instruction']:raise ValueError('Original task instruction changed')
            env=SimpleNamespace(normalized_gripper_closure=lambda:float(raw[6]),
                target_mask=lambda camera,target: data[camera+'_pick_mask'][fi])
            obs=SimpleNamespace(front_rgb=data['front_rgb'][fi],wrist_rgb=data['wrist_rgb'][fi],
                joint_positions=data['joint_positions'][fi],prompt=task.prompt,step=fi)
            if policy is None:
                cls=Scene67SmolVLA if mode=='two_raw_views_warm' else TargetCropSmolVLA
                policy=cls(env,str(checkpoint),replan_steps=1,seed=20263701);policy._load();policy.model.config.num_steps=10
            policy.env=env;policy.reset(task)
            # Read-only fixture injection substitutes an already-verified
            # before-action simulator measurement, NOT expert state/action.
            inputs={'scene64':raw.copy(),'numeric67':encode_numeric(raw)}
            with patch('VLA.smolvla.scene67_adapter.model_input',return_value=inputs),patch('VLA.smolvla.target_crop.model_input',return_value=inputs):
                action=policy.predict(obs)
                if first is None:
                    first=action.copy();policy.reset(task);repeat=policy.predict(obs)
                    np.testing.assert_allclose(repeat,first,rtol=0,atol=1e-6)
            result={k:item[k] for k in ('number','task','target','stage','source_file','source_frame')}
            result.update(actual_neural_action=action.tolist(),canonical_gold_LOCAL_ONLY=item['canonical_gold_LOCAL_ONLY'],
                metrics=metrics(action,'canonical_delta_ee_7d',item['canonical_gold_LOCAL_ONLY'],None),
                maximum_abs_normalized_state=float(np.max(np.abs((inputs['numeric67']-mean)/(std+1e-8)))))
            rows.append(result);save('running_actual_neural_TRAIN_predictions')
        save('completed_actual_TRAIN20_NOT_model_SR')
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}';save('failed_or_window_ended_NOT_zero_SR');raise
    finally:
        if policy is not None:policy.close()
        cache.clear();gc.collect();torch.cuda.empty_cache()


def watch(a):
    a.output.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+a.wait_hours*3600;stage='waiting_exact_postfit_GTcrop120_inference_only';error=None
    frozen={str(p):sha256_file(p) for p in (Path(__file__),ROOT/'VLA/smolvla/scene67_adapter.py',ROOT/'VLA/smolvla/target_crop.py',
        FIXTURE/'audit.json',FIXTURE/'requests_with_LOCAL_gold.json')}
    def save(status='running'):
        atomic_json(a.output/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_files':frozen,
            'simulator_controls':0,'additional_model_ONLY_during_exact_GTcrop_inference':True})
    save()
    try:
        proof=json.loads((FIXTURE/'audit.json').read_text());items=json.loads((FIXTURE/'requests_with_LOCAL_gold.json').read_text())
        if not proof['passed'] or proof['observations']!=20 or len(items)!=20:raise ValueError('Predeclared20 TRAIN fixture required')
        if sha256_file(FIXTURE/'requests_with_LOCAL_gold.json')!=proof['request_file_sha256']:raise ValueError('Fixture changed')
        for path,digest in proof['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Actual selected TRAIN source changed')
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Finite postfit TRAIN diagnostic wait')
            if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen probe changed')
            owner=inference_owner();gpu=resources()
            if (owner and {int(p) for p in gpu['compute_pids']}=={owner} and gpu['free_mib']>=5120
                    and gpu['available_RAM_MiB']>=5120 and not render_jobs()):break
            save();time.sleep(20)
        precision=json.loads((FIT/'train_saved_precision.json').read_text());qual=json.loads((FIT/'own_GTcrop_qualification.json').read_text())
        if not precision['passed'] or precision['optimizer_steps']!=[2500.] or not qual['passed']:raise ValueError('Actual fitted/saved/exact-reload qualification required')
        cp=FIT/'train/checkpoints/010000/pretrained_model'
        import torch
        torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(3072*2**20/torch.cuda.get_device_properties(0).total_memory)
        atomic_json(a.output/'resource_admission.json',{'resources':gpu,'exact_primary_evaluator_pid':owner,
            'additional_allocator_cap_MiB':3072,'existing_trainers_or_simulator_episodes_signalled':False})
        for mode,checkpoint in (('two_raw_views_warm',WARM),('three_views_after_actual2500',cp)):
            stage=mode;save();run_probe(checkpoint,items,mode,a.output/mode,owner)
        warm=json.loads((a.output/'two_raw_views_warm/audit.json').read_text());new=json.loads((a.output/'three_views_after_actual2500/audit.json').read_text())
        atomic_json(a.output/'paired_TRAIN_action_comparison.json',{'completed':True,'observations_each':20,
            'warm_median_XYZ_error_mm':warm['median_XYZ_error_mm'],'fitted_median_XYZ_error_mm':new['median_XYZ_error_mm'],
            'warm_gripper_sign_correct':warm['gripper_sign_correct'],'fitted_gripper_sign_correct':new['gripper_sign_correct'],
            'fixed_TRAIN_fixture_NOT_generalization_or_task_SR':True,'same_flow_steps10':True,
            'not_isolated_effect_of_crop_ALSO_new2500_motion_balanced_fit':True})
        stage='completed_paired_actual_TRAIN20_NOT_SR';save('completed')
    except Exception as exc:error=f'{type(exc).__name__}: {exc}';save('failed_preserve_partial_NOT_zero_SR');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=48);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=54:raise ValueError('Finite probe only')
    if not a.launch:watch(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if marker.exists() or a.output.exists():raise FileExistsError('No duplicate probe')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.audit_smol_fit_r38','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'actual_postfit_TRAIN_probe_NOT_extra_rollout_cohort':True});print(child.pid)
