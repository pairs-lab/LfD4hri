"""Finite prepared GT-target view→native2/reload8→2500 updates→locked120.

Other registered primary VLA cohorts retain GPU priority. The target-view
fit precedes the secondary OpenVLA refinement; no neural worker is stopped.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.run_cross_vla120 import ROOT,gpu_state
from VLA.run_scene67_finetune_cycle import command as native_command,finalise
from VLA.wait_render_then_resume_queues import render_jobs
from VLA.smolvla.target_crop import SCHEMA,CAMERA,NATIVE_CAMERA
from VLA.scene64_conditioning import NUMERIC_SCHEMA

BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015';SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
WARM=ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'
PROJECTION=ROOT/'VLA/audits/scene64_recovery_r30_20261014/full_train64'
ADMISSION=BASE/'canonical_recovery_admission_lightweight_v2.json'
GATE=BASE/'target_crop_r36_native128_v2/audit.json'
COVERAGE=BASE/'target_crop_r36_all_masks/audit.json'
PARENTS=[BASE/name for name in ('all_family_privileged_r33_v4','openpi_EEF_repair_r33_v4')]


def alive(root):
    try:
        s=json.loads((root/'status.json').read_text());p=Path('/proc')/str(s['pid'])
        return (p/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except (OSError,KeyError,IndexError):return False


def fit_command(warm,fit,updates):
    cmd=list(map(str,native_command(warm,fit,updates,4)));cmd[2]='VLA.smolvla.train_target_crop'
    rename={'observation.images.front':'observation.images.camera1','observation.images.wrist':'observation.images.camera2',CAMERA:NATIVE_CAMERA}
    return [('--rename_map='+json.dumps(rename)) if c.startswith('--rename_map=') else
            '--policy.optimizer_lr=0.00005' if c.startswith('--policy.optimizer_lr=') else
            '--policy.scheduler_decay_lr=0.00000125' if c.startswith('--policy.scheduler_decay_lr=') else c for c in cmd]


def run(a):
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=False);stage='waiting_own_actual_three_view_native128';history=[];active=None
    atomic_json(BASE/'Smol_GTcrop_motion_r36_active.json',{'cycle':str(output),'pid':os.getpid(),'purpose':'latest isolated pre-fit generation; old failed CPU entries retained'})
    files=[Path(__file__),ROOT/'VLA/smolvla/target_crop.py',ROOT/'VLA/smolvla/audit_target_crop.py',
        ROOT/'VLA/smolvla/train_target_crop.py',ROOT/'VLA/evaluate_target_crop_r36.py',ROOT/'VLA/evaluate_scene67_smolvla.py',
        ROOT/'VLA/smolvla/train_scene67.py',ROOT/'VLA/smolvla/scene67_dataset.py',ROOT/'VLA/openvla/recovery_balanced.py',
        ROOT/'VLA/audit_target_mask_coverage_r36.py',ROOT/'VLA/smolvla/audit_target_crop_entry.py']
    frozen={str(p):sha256_file(p) for p in files};deadline=time.monotonic()+a.wait_hours*3600
    env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1',
        'VLA_SCENE67_PROJECTION':str(PROJECTION),'VLA_SCENE67_RECOVERY_ADMISSION':str(ADMISSION),
        'VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR':'1','VLA_SMOLVLA_INITIAL_FRAME_PROB':'.1','VLA_SMOLVLA_TARGET_BALANCED':'1',
        'VLA_SMOLVLA_GROUNDING_PROB':'0','VLA_SMOLVLA_SAMPLING_SOURCE':str(ROOT/'VLA/audits/finetune_v9_staged_2584_r13_recovery_roles_20261013/qualified_train'),
        'VLA_SMOLVLA_GPU_MEMORY_LIMIT_MIB':'12000','VLA_TARGET_CROP_GATE':str(GATE)}
    def save(status='running',error=None):
        atomic_json(output/'status.json',{'pid':os.getpid(),'status':status,'stage':stage,'error':error,'history':history,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'frozen_files':frozen,'input_variant':SCHEMA,
            'deadline_remaining_seconds':max(0,deadline-time.monotonic()),'own_fit_started':stage=='train' or any(h['stage']=='train' for h in history),
            'other_VLA_families_keep_priority':True,'same_robot_objects_scoring_safety':True,'success50percent_NOT_established':True})
    def check():
        if time.monotonic()>deadline:raise TimeoutError('Finite remaining deadline; in-flight child preserved')
        if any(sha256_file(p)!=v for p,v in frozen.items()):raise ValueError('Frozen GTcrop recipe changed')
    def execute(label,cmd,extra=None,cpu=False):
        nonlocal stage,active
        stage=label;check();save()
        child_env={**env,**(extra or {}),**({'CUDA_VISIBLE_DEVICES':''} if cpu else {})}
        with (output/(label+'.log')).open('x') as log:
            active=subprocess.Popen(list(map(str,cmd)),cwd=ROOT,env=child_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(output/'active_child.json',{'pid':active.pid,'stage':label,'command':list(map(str,cmd))})
            while active.poll() is None:save();time.sleep(15);check()
        history.append({'stage':label,'exit_code':active.returncode})
        if active.returncode:raise RuntimeError(label+' failed; actual artifacts retained')
    try:
        save()
        while not GATE.exists() or not COVERAGE.exists():
            check();save();time.sleep(15)
        gate=json.loads(GATE.read_text())
        if not gate.get('passed') or gate['schema']!=SCHEMA or gate['samples']!=128:raise ValueError('Own three-view gate did not pass')
        for path,digest in gate['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Actual three-view gate/source changed')
        coverage=json.loads(COVERAGE.read_text())
        if not coverage.get('passed') or coverage['episodes']!=2145 or coverage['counts']['frames']!=117736:
            raise ValueError('Full actual mask-source coverage required')
        for path,digest in coverage['bound_files'].items():
            if sha256_file(path)!=digest:raise ValueError('Full target-mask coverage proof changed')
        warm=output/'exact_weight_warmstart_GTcrop';warm.mkdir()
        cfg=json.loads((WARM/'config.json').read_text());cfg['input_features'][NATIVE_CAMERA]={'type':'VISUAL','shape':[3,384,384]}
        atomic_json(warm/'config.json',cfg);os.link(WARM/'model.safetensors',warm/'model.safetensors')
        atomic_json(warm/'warmstart.json',{'source':str(WARM),'model_sha256':sha256_file(WARM/'model.safetensors'),
            'all_neural_weights_byte_identical':True,'only_new_config_feature':NATIVE_CAMERA,'optimizer_reset_new_recipe':True})
        entry=output/'native_trainer_entry_preGPU.json';entry_fit=output/'entry_stub_NO_neural_model'
        execute('actual_native_trainer_entry_preGPU',[SIM,'-m','VLA.smolvla.audit_target_crop_entry',
            '--warmstart',warm,'--stub-fit',entry_fit,'--output',entry],
            {'VLA_SCENE67_UPDATES':'2','VLA_SMOLVLA_MASTER_AUDIT':str(entry_fit/'master_dtype_audit.json')},cpu=True)
        if not json.loads(entry.read_text()).get('passed'):raise ValueError('Native trainer-entry configuration failed')
        stage='ready_waiting_primary_all_VLAs_before_secondary_OpenVLA_refinement';save()
        while True:
            check();gpu=gpu_state()
            if not any(alive(p) for p in PARENTS) and not gpu['compute_pids'] and gpu['free_mib']>=14500 and not render_jobs():break
            save();time.sleep(30)
        qualification=output/'own_GTcrop_qualification.json';source=Path(env['VLA_SMOLVLA_SAMPLING_SOURCE'])
        for updates,label in ((2,'smoke'),(2500,'train')):
            fit=output/label;cmd=fit_command(warm,fit,updates)
            extra={'VLA_SCENE67_UPDATES':str(updates),'VLA_SMOLVLA_MASTER_AUDIT':str(fit/'master_dtype_audit.json')}
            if updates>2:extra['VLA_SCENE67_SMOKE_QUALIFICATION']=str(qualification)
            execute(label,cmd,extra)
            cp=finalise(fit,updates,4,ADMISSION,PROJECTION)
            contract=json.loads((cp/'scene67_contract.json').read_text());contract['schema']=SCHEMA
            atomic_json(cp/'scene67_contract.json',contract)
            atomic_json(cp/'target_crop_contract.json',{'schema':SCHEMA,'raw_numeric_schema':NUMERIC_SCHEMA,
                'privileged_current_GT_target_segmentation':True,'raw_front_wrist_preserved':True,'future_or_oracle_action_in_input':False,
                'input_bound_files':{str(p):sha256_file(p) for p in (ROOT/'VLA/smolvla/target_crop.py',ROOT/'VLA/scene64_conditioning.py')},
                'new_optimizer_updates':updates,'warmstart':str(WARM),'native_action_loss_unchanged':True,
                'sampling':'original80_task_target_motion_balanced_correction20_task_episode_balanced',
                'stationary_wait_mass_cap_per_task_target':.05,'meaningful_reopen_release_not_capped':True,'optimizer_lr':.00005})
            precision=output/(label+'_saved_precision.json')
            execute(label+'_precision',[SIM,'-m','VLA.smolvla.audit_training_precision','--checkpoint',cp,'--output',precision],cpu=True)
            q=json.loads(precision.read_text())
            if not q['passed'] or q['optimizer_steps']!=[float(updates)]:raise ValueError('Own FP32/actual-update counters failed')
            if updates==2:
                probe=output/'own_GTcrop_native8'
                execute('own_GTcrop_exact_reload8',[SIM,'-m','VLA.evaluate_target_crop_r36','--checkpoint',cp,'--output',probe,'--smoke-source',source])
                result=json.loads((probe/'progress.json').read_text())
                if result['neural_controls']!=8 or result['input_variant']!=SCHEMA:raise ValueError('Own GTcrop eight controls incomplete')
                deps={**frozen,str(GATE):sha256_file(GATE),str(precision):sha256_file(precision),str(probe/'progress.json'):sha256_file(probe/'progress.json')}
                atomic_json(qualification,{'passed':True,'target_crop_schema':SCHEMA,'actual_optimizer_updates':2,
                    'actual_neural_controls':8,'dependencies':deps,'model_SR_established':False})
        execute('GTcrop_development120',[SIM,'-m','VLA.evaluate_target_crop_r36','--checkpoint',cp,'--output',output/'development120'])
        result=json.loads((output/'development120/progress.json').read_text())
        if result['completed_trials']!=120:raise ValueError('Incomplete actual learned120')
        stage='completed_GTcrop_fit_and120_requires_final_review';save('completed')
    except Exception as exc:save('failed_preserve_artifacts_NOT_zero_SR',f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-hours',type=float,default=44);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=56:raise ValueError('Finite deadline required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate new fit')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([str(SIM),'-m','VLA.run_target_crop_r36','--output',str(a.output.resolve()),'--wait-hours',str(a.wait_hours)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'no_existing_job_interrupted':True});print(child.pid)
