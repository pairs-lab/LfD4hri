"""CPU-only completion-history supplements for finished native VLA cohorts.

Never changes a score/protocol, launches a model, reads partial outcomes for
selection, or supplies800 data to training. Goal-region success and observed
correct-grasp history must be distinguished in paper reports.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from VLA.audit_smol_task_completion_r41 import analyze
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'


def roots():
    result={'Smol_chunk2_development120':BASE/'Smol_native_chunk2_conditional120_r41',
        'Smol_GTcrop_round1':BASE/'Smol_GTcrop_motion_r36_v6_priority/development120',
        'Smol_GTcrop_round2_if_admitted':BASE/'Smol_GTcrop_motion_r36_v7_priority_continuation/development120'}
    for model in ('openvla','rdt','cogact'):
        result[model+'_privileged120']=BASE/f'all_family_privileged_r33_v4/{model}_development120'
    for model in ('pi0','pi0_fast'):
        result[model+'_EEF_privileged120']=BASE/f'openpi_EEF_repair_r33_v4/{model}_EEF_development120'
    pointer=json.loads((BASE/'Smol_800_r39_active.json').read_text());root=Path(pointer['cycle'])
    if root.parent!=ROOT/'VLA/smolvla_scene67' or not root.name.startswith('test800_200peraction_'):
        raise ValueError('Unknown frozen800 reporting source')
    result['Smol_frozen800_reporting_ONLY']=root
    return result


def run(a):
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);deadline=time.monotonic()+a.wait_hours*3600
    states={};frozen={str(p):sha256_file(p) for p in (Path(__file__).resolve(),ROOT/'VLA/audit_smol_task_completion_r41.py')}
    def save(status='running'):
        atomic_json(out/'status.json',{'pid':os.getpid(),'status':status,'stage':'read_only_completed_cohort_reporting',
            'models':states,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'no_GPU_or_NN_or_training_or_robot_controls':True,'no800_data_for_fitting_or_model_selection':True,
            'primary_success_conditions_and_results_unchanged':True,'frozen_files':frozen})
    save()
    while time.monotonic()<deadline:
        if any(sha256_file(p)!=h for p,h in frozen.items()):raise ValueError('Frozen reporting code changed')
        for name,root in roots().items():
            if states.get(name,{}).get('status') in ('completed_supplement','failed_diagnostic_NOT_model_failure'):continue
            file=root/'progress.json'
            if not file.exists():states[name]={'status':'pending_cohort_NOT_zero_SR'};continue
            data=json.loads(file.read_text())
            count=data.get('completed_trials',0);planned=data.get('planned_trials')
            if not planned or count!=planned:
                states[name]={'status':'pending_complete_cohort_NOT_zero_SR','completed':count,'planned':planned};continue
            try:
                analyze(file,out/name)
                result=json.loads((out/name/'audit.json').read_text())
                states[name]={'status':'completed_supplement','trials':count,'by_task':result['by_task'],
                    'audit':str(out/name/'audit.json'),'does_NOT_relabel_primary_scores':True}
            except Exception as exc:
                states[name]={'status':'failed_diagnostic_NOT_model_failure','error':f'{type(exc).__name__}: {exc}'}
            save()
        save();time.sleep(30)
    save('reporting_deadline_reached_pending_are_NOT_zero_SR')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-hours',type=float,default=48);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not 0<a.wait_hours<=54:raise ValueError('Finite read-only reporting watcher required')
    if not a.launch:run(a)
    else:
        marker=a.output.with_name(a.output.name+'_launch.json')
        if a.output.exists() or marker.exists():raise FileExistsError('No duplicate reporter')
        with a.output.with_name(a.output.name+'_launcher.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','VLA.watch_task_completion_r41','--output',str(a.output.resolve()),
                '--wait-hours',str(a.wait_hours)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic_json(marker,{'pid':child.pid,'read_only_reporting_NOT_extra_model_evaluation':True});print(child.pid)
