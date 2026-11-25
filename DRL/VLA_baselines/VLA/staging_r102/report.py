"""Read-only live readiness index; resolves the actual trainer heartbeat.

Does not run GPU inference, select checkpoints, launch models or alter results.
"""
import argparse,time
from pathlib import Path
from .common import OUT,R100,read,utc,atomic_json
from VLA.continuous_r86.owner import identity,alive


def optional(path):return read(path) if path.exists() else {}


def render():
    owner=optional(R100/'WORKFLOW_STATUS.json');records={}
    for family in ('seer','gr00t_n15','falcon_esm'):
        root=R100/family;admission=optional(root/'NATIVE_ADMISSION_READY.json')
        fixture=optional(root/'TRAINER_CODE_ROUNDTRIP.json');data=optional(root/'DATA_AUDIT.json')
        state=optional(root/'WORKFLOW_STATUS.json');final=optional(root/'CERTIFIED_RESULT.json')
        phase=state.get('stage','');train=optional(root/phase/'status.json')
        live=identity(train.get('pid')) if train.get('pid') and alive(train['pid']) else None
        cmd=live.get('command',[]) if live else []
        live_training=bool('-m' in cmd and cmd[cmd.index('-m')+1]=='VLA.admission_r100.train'
            and '--family' in cmd and cmd[cmd.index('--family')+1]==family)
        heartbeat=None
        if live_training:
            heartbeat=max(0.,time.time()-(root/phase/'status.json').stat().st_mtime)
        records[family]={'native_data_passed':bool(data.get('passed')),
            'actual_full_model_and_physical4_passed':bool(admission.get('passed')),
            'actual_batch16_optimizer_resume_reload_passed':bool(fixture.get('passed')),
            'phase_start_snapshot_UTC':state.get('UTC'),'phase':phase,
            'live_training':live_training,'actual_trainer_status':train if live_training else None,
            'actual_trainer_heartbeat_age_s':heartbeat,
            'certified_actual_episodes':final.get('actual_trials',0),
            'per_action_final':final.get('per_action'),
            'performance_readiness_NOT_implied_by_engineering_checks':not bool(final),
            'technical_error':optional(root/'WORKFLOW_ERROR.json')}
    component=optional(OUT/'ESM_STATUS.json');depth=optional(OUT/'DEPTH_SCAN_STATUS.json')
    geometry=optional(OUT/'ALL_FRAME_GEOMETRY.json')
    qualification=optional(OUT/'VALIDATION.json')
    result={'UTC':utc(),'owner_phase_snapshot':owner,'families':records,
        'ESM_component':component,'exhaustive_depth_scan':depth,'all_frame_geometry':geometry,
        'staging_runner':optional(OUT/'RUNNER_STATUS.json'),'qualification':qualification,
        'existing_R100_is_sole_finetune_and_performance_executor':True,
        'zero_new_benchmark_trials_from_staging':True}
    OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'LIVE_READINESS.json',result)
    lines=['# R102: live pending-model readiness','',f'UTC: {result["UTC"]}','',
        '|Model|Native data|Full native/reload/physical4|Batch16 resume/reload|Current work|',
        '|---|---|---|---|---|']
    for family,record in records.items():
        status=record['phase']
        if record['live_training']:
            t=record['actual_trainer_status'];status=f"training {t['updates']}/{t['target_updates']}; heartbeat {record['actual_trainer_heartbeat_age_s']:.1f}s"
        elif record['certified_actual_episodes']==800:status='actual800 certified'
        elif family!=owner.get('family'):status='queued; long fit not yet started'
        lines.append('|'+ '|'.join([family,*['PASS' if record[k] else 'pending' for k in (
            'native_data_passed','actual_full_model_and_physical4_passed','actual_batch16_optimizer_resume_reload_passed')],status])+'|')
    lines+=['','## Additional staging','',
        f"CPU regression tests: {qualification.get('tests_passed','pending')}.",
        f"ESM component: {component.get('status','pending')}; cases {component.get('completed_cases',component.get('cases',0))}/228.",
        f"Full TRAIN depth scan: {depth.get('status','pending')}; {depth.get('rows',0)}/2337 rows, {depth.get('frames',0)}/125527 frames.",'',
        f"All-frame camera matrices: {'PASS' if geometry.get('passed') else 'pending'}; {geometry.get('frames',0)}/125527 frames.",'',
        'The ESM job loads authentic native ESM tensors and the real RGB-D processor, but is a COMPONENT check, not another VLA policy or new SR. '
        'The exhaustive scan reuses old physics-pairing proofs; it does not claim new demonstrations or object-visibility proof.','',
        'The phase-start scheduler JSON can be older while a guarded trainer runs. This index reads the actual trainer PID/module/family and fresh update heartbeat instead.', '',
        'The sole performance pipeline remains finetune → fresh checkpoint reload → DEV40 → bounded continuation → CONF40 → frozen800 if ready. '
        'Original metric is NOT strict1cm; PUT release/settle is not required. No five-seed measurements are invented.', '',
        'No model/physics/success source, checkpoint selection, existing outcome file, installation or user edit is modified by this observer.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch-owner-pid',type=int);a=parser.parse_args()
    expected=identity(a.watch_owner_pid) if a.watch_owner_pid else None
    while True:
        render()
        if not expected or not alive(expected['pid']) or identity(expected['pid'])!=expected:break
        time.sleep(30)
