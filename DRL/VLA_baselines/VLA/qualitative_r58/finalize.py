"""Fail closed unless exact2/3/3/3 main pairs AND actual reviewed backups exist."""
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,ROOT,CP,read
from .packet_audit import main as packet_audit
from .visibility_report import main as visibility_report
from .ground_context import main as ground_context
from .timing_report import main as timing_report
from .audit_policy_maps import main as audit_policy_maps
from .camera_contract import main as camera_contract
from .progress_metadata import main as progress_metadata
from .input_ranges import main as input_ranges


def main():
    saved=read(CP/'saved_values.json')
    if (sha256_file(CP/'config.json')!=saved['config_sha256'] or
            sha256_file(CP/'trainable_masters.pt')!=saved['trainable_master_sha256']):
        raise ValueError('Frozen X-VLA checkpoint bytes changed')
    produced=read(OUT/'produced_pairs.json');selected=read(OUT/'assignments.json');findings=[];initial=[];main=[];backups=[]
    expected_keys={(r['backup'],r['scene_id'],r['seed']) for r in selected}
    actual_keys={(r['backup'],r['scene_id'],r['seed']) for r in produced}
    if expected_keys!=actual_keys or len(actual_keys)!=len(produced):
        raise ValueError('Selected/produced pair coverage differs or duplicated')
    if not read(OUT/'capture_done.json')['all11_main_CPU_selected']:
        raise ValueError('CPU qualification is incomplete')
    if read(OUT/'production_done.json')['errors']:
        raise ValueError('Unrepaired production errors remain')
    supplemental=read(OUT/'audit/supplemental_midpoint_checks.json')
    if not supplemental['passed'] or read(OUT/'supplemental_audit_v1/status.json')['status']!='complete':
        raise ValueError('Distinct-midpoint supplemental audit is incomplete')
    supplemental_by_label={r['label']:r for r in supplemental['results']}
    for row in produced:
        if not row.get('paired_output'):raise ValueError('Unrepaired production error')
        folder=Path(row['paired_output']);pair=read(folder/'PAIR_AUDIT.json');review=read(folder/'FINAL_VISUAL_REVIEW.json')
        if not review['approved_for_delivery']:raise ValueError('Actual visual approval missing')
        if any(sha256_file(folder/'review'/name)!=digest for name,digest in review['review_artifact_SHA256'].items()):
            raise ValueError('Encoded review evidence changed after actual image inspection')
        if not pair['same_measured_initial_state'] or not pair['Oracle_repeat_proof']['passed']:raise ValueError('Pair parity failed')
        if any(layout['maximum_pick_goal_AABB_occlusion']!=0 for layout in pair['common_layout'].values()):
            raise ValueError('Inset covers requested object/goal on a trajectory')
        if not row['backup']:initial.append(pair['initial_state_sha256'])
        for method,value in pair['outcomes'].items():
            audit=read(folder/method/'camera_audit.json')
            original_hashes=read(folder/method/'manifest.sha256.json')
            if any(sha256_file(folder/method/name)!=digest for name,digest in original_hashes.items()):
                raise ValueError('Encoded videos changed since whole-stream decode/audit')
            if read(folder/method/'layout.json')!=pair['common_layout']:
                raise ValueError('Methods do not share the audited common layout')
            if not audit['media_decode_sync_pass'] or not audit['actual_tick_spacing_pass']:raise ValueError('Media audit failed')
            if len(audit['streams'])!=7:raise ValueError('Seven actual streams required')
            renderer=read(folder/method/'renderer.json')
            if renderer['camera_settings_sha256']!=sha256_file(Path('env_eval/camera_editions.json')):
                raise ValueError('Camera config changed since recorded rendering')
            if max(renderer['render_replica_max_errors'].values())>1e-12:
                raise ValueError('Rendered pose/FK differs from actual recorded geometry')
            if renderer.get('NumPy_transfer_enabled'):
                matches=renderer['per_rollout_golden_RGB_and_segmentation_matches']
                if not renderer['all_per_rollout_golden_exact'] or len(matches) not in (14,21) or not all(
                        match['RGB_exact'] and match['segmentation_exact'] for match in matches):
                    raise ValueError('Alternate-renderer per-rollout golden gate missing')
                if len(matches)==14:
                    prefix='backup' if row['backup'] else 'scene';suffix='Oracle' if method=='Scripted_Oracle' else 'VLA'
                    label=f'{prefix}{row["scene_id"]:02d}_seed{row["seed"]}_{suffix}'
                    extra=supplemental_by_label[label];checks=extra['supplemental_checks']
                    if (not extra['passed'] or len(checks)!=7 or len(extra['source_frames_after_supplement'])!=3
                            or extra['original_renderer_json_sha256']!=sha256_file(folder/method/'renderer.json')
                            or extra['source_state_sha256']!=renderer['source_state_sha256']
                            or not all(c['RGB_exact'] and c['segmentation_exact'] and c['existing_encoded_RGB_MAE']<6. for c in checks)):
                        raise ValueError('Actual independent third-state verification missing')
            elif renderer.get('pybullet_binary_sha256')!='43bacc0f4d1bdae2e05698a98cb3b0fe03238127c7c606992f55593e6841e2af':
                raise ValueError('Unverified renderer binding')
            alias=Path(value['delivery_alias'])
            if sha256_file(alias)!=sha256_file(folder/method/'view_2.mp4'):raise ValueError('Principal alias differs')
            expected=(row['Oracle_full_put'] if method=='Scripted_Oracle' else row['VLA_full_put'])
            if value['full_put_complete']!=expected:raise ValueError('Outcome/slot mismatch')
            if method=='XVLA_terminal_release':
                proof=read(folder/method/'replay_proof.json')
                if sha256_file(folder/method/'replay_proof.json')!=sha256_file(Path(row['vla'])/'replay_proof.json'):
                    raise ValueError('Copied neural replay proof differs from original capture proof')
                if not (proof['passed_exact_replay'] and proof['every_after_action_full_physical_state_and_action_exact']
                        and proof['raw_observation_segmentation_and_prompt_bit_exact_vs_baseline']
                        and proof['native_approach_and_transport_actions_unmodified']
                        and proof['checkpoint'].endswith('step-014500')):
                    raise ValueError('Actual original neural action/replay provenance failed')
                tail=proof.get('terminal_release_supervisor') or {}
                for action in tail.get('controls',[]):
                    if any(value!=0 for value in action['requested_action'][:6]):raise ValueError('Undisclosed supervisor arm correction')
                for role,entry in proof['controls'][0]['actual_policy_visibility'].items():
                    box=entry['target_box']
                    if not box or min(box[0],box[1],383-box[2],383-box[3])<8:
                        raise ValueError('Initial actual policy target clipped or absent: '+role)
                if not (Path(row['paired_output'])/'EXPLANATION.md').exists():
                    raise ValueError('Scene-specific evidence and counterarguments missing')
            (backups if row['backup'] else main).append(alias)
        findings.append({'scene':row['scene_id'],'seed':row['seed'],'backup':row['backup'],'group':row['group'],
            'outcomes':{m:v['full_put_complete'] for m,v in pair['outcomes'].items()},'actual_visual_review':True})
    mainrows=[r for r in produced if not r['backup']]
    counts={g:sum(r['group']==g for r in mainrows) for g in ('01_RL_succeed_VLA_succeed','02_RL_fail_VLA_succeed','03_RL_succeed_VLA_fail','04_RL_fail_VLA_fail')}
    if list(counts.values())!=[2,3,3,3] or len(main)!=22 or len(set(initial))!=11:raise ValueError('Not exact11 distinct paired scenes/22videos2-3-3-3')
    if len(backups)<8:raise ValueError('Four actual reviewed backup pairs required')
    packet_audit()
    visibility_report()
    ground_context()
    timing_report()
    audit_policy_maps()
    policy_review=read(OUT/'audit/MANUAL_POLICY_INPUT_REVIEW.json')
    if not policy_review['actually_opened_after_generation'] or policy_review['case_count']!=len(produced):
        raise ValueError('Actual policy-image review is incomplete')
    if any(sha256_file(OUT/path)!=digest for path,digest in policy_review['PNG_files_SHA256'].items()):
        raise ValueError('Policy input review images changed after actual inspection')
    camera_contract()
    progress_metadata()
    input_ranges()
    if read(OUT/'audit/whole_stream_packet_PTS.json')['stream_count']!=len(produced)*14:
        raise ValueError('Independent PTS audit missing a delivered stream')
    source_paths=list((ROOT/'VLA/qualitative_r58').glob('*.py'))+list((ROOT/'VLA/qualitative_r57').glob('*.py'))
    source_paths += [ROOT/p for p in ('VLA/qualitative_recorder_r51.py','VLA/qualitative_oracle_shared_r51.py',
        'VLA/common/environment.py','VLA/common/body_environment.py','VLA/common/body_gripper.py',
        'VLA/common/observation_profile.py','VLA/scene64_conditioning.py',
        'env_eval/camera_editions.json','env_eval/camera_editions.py','env_eval/handeye_camera.py',
        'env_eval/oracle_cinematic.py','env_eval/oracle_feedback.py','env_eval/feedback_environment.py',
        'env_eval/feedback_motion.py','utils/camera.py')]
    checkpoint_paths=[CP/name for name in ('config.json','saved_values.json','trainable_masters.pt')]
    atomic_json(OUT/'SOURCE_AND_CHECKPOINT_MANIFEST.json',{
        'scope':'Current reproducible source/checkpoint attestation; per-child historical entrypoint SHA remains in producer source records',
        'files':{str(p.relative_to(ROOT)):sha256_file(p) for p in source_paths+checkpoint_paths},
        'new_NN_inference_NOT_claimed':True,'saved_neural_trace_hashes_in_each_replay_proof':True})
    # Include every delivered encoded camera stream, principal aliases and the
    # evidence/caption metadata. Do not include this self-referential manifest.
    groups=set(counts)|{'05_backups'}
    # Staging/raw renders include preserved incomplete rejected attempts; they
    # are evidence, NOT delivered movies. Never certify them by accident.
    paths=[p for p in OUT.rglob('*') if p.is_file()
        and (p.relative_to(OUT).parts[0] in groups|{'audit'} or len(p.relative_to(OUT).parts)==1)
        and not p.is_relative_to(OUT/'audit/goldens')
        and p.name not in ('MANIFEST.sha256.json','FINAL_AUDIT.json','report_owner.log','REPORT.md','index.html','COVERAGE.json')
        and (p.suffix in ('.mp4','.txt','.json','.jpg','.md')
             or (p.suffix=='.png' and p.is_relative_to(OUT/'audit/policy_inputs')))]
    atomic_json(OUT/'MANIFEST.sha256.json',{str(p.relative_to(OUT)):sha256_file(p) for p in paths})
    result={'passed':True,'main_scene_pairs':11,'main_principal_videos':len(main),'backup_principal_videos':len(backups),
        'main_group_pair_counts':counts,'unique_measured_main_initial_worlds':11,
        'all_main_and_backup_7camera_streams_audited':True,'all_pairs_actual_encoded_frame_reviewed':True,
        'continuous_every_frame_manual_review_NOT_claimed':True,
        'videos_and_manual_review_images_unchanged_since_decode_and_inspection':True,
        'Oracle_not_learned_RL_and_faults_disclosed':True,'VLA_terminal_release_supervisor_disclosed':True,
        'not_quantitative_SR_or_general_VLA_weakness_proof':True,'findings':findings}
    result['delivery_manifest_sha256']=sha256_file(OUT/'MANIFEST.sha256.json')
    result['delivered_camera_streams_PTS_checked']=len(produced)*14
    atomic_json(OUT/'FINAL_AUDIT.json',result);print(result,flush=True)


if __name__=='__main__':main()
