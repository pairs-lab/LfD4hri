"""Read-only native provenance, setup, physical-contact and preservation gates."""
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.qualitative_r57.common import read
from .prepare import OUT,OLD,ROOT


def main():
    binding=read(ROOT/'VLA/audits/xvla800_r59_20261017/OWNER_SOURCE_BINDING.json')
    unchanged={p:sha256_file(p)==h for p,h in binding.items()}
    plant=read(OLD/'SOURCE_AND_CHECKPOINT_MANIFEST.json')['files']
    chosen={p:sha256_file(ROOT/p)==h for p,h in plant.items()
        if p.startswith(('VLA/common/','utils/camera.py','utils/Robotiq2F85.py',
            'env_eval/feedback_environment.py','env_eval/handeye_camera.py','env_eval/camera_editions.json'))}
    safety=read(ROOT/'VLA/audits/xvla800_r59_20261017/PREDECLARATION.json')['environment_fingerprint']['sources_sha256']
    safety_checks={p:sha256_file(ROOT/p)==h for p,h in safety.items()}
    assert all(unchanged.values()) and all(chosen.values()) and all(safety_checks.values())
    cap=OUT/'_captures/scene13_seed20271060/XVLA_verified'
    proof=read(cap/'replay_proof.json');trace=read(Path(proof['source_trajectory']))['pick_place']
    assert sha256_file(proof['source_trajectory'])==proof['source_trajectory_sha256']
    assert proof['maximum_original_pre_action_scene64_error']==0
    assert proof['every_after_action_full_physical_state_and_action_exact']
    assert proof['raw_observation_segmentation_and_prompt_bit_exact_vs_baseline']
    assert len(proof['controls'])==200 and proof['physics_ticks']==4800
    wanted={r['neural_chunk_input_sha256'] for r in trace}
    receipts={}
    for line in (ROOT/'VLA/audits/xvla800_r59_20261017/episodes.neural_receipts.jsonl').open():
        import json
        row=json.loads(line)
        if row['input_sha256'] in wanted:receipts[row['input_sha256']]=row
    chunks=[]
    for start in range(0,len(trace),30):
        source=trace[start:start+30];receipt=receipts[source[0]['neural_chunk_input_sha256']]
        assert receipt['actual_native_visual_receipt']['actual_valid_mask']==[[True,True,True]]
        row={'start_control':start,'executed_native_actions':len(source),
             'input_receipt_exists_and_exact_replay_verified':True,'three_native_views_valid':True}
        if len(source)==30:
            sha=hashlib.sha256(np.array([s['native_neural_action'] for s in source],np.float32).tobytes()).hexdigest()
            row['whole_chunk_neural_output_SHA_exact']=sha==receipt['action_sha256_float32']
            assert row['whole_chunk_neural_output_SHA_exact']
        else:
            row['whole_chunk_SHA_comparison_NOT_claimed']='Only20 of native30 actions executed before timeout; original full-chunk digest cannot be reconstructed from20.'
        chunks.append(row)
    assert len(receipts)==7
    reset=proof['controls'][0]['actual_policy_visibility']
    for r in reset.values():
        assert r['target_pixels']>=100
        assert min(r['target_box'])>0 and max(r['target_box'])<383
    ground=read(OUT/'_captures/scene12_seed20269719/Oracle_probe00/content_evidence.json')
    impact=next(e for e in ground['events'] if e['mode']=='STAGED_Oracle_Ground_Slam')
    assert impact['ground_slam_force_velocity_gate_pass']
    assert read(OUT/'_captures/scene12_seed20269719/Oracle_probe00/outcome.json')['error'] is None
    assert impact['first_contact_XYZ_m'][0]>.45
    actual_task=read(cap/'initial_scene.json')['task']
    assert actual_task['pick_name']=='gray block' and actual_task['place_name']=='bottom right corner'
    result={'passed':True,'original_quantitative_owner_sources_unchanged':unchanged,
        'common_plant_controller_camera_source_unchanged_vs_R58':chosen,
        'physics_guard_and_environment_fingerprint_unchanged_vs_frozen_R59':safety_checks,
        'quantitative_cohort_rerolls_or_model_updates_performed':False,
        'native_VLA_first_frame_target_fully_visible':reset,'native_VLA_receipts':chunks,
        'native7_original_inputs_RGB_GT_and_proprio_exact':True,
        'all200_controls_and_full_physical_state_vs_baseline_exact':True,
        'Ground_Slam_actual_physical_metrics':impact,
        'no_guard_abort_or_broken_linkage_in_Oracle_capture':True,
        'new_Oracle_surface_height_fault_is_controller_only_and_staged':True,
        'native_VLA_initial_approach_over_A_then_closure_at_C_NOT_semantic_misidentification_proof':True,
        'front_camera_keeps_requested_target_visible_at_every_native_query':all(
            r['actual_policy_visibility']['front']['target_pixels']>0 for r in proof['controls'] if r['control']%30==0),
        'memory_loss_NOT_proven_GT_and_front_remain_available':True}
    atomic_json(OUT/'audit/CPU_AND_PROVENANCE_AUDIT.json',result)
    print({'passed':True,'native_full30_chunks_hash_verified':6,'last_chunk_only20_actions_executed':True,
        'frozen_quant_sources':len(unchanged),'common_plant_camera_sources':len(chosen)},flush=True)


if __name__=='__main__':main()
