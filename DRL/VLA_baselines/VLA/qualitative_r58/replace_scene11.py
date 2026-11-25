"""Publish a measured replacement after rejecting clipped policy input.

Requires actual CPU evidence and separately inspected visual previews. It
does not create or alter robot actions, checkpoint or simulation settings.
"""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import OUT,read
from .selection import assess


def main(note):
    root=OUT/'_captures/put_seed20268107'
    cap=root/'XVLA_release_r58';oracle=root/'Oracle_drop_in_transit_capture'
    repeat=root/'Oracle_drop_in_transit_repeat'
    ve=read(cap/'content_evidence.json');proof=read(cap/'replay_proof.json')
    oe=read(oracle/'content_evidence.json');strength=assess(ve,proof)
    if not strength['strong_failure_candidate'] or ve['full_put_complete'] or oe['full_put_complete']:
        raise ValueError('Replacement is not an actual strong fail/fail pair')
    if not proof['passed_exact_replay'] or proof['safety_violation']:
        raise ValueError('Exact native replay or safety failed')
    initial=read(cap/'initial_scene.json')['measured_state_sha256']
    if initial!=read(oracle/'initial_scene.json')['measured_state_sha256']:
        raise ValueError('Paired initial physics differs')
    names=['initial_scene.json','recorded_physics_states.json','physics_event_ticks.json','outcome.json','trajectory.json']
    equal={n:sha256_file(oracle/n)==sha256_file(repeat/n) for n in names}
    if not all(equal.values()):raise ValueError('Oracle exact independent repeat failed')
    visibility={}
    for role,entry in proof['controls'][0]['actual_policy_visibility'].items():
        box=entry['target_box']
        if not box:raise ValueError('Target absent in actual initial policy image')
        margin=min(box[0],box[1],383-box[2],383-box[3])
        if margin<8:raise ValueError('Replacement has clipped/edge target')
        visibility[role]={**entry,'margin_px':margin,'touches_image_boundary':False}
    repeat_path=root/'repeat_drop_in_transit.json'
    atomic_json(repeat_path,{'passed':True,'exact_files':equal,'target':str(oracle),'repeat':str(repeat)})
    assignment={'scene_id':11,'backup':False,'seed':20268107,'group':'04_RL_fail_VLA_fail',
        'vla':str(cap),'oracle':str(oracle),'repeat_proof':str(repeat_path),
        'oracle_fault':'drop_in_transit','VLA_full_put':False,'Oracle_full_put':False,
        'status':'CPU_selected_NEEDS_video_and_visual_review'}
    assignments=read(OUT/'assignments.json')
    if any(r['scene_id']==11 and not r['backup'] for r in assignments):
        raise ValueError('Do not overwrite an already selected main scene11')
    if any(r['seed']==20268107 for r in assignments):raise ValueError('Repeated initial scene')
    atomic_json(OUT/'audit/replacement_scene11_camera_and_content.json',{
        'rejected_seed':20268015,'replacement_seed':20268107,
        'reason':'Banana initially clipped in actual policy front image; no pure-model-failure claim',
        'actual_reset_front_and_wrist_images_opened':True,'actual_rendered_event_images_opened':True,
        'actual_preview_findings':note,'initial_policy_visibility':visibility,
        'VLA_acquisition_evidence':ve,'Oracle_drop_evidence':oe,'selection':strength,
        'no_actions_physics_or_success_thresholds_changed':True})
    assignments.append(assignment)
    atomic_json(OUT/'assignments.json',assignments)
    ledger=read(OUT/'candidate_ledger.json')
    ledger.append({'seed':20268107,'slot_requested':11,'status':'selected_replacement_main_CPU',
        'replacement_for_policy_camera_clipped_seed':20268015,'strength':strength,**assignment})
    atomic_json(OUT/'candidate_ledger.json',ledger)
    atomic_json(OUT/'capture_done.json',{'all11_main_CPU_selected':True,'missing_slots':[],
        'backup_pairs':sum(r['backup'] for r in assignments),
        'camera_clipped_candidate_rejected_not_hidden':True,'replacement_seed':20268107})
    print({'scene11_replaced':True,'initial_policy_visibility':visibility},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preview-note',required=True)
    main(parser.parse_args().preview_note)
