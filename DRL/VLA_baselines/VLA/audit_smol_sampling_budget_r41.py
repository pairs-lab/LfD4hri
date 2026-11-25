"""Forecast unique exposure under the exact queued Smol sampling contract.

Pure CPU metadata calculation, NOT actual optimizer/model readiness proof.
Counts use current TRAIN-only native entry receipts. No benchmark data or
new observations are used; replacement sampling cannot create new scenes.
"""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
CONTRACT=BASE/'Smol_GTcrop_motion_r36_v6_priority/entry_stub_NO_neural_model/actual_target_crop_sampling.json'
ADMISSION=BASE/'canonical_recovery_admission_lightweight_v2.json'


def unique_expectation(count,probability,draws):
    if count<1 or not 0<probability<=1 or draws<1:
        raise ValueError('Valid frame count/probability/draw budget required')
    return float(count*(-np.expm1(draws*np.log1p(-probability)))) if probability<1 else float(count)


def original_groups(contract):
    groups=defaultdict(list)
    for row in contract['source_strata']:
        groups[row['task'],row['target']].append(row)
    if len(groups)!=76 or Counter(t for t,_ in groups)!={t:19 for t in ('touch','pick','move','pick_place')}:
        raise ValueError('All4x19 original task/target groups required')
    wait_receipts={(r['task'],r['target']):r for r in contract['stationary_mass_guard']['rows']}
    result=[]
    for key,rows in groups.items():
        receipt=wait_receipts[key];total=receipt['task_target_mass']
        old_wait=receipt['before_wait_mass'];new_wait=receipt['after_wait_mass']
        for row in rows:
            wait=row['stratum'] in ('open_wait','held_wait')
            factor=new_wait/old_wait if wait else (total-new_wait)/(total-old_wait)
            mass=total/len(rows)*factor
            result.append({**row,'mass':mass,'probability_per_frame':mass/row['frames']})
    if not np.isclose(sum(r['mass'] for r in result),.8):raise ValueError('Original80% mass changed')
    return result


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    contract=json.loads(CONTRACT.read_text());admission=json.loads(ADMISSION.read_text())
    if not admission['passed'] or admission['frames']!=1479:raise ValueError('Exact38 correction admission required')
    index_file=Path(admission['candidate'])/'index.json'
    index={r['file']:r for r in json.loads(index_file.read_text()) if r.get('file')}
    corrected=[index[r['file']] for r in admission['rows']]
    per_task=Counter(r['task']['name'] for r in corrected)
    if len(corrected)!=38 or len(per_task)!=4:raise ValueError('All38 corrections required')
    original=original_groups(contract)
    correction=[{'task':r['task']['name'],'target':r['task']['pick_name'],'file':r['file'],'frames':r['steps'],
        'probability_per_frame':.2/(4*per_task[r['task']['name']]*r['steps'])} for r in corrected]
    if sum(r['frames'] for r in original)!=116257 or sum(r['frames'] for r in correction)!=1479:
        raise ValueError('Actual qualified frame counts changed')
    if not np.isclose(sum(r['probability_per_frame']*r['frames'] for r in correction),.2):
        raise ValueError('Native Smol correction task/EPISODE mass changed')
    draw_budget=2500*4*2
    by_stratum=defaultdict(lambda:dict(frames=0,task_target_groups=0,mass=0.,expected_presentations=0.,expected_unique_frames=0.))
    for r in original:
        group=by_stratum[r['stratum']];group['frames']+=r['frames'];group['task_target_groups']+=1
        group['mass']+=r['mass'];group['expected_presentations']+=draw_budget*r['mass']
        group['expected_unique_frames']+=unique_expectation(r['frames'],r['probability_per_frame'],draw_budget)
    expected_original=sum(r['expected_unique_frames'] for r in by_stratum.values())
    expected_correction=sum(unique_expectation(r['frames'],r['probability_per_frame'],draw_budget) for r in correction)
    atomic_json(out/'audit.json',{'scope':'TRAIN_metadata_sampling_expectation_NOT_actual_seen_frames_OR_model_SR',
        'passed_mass_reconstruction':True,'original_frames':116257,'corrective_frames':1479,
        'optimizer_updates_per_round':2500,'batch_size':2,'gradient_accumulation':4,'presentations_per_round':draw_budget,
        'expected_original_presentations':16000,'expected_correction_presentations':4000,
        'expected_unique_original_frames':expected_original,'expected_unique_corrective_frames':expected_correction,
        'expected_unique_original_fraction':expected_original/116257,
        'original_strata':dict(by_stratum),'correction_task_episode_counts':dict(per_task),
        'no_wrong_held_original_stratum':not any(r['stratum']=='release_wrong_held_body' for r in original),
        'corrective_wrong_held_support_NOT_inferred_from_metadata':True,
        'near_target_reopen_definition':'OPEN, no current verified hold; encoder>5% or previous_close>50%; NOT necessarily fully-closed missed grasp',
        'conditional_round2_same_fixed_sampler_seed_repeats_the_same_ordered_draws_if_recipe_unchanged':True,
        'repeat_optimization_can_still_help_but_NOT_new_unique_data_exposure':True,
        'frozen_pending_recipe_hot_edited':False,'training_updates_or_simulator_controls':0,
        'bound_files':{str(p):sha256_file(p) for p in (Path(__file__).resolve(),CONTRACT,ADMISSION,index_file,
            ROOT/'VLA/openvla/recovery_balanced.py',ROOT/'VLA/smolvla/train_scene67.py',
            ROOT/'VLA/smolvla/balanced_sampling.py',ROOT/'VLA/run_scene67_finetune_cycle.py')}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
