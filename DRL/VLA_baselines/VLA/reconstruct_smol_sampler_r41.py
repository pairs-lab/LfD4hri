"""Reconstruct the latest fixed-seed native67 sampling prefix on CPU.

This is a deterministic reconstruction from saved counters/config and current
bound sampler implementation, NOT an independently logged per-frame history.
Earlier family training exposure is unknown here. No models or800 data load.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import torch
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.smolvla.balanced_sampling import BalancedInitialSampler,frame_weights

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'VLA/audits/scene67_deadline_r31_20261015'
CP=ROOT/'VLA/smolvla_scene67/automatic_full_gate_fit_v2/train/checkpoints/010000/pretrained_model'
FIXTURE=BASE/'native_TRAIN_action_probe_r37_v2'


def main(out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads((CP/'train_config.json').read_text())
    step_file=CP.parent/'training_state/training_step.json';saved=json.loads(step_file.read_text())
    precision_file=BASE/'automatic_full_gate_fit_v2/train_precision.json'
    precision=json.loads(precision_file.read_text())
    contract=json.loads((CP/'scene67_contract.json').read_text())
    manifest=json.loads((CP/'vla_checkpoint_manifest.json').read_text())
    if (config['seed']!=20261015 or config['resume'] or config['num_workers']!=0
        or saved!={**saved,'step':10000,'dp_world_size':1,'batch_size':2,'grad_accum_steps':4}
        or not precision['passed'] or precision['optimizer_steps']!=[2500.]):
        raise ValueError('Exact fresh single-worker10000-microstep/2500-update checkpoint required')
    source=Path(manifest['source']);index_file=source/'index.json';index=json.loads(index_file.read_text())
    admission_file=Path(manifest['correction_admission'])
    if sha256_file(admission_file)!=contract['recovery_admission_sha256']:raise ValueError('Checkpoint recovery admission changed')
    admission=json.loads(admission_file.read_text());corrective_index=Path(admission['candidate'])/'index.json'
    by_name={r['file']:r for r in json.loads(corrective_index.read_text()) if r.get('file')}
    correction=[by_name[r['file']] for r in admission['rows']]
    count_by_task=Counter(r['task']['name'] for r in correction)
    lengths=np.asarray([r['steps'] for r in index]);ends=np.cumsum(lengths);starts=np.r_[0,ends[:-1]]
    original=frame_weights(index,.1,target_balanced=True)
    extra=np.concatenate([np.full(r['steps'],1/(4*count_by_task[r['task']['name']]*r['steps']),np.float64) for r in correction])
    sampler=BalancedInitialSampler(starts,ends,source_index=index,initial_probability=.1,target_balanced=True,seed=config['seed'])
    sampler.weights=torch.from_numpy(np.r_[.8*original,.2*extra])
    if len(sampler)!=117736 or not np.isclose(sampler.weights.sum().item(),1):raise ValueError('Actual native full mixed source required')
    prefix=np.asarray(list(iter(sampler))[:saved['step']*saved['batch_size']*saved['dp_world_size']],np.int64)
    if len(prefix)!=20000:raise ValueError('Exactly20000 reconstructed presentations required')
    np.save(out/'reconstructed_epoch0_prefix20000.npy',prefix,allow_pickle=False)
    frequency=np.bincount(prefix,minlength=len(sampler));offsets={r['file']:(int(s),int(e)) for r,s,e in zip(index,starts,ends,strict=True)}
    items=json.loads((FIXTURE/'requests_with_LOCAL_gold.json').read_text())
    predictions=json.loads((BASE/'Smol_actual_TRAIN20_now_r40_v2/actual_TRAIN20/audit.json').read_text())['rows']
    rows=[]
    for item,pred in zip(items,predictions,strict=True):
        start,end=offsets[Path(item['source_file']).name];fi=item['source_frame'];absolute=start+fi
        row={k:item[k] for k in ('number','task','target','stage','source_file','source_frame')}
        row.update(reconstructed_current_observation_presentations=int(frequency[absolute]),
            reconstructed_gold_in_five_action_chunks=int(frequency[max(start,absolute-4):absolute+1].sum()),
            actual_neural_XYZ_error_mm=pred['metrics']['XYZ_error_mm'])
        rows.append(row)
    unique_original=int(np.count_nonzero(frequency[:116257]));unique_extra=int(np.count_nonzero(frequency[116257:]))
    atomic_json(out/'audit.json',{'scope':'DETERMINISTIC_saved_recipe_reconstruction_NOT_independent_logged_sample_history_OR_SR',
        'reconstructed_presentations':len(prefix),'original_presentations':int(frequency[:116257].sum()),
        'corrective_presentations':int(frequency[116257:].sum()),'unique_original_observations':unique_original,
        'unique_corrective_observations':unique_extra,'original_observation_coverage_fraction':unique_original/116257,
        'selected_TRAIN20':rows,'selected_TRAIN20_observations_with_zero_latest_round_presentations':sum(r['reconstructed_current_observation_presentations']==0 for r in rows),
        'selected_TRAIN20_gold_actions_with_zero_latest_round_chunk_presentations':sum(r['reconstructed_gold_in_five_action_chunks']==0 for r in rows),
        'prior_family_exposure_unknown_NOT_claimed_zero':True,'TRAIN_split_is_NOT_guarantee_every_frame_seen':True,
        'causal_model_failure_or_sufficient_finetuning_NOT_established':True,'GPU_models_or_simulator_controls':0,
        'frozen_live_recipe_checkpoint_dataset_modified':False,
        'bound_files':{str(p):sha256_file(p) for p in (Path(__file__).resolve(),CP/'train_config.json',CP/'scene67_contract.json',
            CP/'vla_checkpoint_manifest.json',step_file,precision_file,index_file,admission_file,corrective_index,
            ROOT/'VLA/smolvla/train_scene67.py',ROOT/'VLA/smolvla/balanced_sampling.py',
            ROOT/'VLA/third_party/lerobot/src/lerobot/scripts/lerobot_train.py')}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
