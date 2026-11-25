"""Replay the completed S512 NumPy sampler, without training or changing it."""
from collections import Counter
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.replacement_r84.common import ROOT,read,verify
from VLA.replacement_r84.data import Data


if __name__=='__main__':
    root=ROOT/'VLA/audits/replacement_r84_20261020/vlanext_s'
    meta=read(root/'fit512/TRAINING.json');gate=read(root/'native_gate4_adam8/TRAINING.json')
    verify(meta['code'])
    if gate['updates']!=4 or meta['updates']!=512:raise ValueError('This replay is only the actual4->512 stage')
    # Gate4 cycles fixed anchors and never consumes NumPy sampling RNG. Actual
    # fit512 resumes that unchanged202684 NumPy stream; RGB augmentation uses
    # Torch RNG, separate from this task/target/source/frame sampler.
    rng=np.random.default_rng(202684);data=Data();counts=Counter();modes=Counter();sources=Counter()
    unique=set();targets=Counter();frames0=0
    for _ in range((meta['updates']-gate['updates'])*meta['effective_batch']):
        ei,fi=data.draw(rng);row=data.rows[ei]
        counts[row['task']]+=1;targets[row['task']+'/'+row['pick_name']]+=1
        sources[row['source_kind']]+=1;modes[data.last_sampling_mode]+=1
        unique.add((ei,fi));frames0+=int(fi==0)
    if dict(counts)!=meta['counts']:
        raise ValueError('Replayed sampler does not match actual stage counts; do not claim measured exposure')
    atomic_json(ROOT/'VLA/audits/continuous_r86_20261021/S512_EXPOSURE_REPLAY.json',{
        'passed_actual_stage_count_match':True,'actual_training_checkpoint':str(root/'fit512'),
        'replayed_actual_training_draws':sum(counts.values()),'per_action':dict(counts),
        'unique_RGB_timestamps':len(unique),'unique_source_trajectories':len({ei for ei,fi in unique}),
        'fraction_of119495_RGB_timestamps_seen_this_stage':len(unique)/119495,
        'sampled_frame0':frames0,'target_strata':len(targets),'source_kinds':dict(sources),'sampling_modes':dict(modes),
        'same_bank_still_has2185_trajectories119495_frames':True,
        'history_states_and_future_label_windows_not_counted_as_unique_RGB':True,
        'not_a_proof_data_or_training_is_sufficient':True,
        'no_new_demonstrations_or_optimizer_updates':True})
