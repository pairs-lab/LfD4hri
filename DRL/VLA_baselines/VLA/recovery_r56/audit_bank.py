"""Read-only full numeric-label audit plus a fresh geometry-exclusion check."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import ROOT, DATA, read, verify


def main(a):
    from VLA.new_vlas.dataset import NativeData
    data = NativeData(DATA, verify_rgb_hashes=False)
    tasks, kinds, recoveries = Counter(), Counter(), Counter()
    motion = Counter()
    maximum = {'adapter_XYZRPY_label_difference':0., 'adapter_open01_difference':0.}
    labels=[]
    for ei, row in enumerate(data.rows):
        numeric=data.numeric(ei)
        raw, canonical, adapter=(numeric[k] for k in ('scene64','canonical7','adapter_action7'))
        n=row['frames']
        if raw.shape!=(n,64) or canonical.shape!=(n,7) or adapter.shape!=(n,7):
            raise ValueError('Full numeric array shape mismatch')
        if not all(np.isfinite(value).all() for value in numeric.values()):
            raise ValueError('Nonfinite qualified numeric value')
        maximum['adapter_XYZRPY_label_difference']=max(maximum['adapter_XYZRPY_label_difference'],float(abs(adapter[:,:6]-canonical[:,:6]).max()))
        maximum['adapter_open01_difference']=max(maximum['adapter_open01_difference'],float(abs(adapter[:,6]-(1-canonical[:,6])/2).max()))
        tasks[row['task']]+=n;kinds[row['source_kind']]+=n
        recoveries[row.get('recovery_kind') or 'original_or_existing']+=n
        lateral=np.linalg.norm(canonical[:,:2],axis=1)>.005
        motion['XY_over5mm']+=int(lateral.sum())
        motion['quiet_XYZ_le1mm']+=int((np.linalg.norm(canonical[:,:3],axis=1)<=.001).sum())
        motion['roll_pitch_correction']+=int((abs(canonical[:,3:5]).max(axis=1)>1e-5).sum())
        motion['closed_empty_states']+=int(((raw[:,6]>.8)&(raw[:,31]<.5)).sum())
        motion['wrong_attached_states']+=int((raw[:,30]>.5).sum())
        motion['correct_verified_hold_states']+=int((raw[:,17]>.5).sum())
        motion['gripper_request_transitions']+=int(((canonical[:,6]>.5)!=(raw[:,10]>.5)).sum())
        labels.append(adapter)
    if max(maximum.values())>1e-7:
        raise ValueError('Native Adapter action channels differ from canonical convention')
    # Validate frozen sources, not the old final results/scores.
    frozen_cp=ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/xvla_round1/fit1500/checkpoints/step-014500'
    cfg=read(frozen_cp/'config.json');verify(cfg['code'])
    from VLA.run_feedback_refinement_r52 import qualify_final_seeds
    geometry=qualify_final_seeds(list(range(20269601,20269631)))
    old=set()
    for path, key in ((ROOT/'VLA/audits/autocycle_r55_20261017/automatic_v1/geometry_exclusion.json','final'),
        (ROOT/'VLA/audits/privileged_visual_r53_20261016/automatic_ablation_v2/final_geometry_exclusion.json',None),
        (ROOT/'VLA/audits/feedback_improvement_r52_20261016/automatic_refinement_v4/final_cohort_exclusion_audit.json',None)):
        value=read(path);value=value[key] if key else value
        old.update(r['physical_scene_sha256'] for r in value['cases'])
    if old & {r['physical_scene_sha256'] for r in geometry['cases']}:
        raise ValueError('R56 final geometry overlaps a prior final')
    values=np.concatenate(labels)
    result={'passed':True,'scope':'read_only_full_numeric_TRAIN_audit_NOT_RGB_requalification_or_neural_SR',
        'data':str(DATA),'data_audit_sha256':sha256_file(DATA/'audit.json'),
        'episodes':len(data.rows),'frames':int(sum(tasks.values())),
        'frames_by_task':dict(tasks),'frames_by_source_kind':dict(kinds),
        'frames_by_recovery_kind':dict(recoveries),'motion_and_feedback_counts':dict(motion),
        'native_action_label_errors':maximum,
        'empirical_delta_XY_median_m':np.median(values[:,:2],axis=0).tolist(),
        'all_numeric_cache_hashes_verified':True,'all_original_RGB_files_rehashed_this_audit':False,
        'sampled_RGB_and_labels_rehashed_by_actual_fitter':True,
        'frozen_XVLA_bound_sources_unchanged':True,
        'new_final_geometry_exclusion':geometry,
        'known_remaining_limits':['wrist self/target occlusion during close interaction',
            '40 new recovery trajectories cover20 TRAIN layouts, not40 independent layouts',
            'no wrong-held/drop recovery examples in this qualified bank if corresponding counts are zero',
            'finite collection and training do NOT prove absence of distribution shift']}
    atomic_json(a.output,result)
    print({k:result[k] for k in ('passed','episodes','frames','motion_and_feedback_counts','native_action_label_errors')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
