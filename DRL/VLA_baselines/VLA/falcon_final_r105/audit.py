"""Fresh all-frame numeric audit; no final observations become training data."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import OUT, R100, R104, PARENT, TASKS, FINAL, DEV, CONF, SMOKE, read, atomic_json, utc, verify_cohorts
from VLA.common.provenance import sha256_file


def main():
    from VLA.admission_r100.train import qualified
    from VLA.admission_r100.falcon_native import Data
    from VLA.falcon_esm_r91.native import normalize, denormalize
    from VLA.scene64_conditioning import encode_numeric
    from VLA.continuation_review_r101 import trace_diagnosis
    parent = qualified(PARENT, 'falcon_esm')
    if parent['updates'] != 2050:
        raise ValueError('Wrong immutable parent checkpoint')
    certificate = read(R104/'falcon_esm/CERTIFIED_RESULT.json')
    old = Path(certificate['source'])
    if not certificate['audit_passed'] or sha256_file(old) != certificate['source_sha256']:
        raise ValueError('Original800 ledger integrity changed')
    progress = read(old)
    verify_cohorts({r['seed'] for r in progress['rows']})
    data = Data()
    counts = Counter(); targets = Counter(); flags = defaultdict(Counter)
    clips = np.zeros(67, np.int64); saturation = np.zeros(6, np.int64)
    sources = {}; cases = []; frames = 0
    stats = parent['stats']; mean = np.asarray(stats['scene67']['mean'], np.float32)
    std = np.asarray(stats['scene67']['std'], np.float32)
    if np.any(std <= 0):
        raise ValueError('Invalid frozen numeric standard deviations')
    low, high = (np.asarray(stats['actions'][k], np.float32)[:6] for k in ('q01','q99'))
    for ei, row in enumerate(data.rows):
        n = data.numeric(ei); raw = n['scene64']; action = n['canonical7']
        if raw.shape != (row['frames'],64) or action.shape != (len(raw),7):
            raise ValueError('Numeric chronology/shape mismatch')
        if not np.isfinite(action).all() or not np.isin(action[:,6],[-1.,1.]).all():
            raise ValueError('Nonfinite action or non-binary canonical grip needs diagnosis')
        encoded = np.stack([encode_numeric(r) for r in raw])
        clips += (np.abs((encoded-mean)/std)>5).sum(0)
        saturation += ((action[:,:6]<low)|(action[:,:6]>high)).sum(0)
        normalized = normalize(action, stats); decoded = denormalize(normalized, stats)
        np.testing.assert_array_equal(decoded[:,6],action[:,6])
        np.testing.assert_allclose(decoded[:,:6],np.clip(action[:,:6],low,high),atol=1e-7,rtol=0)
        task = row['task']; counts[task] += len(raw); targets[task+'/'+row['pick_name']] += len(raw)
        empty = raw[:,31] < .5
        events = {'closing_empty':empty & (action[:,6]>.2),
            'contact_empty':empty & (raw[:,28:30]>.5).any(1),
            'bilateral_correct_hold':raw[:,17]>.5,
            'lateral':np.linalg.norm(action[:,:2],axis=1)>.003,
            'rotation':np.max(np.abs(action[:,3:6]),axis=1)>.01}
        for key, mask in events.items(): flags[task][key] += int(mask.sum())
        # Four real native-input snapshots per action. Labels only select
        # audit times; no phase or future action is added to policy input.
        if row['source_kind']=='complete_original' and not any(c['task']==task for c in cases):
            stages = {'reset':0, 'approach':len(raw)//3, 'end':len(raw)-1}
            close = np.flatnonzero(events['closing_empty'])
            held = np.flatnonzero(events['bilateral_correct_hold'])
            stages['closure_or_contact'] = int(close[0]) if len(close) else len(raw)//2
            stages['hold_or_end'] = int(held[0]) if len(held) else len(raw)-1
            for stage,fi in stages.items():
                cases.append({'task':task,'episode':ei,'frame':fi,'stage':stage,
                    'instruction':row['instruction'],'source':row['source'],
                    'source_sha256':row['source_sha256']})
        cache = data.root/row['cache']
        sources[str(cache)] = row['cache_sha256']
        frames += len(raw)
        if (ei+1)%200==0:
            atomic_json(OUT/'NUMERIC_AUDIT_STATUS.json',{'UTC':utc(),'episodes':ei+1,'frames':frames})
            print({'numeric_episodes':ei+1,'frames':frames},flush=True)
    if len(data.rows)!=2337 or frames!=125527 or len(targets)!=76:
        raise ValueError('Full qualified TRAIN task/target coverage changed')
    if any(not flags[t]['closing_empty'] or not flags[t]['bilateral_correct_hold'] for t in TASKS[1:]):
        raise ValueError('Missing original grasp/contact learning support')
    inherited = {}
    for name in ('FULL_DEPTH_SCAN.json','ALL_FRAME_GEOMETRY.json','ESM_COMPONENT_PROOF.json'):
        p = ROOT_STAGING/name; proof = read(p)
        if not proof['passed']:
            raise ValueError('Prior full-depth/geometry/ESM gate not passed')
        inherited[str(p)] = {'sha256':sha256_file(p),'proof_reused_NOT_new_physics_replay':True}
    depth_index = R100/'falcon_esm/DEPTH_INDEX.json'
    if read(ROOT_STAGING/'FULL_DEPTH_SCAN.json')['index_sha256'] != sha256_file(depth_index):
        raise ValueError('Full-depth scan does not describe the consumed depth bank')
    result = {'passed':True,'UTC':utc(),'trajectories':len(data.rows),'frames':frames,
        'unique_scene_groups':len({r['scene_group'] for r in data.rows}),
        'task_target_strata':len(targets),'per_action_frames':dict(counts),
        'all_numeric_frames_rechecked_with_actual_causal_encoder':True,
        'per_action_measured_event_frames':{k:dict(v) for k,v in flags.items()},
        'canonical_gripper_roundtrip_all_frames_exact':True,
        'canonical_arm_roundtrip_matches_declared_quantile_clipping':True,
        'scene67_clip5_counts_per_feature':clips.tolist(),
        'scene67_clip5_fraction_per_feature':(clips/frames).tolist(),
        'action_quantile_saturation_count_per_arm_axis':saturation.tolist(),
        'original_final800_observable_failure_partition':trace_diagnosis(progress),
        'original800_used_as_diagnostics_NOT_training_or_DEV_selection':True,
        'native_images_to_review':cases,'inherited_full_depth_and_geometry_proofs':inherited,
        'depth_index_sha256':sha256_file(depth_index),
        'train_numeric_source_bindings':sources,
        'old_result_unchanged':{'source':str(old),'sha256':certificate['source_sha256']},
        'new_cohorts':{'DEV':DEV,'CONF':CONF,'SMOKE':SMOKE,'FINAL':FINAL},
        'new_cohorts_disjoint_from_each_other_and_original800':True,
        'same_qualified_TRAIN_bank_and_split_receipts_no_new_evaluation_data':True,
        'fresh_seed_ids_reserved_NOT_a_proof_of_exhaustive_geometric_scene_nonoverlap':True,
        'cause_of_low_success_NOT_established_by_integrity_audit':True}
    atomic_json(OUT/'CPU_AUDIT.json',result)
    print({'passed':True,'frames':frames,'cases':len(cases)},flush=True)


ROOT_STAGING = Path(__file__).resolve().parents[2]/'VLA/audits/native_staging_r102_20261025'


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    main()
