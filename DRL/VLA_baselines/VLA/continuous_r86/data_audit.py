"""Recompute native train/runtime prefix and privileged transport contracts.

All numeric TRAIN frames are checked; RGB hash/geometry/camera coverage remains
the separately linked R83/R84 qualification, not falsely redone by this script.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.replacement_r84.data import Data, observation74, normalize_actions, denormalize_actions
from VLA.new_vlas.action_contracts import native8
from .audit import training_audit


def main(a):
    data = Data()
    audit = json.loads(Path('VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json').read_text())
    stats = audit['stats']
    mean, std = (np.asarray(stats['scene67'][k], np.float32) for k in ('mean', 'std'))
    maximum_prefix_error = maximum_transport_error = maximum_roundtrip_error = 0.
    frames = 0; counts = Counter(); saturated = np.zeros(67, np.int64)
    clipping = np.zeros(7, np.int64)
    for ei, row in enumerate(data.rows):
        n = data.numeric(ei); raw = n['scene64']; old = n['native8']
        if raw.shape != (row['frames'],64) or old.shape != (row['frames'],8):
            raise ValueError('Original full-bank numeric shape changed')
        pose = n['state20']
        # Independent native UR20 -> current runtime8 conversion; do not use
        # tool/pad rotation as a substitute for the actual EEF/FK convention.
        rebuilt = native8(pose[:, :3], pose[:, 3:9][:, [0,2,4,1,3,5]], raw[:,27])
        error = float(abs(rebuilt-old).max())
        maximum_prefix_error = max(maximum_prefix_error, error)
        if error > 2e-5:
            raise ValueError('Actual EEF axis-angle/aperture train/runtime mismatch')
        encoded = np.concatenate([raw[:, :35],
            np.stack((np.sin(raw[:,35:38]),np.cos(raw[:,35:38])),axis=-1).reshape(len(raw),6),raw[:,38:]],axis=1)
        z = (encoded-mean)/(std+np.float32(1e-8)); saturated += (abs(z)>5).sum(axis=0)
        # Actual transport code is exercised across each frame with a JSON
        # round-trip (the same numeric wire type as the real server).
        for fi in range(len(raw)):
            measured = observation74(rebuilt[fi,:6], json.loads(json.dumps(raw[fi].tolist())), stats)
            expected = np.r_[old[fi,:6],raw[fi,6],np.clip(z[fi],-5,5)]
            e = float(abs(measured-expected).max()); maximum_transport_error = max(maximum_transport_error,e)
            if e > 2e-5: raise ValueError('Privileged74 wire/preprocessing parity failed')
        actions = n['canonical7']
        normalized = normalize_actions(actions, stats)
        decoded = denormalize_actions(normalized, stats)
        delta = abs(decoded-actions); clipping += (delta>2e-7).sum(axis=0)
        maximum_roundtrip_error = max(maximum_roundtrip_error,float(delta.max()))
        if not np.isfinite(np.r_[raw.ravel(),actions.ravel(),normalized.ravel()]).all():
            raise ValueError('Nonfinite complete-bank sample')
        if not np.isin(actions[:,6],[-1,1]).all():
            raise ValueError('Unexpected non-binary original gripper label')
        open01 = (1-actions[:,6])/2
        recovered = np.where(open01>.5,-1.,1.)
        np.testing.assert_array_equal(recovered, actions[:,6])
        frames += len(raw); counts[row['task']] += len(raw)
    if frames!=119495 or len(data.rows)!=2185 or set(counts)!=set(('touch','pick','move','pick_place')):
        raise ValueError('Incomplete4-action qualified bank')
    result = {'passed':True,'all_numeric_TRAIN_frames':frames,'trajectories':len(data.rows),'per_action_frames':dict(counts),
        'max_rebuilt_current_EEF_prefix_error':maximum_prefix_error,
        'max_JSON_wire_privileged74_parity_error':maximum_transport_error,
        'max_action_roundtrip_error':maximum_roundtrip_error,'clipped_action_channel_counts':clipping.tolist(),
        'bounded_numeric67_saturation_fraction':(saturated/frames).tolist(),
        'all_signed_close_to_Dita_OPEN01_roundtrips_passed':True,
        'causal_history_and_tail_tests':'separate existing R84 native gates; no future labels added here',
        'RGB_was_NOT_full_redecoded_by_this_numeric_audit':True,
        'existing_camera_and_split_receipt':'VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json',
        'existing_receipt_sha256':sha256_file('VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json'),
        'data_sufficient_for_high_SR_NOT_proven':True,
        'expert_data_NOT_proven_TD3_generated':True,
        'wrong_object_and_drop_recovery_coverage_NOT_certified_complete':True}
    atomic_json(a.output,result)
    if a.checkpoint:
        atomic_json(a.output.with_name(a.output.stem+'_TRAIN_PREDICTIONS.json'), training_audit(a.checkpoint,data))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);a=p.parse_args();main(a)
