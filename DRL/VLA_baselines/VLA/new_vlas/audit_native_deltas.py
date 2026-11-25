"""Quantify intended native convention fixes over EVERY source/correction frame."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    old_rows, new_rows = (json.loads((root/'index.json').read_text()) for root in (args.old, args.new))
    if len(old_rows) != len(new_rows):
        raise ValueError('Do not remove examples to hide a convention mismatch')
    before, after, frames, max_matrix_error = 0, 0, 0, 0.
    for left, right in zip(old_rows, new_rows, strict=True):
        if (left['source'], left['source_sha256']) != (right['source'], right['source_sha256']):
            raise ValueError('Original RGB/task/action bytes must remain unchanged')
        with np.load(args.old/left['cache']) as a, np.load(args.new/right['cache']) as b:
            for key in ('scene64', 'canonical7'):
                np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a['native8'][:, [0, 1, 2, 6, 7]], b['native8'][:, [0, 1, 2, 6, 7]])
            np.testing.assert_array_equal(a['adapter_action7'][:, :6], b['adapter_action7'][:, :6])
            np.testing.assert_allclose(a['adapter_action7'][:, 6]+b['adapter_action7'][:, 6], 1., rtol=0, atol=1e-7)
            for key in ('state20', 'action20'):
                np.testing.assert_array_equal(a[key][:, :3], b[key][:, :3])
                np.testing.assert_array_equal(a[key][:, 9], b[key][:, 9])
                np.testing.assert_array_equal(a[key][:, 3:9][:, [0, 3, 1, 4, 2, 5]], b[key][:, 3:9])
                np.testing.assert_array_equal(b[key][:, 10:], 0.)
            old_v, new_v = a['native8'][:, 3:6], b['native8'][:, 3:6]
            before += int((np.linalg.norm(np.diff(old_v, axis=0), axis=1) > 3).sum())
            after += int((np.linalg.norm(np.diff(new_v, axis=0), axis=1) > 3).sum())
            error = float(np.abs(Rotation.from_rotvec(old_v).as_matrix()-Rotation.from_rotvec(new_v).as_matrix()).max())
            max_matrix_error = max(error, max_matrix_error)
            frames += len(old_v)
    if after or max_matrix_error > 1e-6:
        raise ValueError('Equivalent continuous orientation representation not fully qualified')
    atomic_json(args.output/'audit.json', {'passed': True, 'frames': frames, 'episodes': len(new_rows),
        'original_images_tasks_canonical_labels_and_scene64_unchanged': True,
        'xvla_UR_native_interleaved_rotation_and_zero_second_arm_checked_all_frames': True,
        'adapter_action_gripper_changed_to_native_OPEN01_all_frames': True,
        'axisangle_temporal_jumps_over3rad_before': before,
        'axisangle_temporal_jumps_over3rad_after': after,
        'max_actual_rotation_matrix_difference': max_matrix_error,
        'no_robot_scene_camera_controller_or_success_changes': True,
        'new_data_audit_sha256': sha256_file(args.new/'audit.json'), 'model_SR_established': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
