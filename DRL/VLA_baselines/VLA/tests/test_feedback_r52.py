import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from VLA.collect_pose_feedback_r52 import upright_error
from VLA.evaluate_task_chunks_r52 import choose_schedule, language_task
from VLA.new_vlas.action_contracts import normalize_bounds
from VLA.prepare_feedback_data_r52 import floor_bounds, physical_statistics
from VLA.run_cross_vla120 import SEEDS
from VLA.collect_multidirection_feedback_r52 import fault_action, TASKS, TARGETS
from VLA.fit_feedback_r52 import draw


class FeedbackRepairTests(unittest.TestCase):
    def test_sampler_exposes_sparse_lateral_control_not_just_vertical_prior(self):
        class Data:
            groups = {task: {kind: {'block': [0]} for kind in
                ('pose_corrective', 'fresh_corrective', 'complete_original')} for task in TASKS}
            def numeric(self, i):
                a, r = np.zeros((100, 7)), np.zeros((100, 64))
                a[:, 6] = -1
                a[80, 0] = .03
                a[81, 2] = .03
                a[82, 6] = 1
                return {'canonical7': a, 'scene64': r}
        rng, sampled = np.random.default_rng(52), []
        for _ in range(2000):
            sampled.append(draw(Data(), rng)[1])
        self.assertGreater(sampled.count(80)/len(sampled), .30)

    def test_actual_fault_plan_covers_all_four_tilt_quadrants_per_task(self):
        for task in TASKS:
            values = [fault_action(task, target, 'wrist_tilt') for target in TARGETS]
            self.assertEqual(set(map(tuple, np.sign(np.asarray(values)[:, 3:5]).astype(int))),
                             {(1,1), (1,-1), (-1,1), (-1,-1)})
            self.assertTrue(all(v[6] == -1 for v in values))

    def test_empty_close_fault_remains_translation_only_and_closed(self):
        for task in TASKS:
            for target in TARGETS:
                action = fault_action(task, target, 'closed_empty')
                self.assertEqual(action[6], 1.)
                self.assertLess(np.linalg.norm(action[:3]), .021)
                np.testing.assert_array_equal(action[3:6], 0)

    def test_upright_is_zero_correction(self):
        r = Rotation.from_euler('xyz', [0., np.pi, 0.]).as_matrix()
        relative, tilt = upright_error(r[:, :2].T.reshape(6))
        self.assertLess(tilt, 1e-6)
        np.testing.assert_allclose(relative, 0, atol=1e-6)

    def test_real_tilt_correction_reduces_measured_tilt(self):
        rotation = Rotation.from_euler('xyz', [0., np.pi, 0.])*Rotation.from_euler('xyz', [.045, -.04, .025])
        relative, before = upright_error(rotation.as_matrix()[:, :2].T.reshape(6))
        after_rotation = rotation*Rotation.from_euler('xyz', np.clip(.65*relative, -.025, .025))
        _, after = upright_error(after_rotation.as_matrix()[:, :2].T.reshape(6))
        self.assertLess(after, before)

    def test_range_floor_does_not_mutate_parent(self):
        old = {'q01': [-.0001]*7, 'q99': [.0001]*7, 'mask': [True]*6+[False]}
        before = copy.deepcopy(old)
        new = floor_bounds(old, [0., 0., 0., .03, .03, .08, 0.])
        self.assertEqual(old, before)
        self.assertEqual(new['q99'][3], .03)

    def test_old_action_range_cannot_represent_correction(self):
        old = {'q01': [-.05]*3+[-.0001]*2+[-.08, 0.], 'q99': [.05]*3+[.0001]*2+[.08, 1.], 'mask': [True]*6+[False]}
        action = np.array([0., 0., 0., .025, -.02, 0., 1.])
        encoded = normalize_bounds(action, old)
        decoded = .5*(encoded+1)*(np.array(old['q99'])-old['q01'])+old['q01']
        self.assertGreater(abs(decoded[3]-action[3]), .02)

    def test_new_action_range_roundtrip_keeps_radian_magnitude(self):
        old = {'q01': [-.05]*3+[-.0001]*2+[-.08, 0.], 'q99': [.05]*3+[.0001]*2+[.08, 1.], 'mask': [True]*6+[False]}
        stats = floor_bounds(old, [0., 0., 0., .03, .03, .08, 0.])
        action = np.array([0., 0., 0., .025, -.02, 0., 1.])
        encoded = normalize_bounds(action, stats)
        decoded = .5*(encoded+1)*(np.array(stats['q99'])-stats['q01'])+stats['q01']
        np.testing.assert_allclose(decoded[:6], action[:6], atol=1e-7)

    def test_language_prefix_never_requires_target_coordinates(self):
        for prompt, task in [('Touch the egg', 'touch'), ('Pick the pear up', 'pick'),
                             ('Move the block right', 'move'), ('Put the block in a bowl', 'pick_place')]:
            self.assertEqual(language_task(prompt), task)
        with self.assertRaises(ValueError):
            language_task('unsupported task')

    def make_progress(self, directory, prefix, outcomes=None, cp='same_checkpoint', drop=False):
        tasks = ('touch', 'pick', 'move', 'pick_place')
        rows = [{'seed': seed, 'task': task, 'success': bool((outcomes or {}).get(task, False)),
                 'safety_violation': None} for seed in SEEDS[:2] for task in tasks]
        if drop:
            rows.pop()
        path = Path(directory)/f'prefix{prefix}.json'
        path.write_text(json.dumps({'family': 'xvla', 'privileged67': True, 'checkpoint': cp,
            'native_execution_chunk': prefix, 'rows': rows}))
        return path

    def test_per_task_selection_uses_full_same_weights_matches(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [self.make_progress(d, 4, {'move': True}), self.make_progress(d, 6),
                     self.make_progress(d, 30, {'touch': True, 'pick': True})]
            result = choose_schedule(paths, 'xvla')
            self.assertEqual(result['tasks']['move']['selected_prefix'], 4)
            self.assertEqual(result['tasks']['touch']['selected_prefix'], 30)
            self.assertEqual(result['tasks']['pick_place']['selected_prefix'], 30)

    def test_cannot_pool_different_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [self.make_progress(d, 4), self.make_progress(d, 6), self.make_progress(d, 30, cp='other')]
            with self.assertRaises(ValueError):
                choose_schedule(paths, 'xvla')

    def test_incomplete_prefix_is_not_scored_as_zero(self):
        with tempfile.TemporaryDirectory() as d:
            paths = [self.make_progress(d, 4, drop=True), self.make_progress(d, 6), self.make_progress(d, 30)]
            with self.assertRaises(ValueError):
                choose_schedule(paths, 'xvla')

    def test_too_few_candidates_is_not_complete_selection(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                choose_schedule([self.make_progress(d, 4), self.make_progress(d, 30)], 'xvla')


if __name__ == '__main__':
    unittest.main()
