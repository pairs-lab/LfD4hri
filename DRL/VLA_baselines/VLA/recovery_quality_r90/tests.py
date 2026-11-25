"""No GPU, no simulator and no active checkpoint mutations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import numpy as np

from .audit import development_audit, regimes, TASKS
from .collect import intervention, TARGETS, KINDS


class Contracts(unittest.TestCase):
    def test_safety_recorder_does_not_change_actions_or_observations(self):
        from .evaluate import diagnostic_class
        class Plant:
            def __init__(self, seed):
                self.task = SimpleNamespace(name='pick', prompt='Pick blue block')
                self.step_count = 4
            def step(self, action):
                self.received = action
                self.result = (object(), 0., True, {'safety_violation':'diagnostic_test',
                    'safety_diagnostics_before_rollback':{'mimic_error':.16},'rollback_restored':True})
                return self.result
        with tempfile.TemporaryDirectory() as tmp:
            cls = diagnostic_class(Plant, tmp)
            action = np.array([0.,0.,-.01,0.,0.,0.,-1.])
            for seed in (11,12):
                plant = cls(seed=seed)
                result = plant.step(action)
                self.assertIs(plant.received, action)
                self.assertIs(result, plant.result)
                self.assertTrue((Path(tmp)/'SAFETY_DIAGNOSTICS'/f'seed{seed}'/'pick_4.json').exists())

    def test_corrective_sampler_preserves_chosen_task_target(self):
        from .data import Data, OriginalData
        from collections import Counter
        fake = Data.__new__(Data)
        fake.rows = [{'task':'move','pick_name':'blue block','source_kind':'complete_original','frames':60},
            {'task':'move','pick_name':'blue block','source_kind':'near_miss_corrective','frames':60}]
        fake.corrective = {('move','blue block'):[1]}; fake.exposure=Counter()
        rng = np.random.default_rng(90)
        with mock.patch.object(OriginalData, 'draw', return_value=(0,30)):
            for _ in range(100):
                ei,fi = fake.draw(rng)
                self.assertEqual(fake.rows[ei]['task'], 'move')
                self.assertEqual(fake.rows[ei]['pick_name'], 'blue block')
                self.assertTrue(0<=fi<60)
        self.assertGreater(fake.exposure['near_miss_corrective'], 0)
        self.assertGreater(fake.exposure['complete_original'], 0)

    def test_production_data_rejects_two_target_pilot(self):
        from .data import Data, OriginalData
        from . import data as module
        proof = {'passed':True,'split':'train','pre_action_no_future_labels':True,
                 'minimum_targets_per_task_kind':2}
        with mock.patch.object(OriginalData, '__init__', return_value=None), \
             mock.patch.object(module, 'read', side_effect=[proof,[]]):
            with self.assertRaisesRegex(ValueError, 'pilot'):
                Data()

    def test_missing_grasp_is_not_drop(self):
        raw = np.zeros((3, 64)); action = np.zeros((3, 7))
        raw[:, 9] = .09; raw[:, 11] = .03; action[:, 2] = -.01
        r = regimes(raw, action)
        self.assertEqual(int(r['unheld_under10cm_offset_over2cm'].sum()), 3)
        self.assertEqual(int(r['actual_hold_lost_between_pre_action_frames'].sum()), 0)
        self.assertEqual(int(r['unheld_under14cm_descending_over5mm'].sum()), 3)

    def test_verified_hold_loss_and_transport(self):
        raw = np.zeros((3, 64)); action = np.zeros((3, 7))
        raw[:2, 17] = raw[:2, 31] = 1
        action[:, 0] = .02
        r = regimes(raw, action)
        self.assertEqual(int(r['actual_hold_lost_between_pre_action_frames'].sum()), 1)
        self.assertEqual(int(r['held_translation_over5mm'].sum()), 2)

    def test_wrong_object_not_correct_grasp(self):
        raw = np.zeros((1, 64)); action = np.zeros((1, 7))
        raw[:, 30:32] = 1
        r = regimes(raw, action)
        self.assertTrue(r['wrong_attached'][0])
        self.assertFalse(r['correct_bilateral_hold'][0])
        self.assertFalse(r['unheld_under14cm'][0])

    def test_reject_shape_and_nan(self):
        with self.assertRaises(ValueError): regimes(np.zeros((1, 63)), np.zeros((1, 7)))
        raw = np.zeros((1, 64)); raw[0, 4] = np.nan
        with self.assertRaises(ValueError): regimes(raw, np.zeros((1, 7)))

    def test_never_read_final_or_confirmation_for_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'progress.json'
            for role in ('development_confirmation', 'fresh_secondary_final800'):
                p.write_text(json.dumps({'cohort_role': role}))
                with self.assertRaises(ValueError): development_audit(p)

    def test_interventions_never_descend_or_rotate(self):
        for task in TASKS:
            for target in TARGETS:
                for kind in KINDS:
                    for tick in range(8):
                        a = intervention(task, target, kind, tick)
                        np.testing.assert_array_equal(a[2:6], 0.)
                        self.assertEqual(float(a[6]), 1. if kind.endswith('closed') else -1.)
                        if tick: np.testing.assert_array_equal(a[:3], 0.)
        with self.assertRaises(ValueError): intervention('pick', 'unknown', KINDS[0], 0)


if __name__ == '__main__':
    unittest.main()
