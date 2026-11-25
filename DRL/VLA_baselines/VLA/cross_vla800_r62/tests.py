"""CPU fixtures only. Synthetic outcomes below are NEVER experiment data."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from VLA.common.artifact_io import atomic_json
from .common import SEEDS,TASKS,matrix,admissible,method_for,checkpoint_hashes,verify
from .report import forecast
from .media import DiscardWriter,discard_snapshot


def rows(function):
    return [{'seed':s,'task':t,'success':bool(function(i,j))}
            for i,s in enumerate(SEEDS) for j,t in enumerate(TASKS)]


class StatisticalTests(unittest.TestCase):
    def test_refuse_partial_duplicate_wrong_cohort_and_nonboolean(self):
        data=rows(lambda i,j:i%2)
        for bad in (data[:-1],data[:-1]+[data[0]],
                    [{**v,'seed':-1} for v in data],
                    [{**v,'success':int(v['success'])} for v in data]):
            with self.assertRaises(ValueError):matrix(bad)

    def test_order_independent_and_exact_four_task_blocks(self):
        a=rows(lambda i,j:i%2==j%2)
        np.testing.assert_array_equal(matrix(a),matrix(list(reversed(a))))

    def test_forecast_explicitly_not_measured_seeds(self):
        f=forecast(rows(lambda i,j:i%2==0),draws=2000)
        self.assertEqual(f['actual_additional_cohorts_run'],0)
        self.assertFalse(f['training_seed_variance_estimable'])
        self.assertGreater(f['overall']['expected_sampling_SD_per_800_trial_cohort'],.035)
        self.assertAlmostEqual(f['by_task']['touch']['expected_sampling_SD_per_200_trial_cohort'],np.sqrt(.25/200))

    def test_extremes_have_nonzero_posterior_prediction_uncertainty(self):
        for success in (False,True):
            f=forecast(rows(lambda i,j:success),draws=3000)
            lo,hi=f['overall']['future_one_cohort_SR_95_prediction_interval']
            self.assertLess(lo,hi)
            for t in TASKS:
                lo,hi=f['by_task'][t]['future_one_cohort_SR_95_prediction_interval']
                self.assertLess(lo,hi)

    def test_anti_correlated_tasks_are_not_800_independent_samples(self):
        f=forecast(rows(lambda i,j:i%2==j%2),draws=1000)
        self.assertEqual(f['overall']['expected_sampling_SD_per_800_trial_cohort'],0)
        self.assertGreater(f['by_task']['touch']['expected_sampling_SD_per_200_trial_cohort'],.03)


class ProtocolTests(unittest.TestCase):
    def test_symlinked_checkpoint_directory_and_changed_target_are_bound(self):
        with tempfile.TemporaryDirectory(prefix='r62_checkpoint_fixture_') as name:
            root=Path(name);cp=root/'cp';cp.mkdir();weights=root/'weights';weights.mkdir()
            atomic_json(weights/'value.json',{'weight':1})
            (cp/'params').symlink_to(weights,target_is_directory=True)
            bound=checkpoint_hashes(cp)
            self.assertIn(str(cp/'params/value.json'),bound);verify(bound)
            other=root/'other';other.mkdir();atomic_json(other/'value.json',{'weight':2})
            (cp/'params').unlink();(cp/'params').symlink_to(other,target_is_directory=True)
            with self.assertRaises(ValueError):verify(bound)

    def test_discard_writer_keeps_original_render_branch_truthy(self):
        with tempfile.TemporaryDirectory(prefix='r62_writer_fixture_') as name:
            writer=DiscardWriter(Path(name)/'not_created.mp4')
            self.assertTrue(writer);writer.append_data(np.zeros((4,4,3),np.uint8));writer.close()
            self.assertEqual(writer.frames,1);self.assertEqual(list(Path(name).iterdir()),[])
            with self.assertRaises(ValueError):writer.append_data(None)

    def test_snapshot_retains_exact_segmentation_call_order(self):
        class Env:
            def __init__(self):self.calls=[]
            def target_mask(self,*v):self.calls.append(v)
        env=Env();discard_snapshot(None,None,None,env)
        self.assertEqual(env.calls,[('front','pick'),('front','place'),('wrist','pick'),('wrist','place')])

    def test_resource_admission_never_uses_a_loaded_training_slot(self):
        self.assertTrue(admissible({'RAM_available_MiB':5000,'GPU_free_MiB':15000,'disk_free_GiB':40}))
        for key in ('RAM_available_MiB','GPU_free_MiB','disk_free_GiB'):
            v={'RAM_available_MiB':5000,'GPU_free_MiB':15000,'disk_free_GiB':40};v[key]=1
            self.assertFalse(admissible(v))

    def test_method_selection_refuses_final_outcomes(self):
        with tempfile.TemporaryDirectory(prefix='r62_selection_fixture_') as name:
            root=Path(name);cp=root/'cp';cp.mkdir()
            d=root/'development.json';c=root/'confirmation.json'
            def fixture(seeds):
                rr=[{'seed':s,'task':t,'success':False} for s in seeds for t in TASKS]
                return {'rows':rr,'completed_trials':40,'planned_trials':40,'successes':0}
            atomic_json(d,fixture(range(20268011,20268021)))
            atomic_json(c,fixture(range(20268101,20268111)))
            atomic_json(root/'status.json',{'selected_methods':{'pi0':{
                'checkpoint':str(cp),'mode':'map3','prefix':4,
                'development_progress':str(d),'confirmation_progress':str(c)}}})
            with patch('VLA.cross_vla800_r62.common.MAIN',root):
                method_for('pi0')
                atomic_json(d,fixture(range(20271001,20271011)))
                with self.assertRaises(ValueError):method_for('pi0')


if __name__=='__main__':unittest.main()
