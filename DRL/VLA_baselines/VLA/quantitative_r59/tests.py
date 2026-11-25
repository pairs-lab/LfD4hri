import math
import unittest
from VLA.quantitative_r59.common import SEEDS, TASKS
from VLA.quantitative_r59.report import forecast


class StatisticalTests(unittest.TestCase):
    def test_no_partial_or_duplicate_cohort_forecast(self):
        with self.assertRaises(ValueError):forecast([])
        with self.assertRaises(ValueError):forecast([{'seed':SEEDS[0],'task':'touch','success':True}]*800)

    def test_future_predictions_explicit_not_measured_seeds(self):
        rows=[{'seed':s,'task':t,'success':i%2==0} for i,s in enumerate(SEEDS) for t in TASKS]
        result=forecast(rows)
        for task in TASKS:
            v=result['by_task'][task]
            self.assertEqual(v['measured_current_rate'],.5)
            self.assertTrue(v['NOT_a_measured_five_seed_mean_std'])
            self.assertAlmostEqual(v['plug_in_expected_between_200_trial_cohort_SD'],math.sqrt(.25/200))
        # Perfectly correlated task outcomes must NOT pretend800independent trials.
        self.assertGreater(result['overall_scene_block_bootstrap']['plug_in_expected_between_800_trial_cohort_SD'], .035)

    def test_all_success_has_nonzero_predictive_uncertainty(self):
        result=forecast([{'seed':s,'task':t,'success':True} for s in SEEDS for t in TASKS])
        v=result['by_task']['touch']
        self.assertTrue(v['all_success_or_all_failure_has_uncertainty_despite_zero_plugin_SD'])
        self.assertLess(v['Jeffreys_Beta_half_half_posterior_predictive']['future_single_cohort_rate_95_prediction_interval'][0],1.)


if __name__=='__main__':unittest.main()
