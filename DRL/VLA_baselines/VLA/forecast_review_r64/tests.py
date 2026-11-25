"""CPU statistics regressions. Synthetic fixtures are not model experiments."""
import unittest

import numpy as np

from .review import (TASKS, SEEDS, N, FUTURE, counts, beta_binomial_pmf,
                     exact, sd_draws, discrete_quantile_check, ppf)


class Tests(unittest.TestCase):
    def test_actual_denominator_and_full_unique_cohort(self):
        rows = [{'seed': s, 'task': t, 'success': i % 2 == 0}
                for i, s in enumerate(SEEDS) for t in TASKS]
        self.assertEqual(counts(rows), dict.fromkeys(TASKS, 100))
        for bad in (rows[:-1], rows + [rows[0]], [*rows[:-1], rows[0]]):
            with self.assertRaises(ValueError): counts(bad)
        bad = [dict(r) for r in rows]; bad[0]['success'] = 1
        with self.assertRaises(ValueError): counts(bad)

    def test_pmf_normalization_and_moments(self):
        for a, b in ((.5, 200.5), (153.5, 47.5), (200.5, .5)):
            m = 800; p = beta_binomial_pmf(m, a, b); x = np.arange(m + 1)
            mean = m * a / (a + b)
            variance = m * a * b * (a + b + m) / ((a + b)**2 * (a + b + 1))
            self.assertAlmostEqual(p.sum(), 1., places=14)
            self.assertAlmostEqual(p @ x, mean, places=8)
            self.assertAlmostEqual(p @ (x - mean)**2, variance, places=7)

    def test_exact_predictive_mean_includes_observed_cohort(self):
        q, _ = exact(153)
        self.assertAlmostEqual(q['five_cohort_mean_expectation'], (153 + 800 * 153.5/201)/1000, places=14)
        with self.assertRaises(ValueError): exact(628)  # Not 628/800 for any action.

    def test_unseen_outcomes_keep_uncertainty(self):
        for k in (0, N):
            q, _ = exact(k)
            self.assertGreater(q['expected_five_cohort_sample_variance'], 0.)
            lo, hi = q['five_cohort_mean_95_prediction_interval']
            self.assertGreater(hi, lo)

    def test_sample_variance_moment_shared_p(self):
        q, _ = exact(153)
        sd = sd_draws(153, 200_000, 424242)
        v = sd**2
        se = v.std(ddof=1) / np.sqrt(len(v))
        self.assertLess(abs(v.mean() - q['expected_five_cohort_sample_variance']), 6 * se)

    def test_exact_quantile_and_corruption_rejection(self):
        q, pmf = exact(153)
        for p in (.025, .5, .975):
            point = (153 + int(ppf(pmf, [p])[0])) / (N * (FUTURE + 1))
            self.assertTrue(discrete_quantile_check(pmf, point, p, 153, 1000, 1e-10))
        self.assertFalse(discrete_quantile_check(pmf, .99, .5, 153, 1000, .03))

    def test_small_prior_sensitivity_not_selection(self):
        for k in (0, 153, 163, 158, N):
            jeff, _ = exact(k); uniform, _ = exact(k, 1.)
            self.assertLess(abs(jeff['five_cohort_mean_expectation'] - uniform['five_cohort_mean_expectation']), .0021)


if __name__ == '__main__':
    unittest.main()
