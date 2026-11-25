import unittest
from VLA.flower800_r77.evaluate import validate_cohort
from VLA.cross_vla800_r62.common import SEEDS


class CohortTests(unittest.TestCase):
    def test_exact_200_seeds_is_800_trials(self):
        s=validate_cohort({'role':'final_fixed_evaluation800','seeds':SEEDS})
        self.assertEqual(len(s)*4,800);self.assertEqual(len(set(s)),200)
    def test_no_development_role_alias(self):
        with self.assertRaises(ValueError):validate_cohort({'role':'development_selection','seeds':SEEDS})
    def test_no_seed_replacement(self):
        with self.assertRaises(ValueError):validate_cohort({'role':'final_fixed_evaluation800','seeds':SEEDS[:-1]+[0]})
    def test_no_partial_cohort(self):
        with self.assertRaises(ValueError):validate_cohort({'role':'final_fixed_evaluation800','seeds':SEEDS[:10]})
    def test_control_is_separate_from_800(self):
        self.assertEqual(validate_cohort({'role':'implementation_parity_control4','seeds':[20268011]},True),[20268011])


if __name__=='__main__':unittest.main()
