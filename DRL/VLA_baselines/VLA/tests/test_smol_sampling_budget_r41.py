import json
import unittest
from VLA.audit_smol_sampling_budget_r41 import CONTRACT,original_groups,unique_expectation


class SamplingBudgetTests(unittest.TestCase):
    def test_expectation_is_unique_not_presentations(self):
        self.assertAlmostEqual(unique_expectation(2,.5,2),1.5)
        self.assertEqual(unique_expectation(1,1,20),1.)
        self.assertLessEqual(unique_expectation(100,.001,20000),100)

    def test_original_native_mass_and_wait_cap_exact(self):
        groups=original_groups(json.loads(CONTRACT.read_text()))
        self.assertAlmostEqual(sum(g['mass'] for g in groups),.8)
        self.assertAlmostEqual(sum(g['mass'] for g in groups if g['stratum']=='open_wait'),.04)
        self.assertEqual(sum(g['frames'] for g in groups),116257)

    def test_invalid_budget_rejected(self):
        for args in ((0,.2,20),(1,0,20),(1,2,20),(1,.2,0)):
            with self.assertRaises(ValueError):unique_expectation(*args)


if __name__=='__main__':unittest.main()
