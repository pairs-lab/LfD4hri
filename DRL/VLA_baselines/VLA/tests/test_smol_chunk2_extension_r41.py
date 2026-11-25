import unittest
from VLA.extend_smol_chunk2_r41 import SEEDS,extend_allowed,compare120
from VLA import evaluate_smol800_r39 as cohort


class ExtensionTests(unittest.TestCase):
    def test_only_complete_positive_safe_and_time_bounded40_extends(self):
        proof={'trials_each':40,'paired_improvements':2,'paired_regressions':1,'native2_safety_stops':3,'native1_safety_stops':3}
        self.assertTrue(extend_allowed(proof,21600))
        for change in ({'trials_each':39},{'paired_improvements':1},{'native2_safety_stops':4}):
            self.assertFalse(extend_allowed({**proof,**change},21600))
        self.assertFalse(extend_allowed(proof,21599))

    def test_full_same120_comparison_and_no_missing40(self):
        ref=[{'seed':s,'task':t,'success':False,'safety_violation':None} for s in SEEDS for t in cohort.TASKS]
        new=[dict(r) for r in ref];new[0]['success']=True
        result=compare120(ref,new)
        self.assertEqual(result['trials_each'],120);self.assertEqual(result['paired_improvements'],1)
        self.assertTrue(result['frozen800_and_pending_fit_unchanged'])
        with self.assertRaises(ValueError):compare120(ref,new[40:])


if __name__=='__main__':unittest.main()
