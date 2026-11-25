import unittest
from VLA.resume_scene67_evidence import terminal
from VLA.common.evaluation_statistics import success_statistics


class Scene67ResumeTests(unittest.TestCase):
    def trace(self,n,success=False,safety=None):
        return [{'step':i+1,'success':success and i==n-1,'safety_violation':safety if i==n-1 else None} for i in range(n)]
    def test_preserve_success(self):self.assertTrue(terminal(self.trace(3,True),80))
    def test_preserve_timeout(self):self.assertTrue(terminal(self.trace(80),80))
    def test_preserve_safety_failure(self):self.assertTrue(terminal(self.trace(3,safety='unsafe'),80))
    def test_partial_cannot_be_rerolled(self):self.assertFalse(terminal(self.trace(3),80))
    def test_missing_control_rejected(self):
        with self.assertRaises(ValueError):terminal(self.trace(4)[1:],80)
    def test_statistics_only_on_complete_seed_blocks(self):
        rows=[{'seed':1,'task':t,'success':False} for t in ('touch','pick','move','pick_place')]
        self.assertEqual(success_statistics(rows)['seed_blocks'],1)
        rows.append({'seed':2,'task':'touch','success':False})
        self.assertIsNone(success_statistics(rows) if len(rows)%4==0 else None)


if __name__=='__main__':unittest.main()
