import unittest
from VLA.resume_smol800_r40 import verify_prefix


class PreservedPrefixTests(unittest.TestCase):
    def fixture(self):
        return {'trials':[{'seed':s,'task':{'name':t}} for s in (1,2) for t in ('touch','pick','move','pick_place')]}

    def test_partial_seed_keeps_every_terminal_outcome_and_next_slot(self):
        plan=self.fixture();rows=[{'seed':r['seed'],'task':r['task']['name'],'success':False} for r in plan['trials'][:3]]
        self.assertEqual(verify_prefix(plan,rows)[0],(1,'pick_place'))
        self.assertEqual(len(rows),3)

    def test_no_completed_row_duplicates_or_skips(self):
        plan=self.fixture()
        for rows in ([{'seed':1,'task':'pick'}],[{'seed':1,'task':'touch'}]*2):
            with self.assertRaises(ValueError):verify_prefix(plan,rows)

    def test_complete_prefix_has_no_remaining_slots(self):
        plan=self.fixture();rows=[{'seed':r['seed'],'task':r['task']['name']} for r in plan['trials']]
        self.assertEqual(verify_prefix(plan,rows),[])


if __name__=='__main__':unittest.main()
