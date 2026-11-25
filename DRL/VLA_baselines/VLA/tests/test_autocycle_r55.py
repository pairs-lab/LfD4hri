import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from VLA.autocycle_r55.decision import qualified_progress,stable,candidate_key,continue_after_round
from VLA.autocycle_r55.sampling import draw,contract
from VLA.autocycle_r55.report import wilson,final_result


TASKS=('touch','pick','move','pick_place')


def progress(successes=24,seeds=range(20268011,20268021),abort=0):
    rows=[{'seed':s,'task':t,'success':False,'safety_violation':None} for s in seeds for t in TASKS]
    for i,row in enumerate(rows):row['success']=i<successes;row['safety_violation']='stop' if i<abort else None
    return {'completed_trials':len(rows),'planned_trials':len(rows),'successes':successes,'rows':rows}


class AutocycleTests(unittest.TestCase):
    def test_final_never_accepted_for_selection(self):
        for seeds in (range(20269701,20269711),range(20269801,20269811),range(20269901,20269911)):
            with patch('VLA.autocycle_r55.decision.read',return_value=progress(seeds=seeds)):
                with self.assertRaises(ValueError):qualified_progress('ignored')

    def test_partial_or_duplicate_cohort_rejected(self):
        p=progress();p['rows'][1]=p['rows'][0]
        with patch('VLA.autocycle_r55.decision.read',return_value=p):
            with self.assertRaises(ValueError):qualified_progress('ignored')

    def test_confirmation_is_explicit_development(self):
        p=progress(seeds=range(20268101,20268111))
        with patch('VLA.autocycle_r55.decision.read',return_value=p):self.assertEqual(qualified_progress('ignored')['successes'],24)

    def test_stable_requires_both_cohorts_and_safety(self):
        c={'development_progress':'a','confirmation_progress':'b'}
        for a,b,aborts,expected in ((29,23,0,False),(24,24,4,True),(24,24,5,False)):
            values={'a':progress(a,abort=aborts),'b':progress(b,seeds=range(20268101,20268111))}
            with patch('VLA.autocycle_r55.decision.read',side_effect=lambda p:values[str(p)]):self.assertEqual(stable(c),expected)

    def test_round_two_requires_measured_gain(self):
        old={'development_progress':'a','confirmation_progress':'b'}
        new={'development_progress':'c','confirmation_progress':'d'}
        values={'a':progress(10),'b':progress(10,seeds=range(20268101,20268111)),
                'c':progress(11),'d':progress(10,seeds=range(20268101,20268111))}
        with patch('VLA.autocycle_r55.decision.read',side_effect=lambda p:values[str(p)]):
            self.assertFalse(continue_after_round(old,new,1))
            values['d']=progress(11,seeds=range(20268101,20268111))
            self.assertTrue(continue_after_round(old,new,1));self.assertFalse(continue_after_round(old,new,2))

    def test_aggregate60percent_cannot_hide_a_zero_success_task(self):
        a=progress();b=progress(seeds=range(20268101,20268111))
        for p in (a,b):
            remaining=4
            for row in p['rows']:
                row['success']=row['task'] in ('touch','pick') or (row['task']=='move' and remaining>0)
                if row['task']=='move' and row['success']:remaining-=1
        with patch('VLA.autocycle_r55.decision.read',side_effect=lambda path:a if path=='a' else b):
            self.assertFalse(stable({'development_progress':'a','confirmation_progress':'b'}))

    def test_tie_preserves_confirmed_baseline(self):
        c={'development_progress':'a','confirmation_progress':'b','preserved_baseline':True}
        with patch('VLA.autocycle_r55.decision.read',side_effect=lambda p:progress(seeds=range(20268011,20268021) if p=='a' else range(20268101,20268111))):
            self.assertGreater(candidate_key(c),candidate_key({**c,'preserved_baseline':False}))

    def test_sampler_never_modifies_labels_or_adds_input(self):
        actions=np.zeros((20,7),np.float32);actions[:10,0]=.02;actions[:,6]=1
        states=np.zeros((20,64),np.float32)
        groups={t:{k:{'object':[0]} for k in ('complete_original','fresh_corrective','pose_corrective')} for t in TASKS}
        data=SimpleNamespace(groups=groups,numeric=lambda _: {'canonical7':actions,'scene64':states})
        before=actions.copy()
        for profile in ('acquisition','transport'):
            self.assertAlmostEqual(sum(contract(profile)['phases']),1.)
            for _ in range(20):
                ei,fi=draw(data,np.random.default_rng(_),profile);self.assertEqual(ei,0);self.assertTrue(0<=fi<20)
        np.testing.assert_array_equal(before,actions)

    def test_report_rejects_different_final_cohort(self):
        p=progress(60,seeds=range(20269801,20269831))
        with patch('VLA.autocycle_r55.report.read',return_value=p):
            with self.assertRaises(ValueError):final_result(Path('ignored'))

    def test_wilson_interval_is_not_an_accuracy_guarantee(self):
        lo,hi=wilson(72,120);self.assertLess(lo,.6);self.assertGreater(hi,.6);self.assertGreater(lo,.5)


if __name__=='__main__':unittest.main()
