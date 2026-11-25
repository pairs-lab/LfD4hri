import json
from pathlib import Path
import tempfile
import unittest
from VLA.evaluate_smol800_r39 import trial_specs,counts,admitted,write_report,NEW_SEEDS,TASKS,CHECKPOINT,OWNER,ROOT


class CohortTests(unittest.TestCase):
    def test_exactly200_new_fixed_seeds_all_four_actions(self):
        specs=trial_specs()
        self.assertEqual(len(specs),800)
        self.assertEqual(len({v['seed'] for v in specs}),200)
        self.assertEqual(len({(v['seed'],v['task']['name']) for v in specs}),800)
        self.assertEqual([v['task']['name'] for v in specs[:4]],list(TASKS))
        self.assertEqual(trial_specs(NEW_SEEDS[:2]),specs[:8])

    def test_no_duplicate_count_or_zero_fill(self):
        row={'seed':1,'task':'pick','success':False}
        self.assertEqual(counts([row])['touch']['trials'],0)
        self.assertIsNone(counts([row])['touch']['success_rate'])
        with self.assertRaises(ValueError):counts([row,row])

    def test_partial_report_never_claims_completed800(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan={'checkpoint':str(CHECKPOINT)}
            row={'seed':1,'task':'pick','success':False,'steps':120,'safety_violation':None,'trajectory':'actual.json'}
            write_report(root,plan,[row],'running')
            self.assertIn('INCOMPLETE: 1/800',(root/'REPORT.md').read_text())
            self.assertIn('pending',(root/'REPORT.md').read_text())
            with self.assertRaises(ValueError):write_report(root,plan,[row],'completed800')

    def test_complete_report_requires200_each(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'seed':s,'task':t,'success':True,'steps':1,'safety_violation':None,'trajectory':'actual.json'} for s in NEW_SEEDS for t in TASKS]
            write_report(root,{'checkpoint':str(CHECKPOINT)},rows,'completed800')
            self.assertIn('COMPLETED 800',(root/'REPORT.md').read_text())
            self.assertEqual(sum(v['trials'] for v in counts(rows).values()),800)


class ResourceTests(unittest.TestCase):
    def fixture(self):
        return ({'trainer_pid':22},{'status':'running','stage':'rdt_fit_2000'},{'pid':22},
            {'cwd':str(ROOT),'tokens':[b'VLA.rdt.train_scene67',str(OWNER/'rdt_fit_2000').encode()]},
            {'compute_pids':[22],'free_mib':7000,'available_RAM_MiB':6000})

    def test_held_lease_plus_only_owned_capped_trainer(self):self.assertTrue(admitted(*self.fixture()))

    def test_unknown_gpu_or_missing_lease_blocks_admission(self):
        args=list(self.fixture());args[0]=None;self.assertFalse(admitted(*args))
        args=list(self.fixture());args[-1]['compute_pids']=[22,33];self.assertFalse(admitted(*args))

    def test_trainer_finished_exclusive_under_same_lease(self):
        args=list(self.fixture());args[-1]['compute_pids']=[55];args[3]=None
        self.assertTrue(admitted(*args,own_pid=55))
        args[-1]['available_RAM_MiB']=5000;self.assertFalse(admitted(*args,own_pid=55))

    def test_initial_empty_gap_never_races_waiting_fit(self):
        args=list(self.fixture());args[-1]['compute_pids']=[];args[3]=None
        self.assertFalse(admitted(*args))

    def test_continuation_ignores_only_own_gpu_pid_not_foreign_user(self):
        args=list(self.fixture());args[-1]['compute_pids']=[22,55]
        self.assertTrue(admitted(*args,own_pid=55))
        args[-1]['compute_pids']=[22,55,66];self.assertFalse(admitted(*args,own_pid=55))


if __name__=='__main__':unittest.main()
