import copy
import unittest
from VLA.run_smol_priority_r38 import eligible,same_supervisor,allow_round2,ROOT,OWNER


class PriorityTests(unittest.TestCase):
    def fixture(self):
        return ({'status':'running','stage':'rdt_fit_2000'}, {'pid':22,'stage':'rdt_fit_2000'},
            {'cwd':str(ROOT),'state':'S','start_ticks':'123','children':'22','tokens':[b'python',b'VLA.run_all_privileged_r33',str(OWNER).encode()]},
            {'cwd':str(ROOT),'state':'R','start_ticks':'124','children':'','tokens':[b'python',b'VLA.rdt.train_scene67',str(OWNER/'rdt_fit_2000').encode()]})

    def test_only_exact_supervisor_and_actual_rdt_trainer(self):self.assertTrue(eligible(*self.fixture()))

    def test_new_stage_or_extra_child_not_eligible(self):
        parts=list(self.fixture());parts[0]['stage']='rdt_development120';self.assertFalse(eligible(*parts))
        parts=list(self.fixture());parts[2]['children']='22 33';self.assertFalse(eligible(*parts))

    def test_foreign_or_already_stopped_owner_rejected(self):
        for key,value in (('cwd','/tmp'),('state','T')):
            parts=list(self.fixture());parts[2][key]=value;self.assertFalse(eligible(*parts))
        parts=list(self.fixture());parts[3]['tokens']=[b'another.trainer'];self.assertFalse(eligible(*parts))

    def test_recycled_supervisor_pid_not_resumed(self):
        process=self.fixture()[2];lease={'supervisor_start_ticks':'123'}
        self.assertTrue(same_supervisor(lease,process))
        process['start_ticks']='new';self.assertFalse(same_supervisor(lease,process))

    def test_conditional_round2_requires_real_improvement_and_time(self):
        self.assertTrue(allow_round2(50,39,8))
        for args in ((39,39,10),(38,39,10),(72,39,10),(50,39,7.99)):
            self.assertFalse(allow_round2(*args))
        self.assertFalse(allow_round2(18,14,10,trials=40))


if __name__=='__main__':unittest.main()
