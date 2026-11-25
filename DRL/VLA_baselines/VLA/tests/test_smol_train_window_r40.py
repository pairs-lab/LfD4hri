import unittest
from VLA.probe_smol_train_now_r40 import stable_window


class WindowTests(unittest.TestCase):
    def fixture(self):return ({'status':'running','stage':'waiting_RAM_before_fresh_seed_worker','active_child_pid':None},None,
        {'compute_pids':[1,2,3],'available_RAM_MiB':7000,'free_mib':6000},{1,2,3})

    def test_exact_quiet_inter_seed_window_only(self):self.assertTrue(stable_window(*self.fixture()))

    def test_active_seed_is_never_interrupted(self):
        a=list(self.fixture());a[0]['stage']='actual_frozen_native10_seed_worker';self.assertFalse(stable_window(*a))
        a=list(self.fixture());a[1]={'pid':4};self.assertFalse(stable_window(*a))

    def test_unknown_GPU_or_low_RAM_blocks_model_allocation(self):
        a=list(self.fixture());a[2]['compute_pids'].append(4);self.assertFalse(stable_window(*a))
        a=list(self.fixture());a[2]['available_RAM_MiB']=6000;self.assertFalse(stable_window(*a))

    def test_resident_model_not_blocked_by_its_own_allocated_RAM(self):
        a=list(self.fixture());a[2]['available_RAM_MiB']=4000;a[2]['free_mib']=4000
        self.assertFalse(stable_window(*a))
        self.assertTrue(stable_window(*a,initial_admission=False))


if __name__=='__main__':unittest.main()
