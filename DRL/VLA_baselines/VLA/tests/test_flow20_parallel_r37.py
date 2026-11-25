import copy
import unittest
from VLA.run_flow20_parallel_r37 import allowed,OWNER,ROOT


class GuardTests(unittest.TestCase):
    def fixture(self):
        return ({'stage':'rdt_fit_2000','status':'running'}, {'pid':123},
            {'cwd':str(ROOT),'tokens':[b'python',b'-m',b'VLA.rdt.train_scene67',b'--output',str(OWNER/'rdt_fit_2000').encode()]},
            {'compute_pids':['123'],'free_mib':7000,'available_RAM_MiB':6000})

    def test_only_exact_owned_capped_rdt_and_sufficient_memory(self):
        self.assertTrue(allowed(*self.fixture()))

    def test_unknown_gpu_process_blocks_start(self):
        parts=list(self.fixture());parts[-1]['compute_pids'].append('456')
        self.assertFalse(allowed(*parts))

    def test_uncapped_or_wrong_phase_blocks_start(self):
        for key,value in (('stage','cogact_cache_2000'),('status','failed')):
            parts=list(self.fixture());parts[0][key]=value;self.assertFalse(allowed(*parts))
        parts=list(self.fixture());parts[2]['tokens'][2]=b'other.module';self.assertFalse(allowed(*parts))

    def test_low_available_RAM_or_VRAM_blocks_start(self):
        for key in ('free_mib','available_RAM_MiB'):
            parts=list(self.fixture());parts[-1][key]=5119;self.assertFalse(allowed(*parts))

    def test_dead_or_different_workspace_process_is_not_owned(self):
        parts=list(self.fixture());parts[2]=None;self.assertFalse(allowed(*parts))
        parts=list(self.fixture());parts[2]['cwd']='/tmp';self.assertFalse(allowed(*parts))


if __name__=='__main__':unittest.main()
