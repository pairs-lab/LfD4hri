import unittest
from unittest.mock import patch
from VLA.run_scene67_parallel_r34 import allowed_parallel,ROOT,OPENVLA


class ParallelInferenceGuardTests(unittest.TestCase):
    def test_requires_exact_inference_owners_and_no_other_GPU_job(self):
        def identity(pid):
            return {'cwd':str(ROOT),'tokens':[
                b'VLA.openvla.scene64_server' if pid==10 else b'VLA.evaluate_privileged_r33',str(OPENVLA).encode()]}
        state={'stage':'openvla_development120','status':'running'}
        resources={'compute_pids':['10'],'free_mib':6000,'available_RAM_MiB':6500}
        with patch('VLA.run_scene67_parallel_r34.identity',side_effect=identity):
            self.assertTrue(allowed_parallel(state,resources,{'pid':10},{'pid':11}))
            self.assertFalse(allowed_parallel({**state,'stage':'openvla_fit_1000'},resources,{'pid':10},{'pid':11}))
            self.assertFalse(allowed_parallel(state,{**resources,'compute_pids':['10','99']},{'pid':10},{'pid':11}))
            self.assertFalse(allowed_parallel(state,{**resources,'free_mib':4500},{'pid':10},{'pid':11}))
            self.assertFalse(allowed_parallel(state,{**resources,'available_RAM_MiB':4000},{'pid':10},{'pid':11}))


if __name__=='__main__':unittest.main()
