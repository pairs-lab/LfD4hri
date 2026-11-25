import copy
import unittest
from unittest.mock import patch
from .common import CP,SEEDS,same
from .evaluate import validate


class Contracts(unittest.TestCase):
    def setUp(self):
        self.method={'family':'dita','checkpoint':str(CP),'actual_checkpoint_optimizer_updates':4096,
            'frozen_for_user_requested_benchmark':True,'frozen_after_confirmation':False,
            'benchmark_authorization':{'user_requested_800_despite_failed_pilot_gate':True}}
        self.cohort={'role':'fresh_secondary_final800','seeds':SEEDS.copy()}
    def test_exact4096_and800(self):
        validate(self.method,self.cohort)
    def test_old_checkpoint_rejected(self):
        self.method['checkpoint']=str(CP.with_name('fit1024'))
        with self.assertRaises(ValueError):validate(self.method,self.cohort)
    def test_not_fake_confirmation(self):
        self.method['frozen_after_confirmation']=True
        with self.assertRaises(ValueError):validate(self.method,self.cohort)
    def test_duplicate_scene_rejected(self):
        self.cohort['seeds'][-1]=self.cohort['seeds'][0]
        with self.assertRaises(ValueError):validate(self.method,self.cohort)
    def test_no_pid_reuse_signals(self):
        expected={'pid':123,'start_ticks':'42','command':['native_owner']}
        with patch('VLA.dita4096_benchmark_r92.common.identity',return_value={**expected,'start_ticks':'43'}):
            self.assertFalse(same(expected))
        with patch('VLA.dita4096_benchmark_r92.common.identity',return_value=copy.deepcopy(expected)):
            self.assertTrue(same(expected))
    def test_single_override_only(self):
        from VLA.dita800_r87.evaluate import compile_unchanged_physical_loop
        from VLA.autonomy_r83 import evaluate as original
        self.assertTrue(callable(compile_unchanged_physical_loop(original)))
    def test_training_child_can_never_be_stopped_by_cleanup(self):
        import signal
        from .run import owned_signal
        expected={'pid':123,'start_ticks':'42','command':['python','-m','VLA.replacement_r84.train']}
        with patch('VLA.dita4096_benchmark_r92.run.same',return_value=True), \
             patch('VLA.dita4096_benchmark_r92.run.os.kill') as kill:
            with self.assertRaises(ValueError):owned_signal(expected,signal.SIGTERM)
            kill.assert_not_called()
    def test_parent_resume_idempotent(self):
        from .run import resume_parent
        expected={'pid':123,'start_ticks':'42','command':['python','-m','VLA.continuous_r86.owner']}
        with patch('VLA.dita4096_benchmark_r92.run.same',return_value=True), \
             patch('VLA.dita4096_benchmark_r92.run.proc_state',return_value='S'), \
             patch('VLA.dita4096_benchmark_r92.run.os.kill') as kill:
            self.assertFalse(resume_parent(expected));kill.assert_not_called()


if __name__=='__main__':unittest.main()
