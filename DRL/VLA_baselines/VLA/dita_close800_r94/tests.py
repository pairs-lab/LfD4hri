import copy
import signal
import unittest
from unittest.mock import patch

from .common import CP, SEEDS, authorize_signal, reload_only, validate_checkpoint_metadata
from .evaluate import validate


class Contracts(unittest.TestCase):
    def setUp(self):
        self.method={'family':'dita','checkpoint':str(CP),'actual_checkpoint_optimizer_updates':10000,
            'frozen_for_user_requested_benchmark':True,'frozen_after_confirmation':False,
            'terminal_checkpoint_no_further_Dita_finetune':True,
            'benchmark_authorization':{'user_requested_800_despite_failed_pilot_gate':True}}
        self.cohort={'role':'fresh_secondary_final800','seeds':SEEDS.copy()}

    def test_exact_target_and800(self): validate(self.method,self.cohort)

    def test_old_checkpoint_rejected(self):
        self.method['checkpoint']=str(CP.with_name('fit4096'))
        with self.assertRaises(ValueError): validate(self.method,self.cohort)

    def test_not_fake_confirmation(self):
        self.method['frozen_after_confirmation']=True
        with self.assertRaises(ValueError): validate(self.method,self.cohort)

    def test_200_unique_scenes_four_actions(self):
        self.assertEqual(len(set(SEEDS)),200)
        self.cohort['seeds'][-1]=self.cohort['seeds'][0]
        with self.assertRaises(ValueError): validate(self.method,self.cohort)

    def test_nonterminal_model_rejected(self):
        self.method.pop('terminal_checkpoint_no_further_Dita_finetune')
        with self.assertRaises(ValueError): validate(self.method,self.cohort)

    def test_no_active_optimizer_signals_any_role(self):
        for module in ('VLA.recovery_r85.dita_train','VLA.replacement_r84.train','VLA.recovery_quality_r90.train'):
            expected={'command':['python','-m',module,'fit','--output',str(CP)]}
            for role in ('queue_parent','idle_recovery','own_benchmark','own_reload'):
                for sig in (signal.SIGSTOP,signal.SIGCONT,signal.SIGTERM,signal.SIGKILL):
                    with self.assertRaises(ValueError): authorize_signal(expected,sig,role)

    def test_only_scheduler_parent_can_be_reserved(self):
        expected={'command':['python','-m','VLA.continuous_r86.owner']}
        authorize_signal(expected,signal.SIGSTOP,'queue_parent')
        with self.assertRaises(ValueError): authorize_signal(expected,signal.SIGKILL,'queue_parent')

    def test_no_wrong_server_stopped(self):
        expected={'command':['python','-m','VLA.dita_transfer_r84.server','--method','another.json','--port','5989']}
        with self.assertRaises(ValueError): authorize_signal(expected,signal.SIGTERM,'own_benchmark')

    def test_native_reload_only_no_new_Dita_fit(self):
        reload_only(['VLA.recovery_r85.dita_train','reload','--checkpoint',str(CP)])
        with self.assertRaises(ValueError): reload_only(['VLA.recovery_r85.dita_train','fit','--checkpoint',str(CP)])

    def test_completed_exact10k_metadata_required(self):
        meta={'family':'dita','completed':True,'updates':10000}
        validate_checkpoint_metadata(meta)
        for change in ({'updates':9999},{'completed':False},{'family':'vlanext_s'}):
            with self.assertRaises(ValueError): validate_checkpoint_metadata({**meta,**change})

    def test_signal_refuses_pid_reuse(self):
        from .run import checked_signal
        expected={'command':['python','-m','VLA.continuous_r86.owner'],'pid':123,'start_ticks':'42'}
        with patch('VLA.dita_close800_r94.run.same',return_value=False), \
             patch('VLA.dita_close800_r94.run.os.kill') as kill:
            self.assertFalse(checked_signal(expected,signal.SIGSTOP,'queue_parent'))
            kill.assert_not_called()

    def test_other_queue_excludes_closed_Dita(self):
        from .other_models import RECOVERY_FAMILIES
        self.assertEqual(RECOVERY_FAMILIES,('vlanext_s',))

    def test_only_original_eligibility_is_overridden(self):
        from VLA.dita800_r87.evaluate import compile_unchanged_physical_loop
        from VLA.autonomy_r83 import evaluate as original
        self.assertTrue(callable(compile_unchanged_physical_loop(original)))

    def test_cleanup_does_not_resume_running_parent_or_signal_optimizer(self):
        from .run import cleanup
        expected={'pid':123,'command':['python','-m','VLA.continuous_r86.owner'],'start_ticks':'42'}
        with patch('VLA.dita_close800_r94.run.same',return_value=True), \
             patch('VLA.dita_close800_r94.run.proc_state',return_value='S'), \
             patch('VLA.dita_close800_r94.run.checked_signal') as send:
            result=cleanup({'parent':expected},[])
            self.assertFalse(result['original_parent_resumed_on_error'])
            self.assertEqual(result['active_optimizer_signals_sent'],0)
            send.assert_not_called()


if __name__=='__main__': unittest.main()
