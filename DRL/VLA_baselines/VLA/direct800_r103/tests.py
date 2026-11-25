"""Synthetic scheduling/authorization tests, never robot SR measurements."""
from pathlib import Path
import tempfile,unittest
from unittest import mock
from VLA.admission_r100.common import atomic_json,read
from . import workflow as direct


class DirectTests(unittest.TestCase):
    def test_never_retire_an_actual_trainer(self):
        expected={'pid':123,'command':['python','-m','VLA.admission_r100.train']}
        with mock.patch.object(direct,'terminal',return_value=False),mock.patch.object(direct.os,'kill') as kill:
            with self.assertRaises(ValueError):direct.retire_scheduler(expected)
            kill.assert_not_called()

    def test_watchdog_retired_before_scheduler_and_no_trainer_signal(self):
        with tempfile.TemporaryDirectory(prefix='r103_SYNTHETIC_') as tmp:
            root=Path(tmp);old=root/'old';new=root/'new'
            atomic_json(new/'HANDOFF.json',{'old_owner':{'pid':1},'old_watchdog':{'pid':2},'adopted_trainer':{'pid':3}})
            with mock.patch.object(direct,'OUT',new),mock.patch.object(direct,'R100',old),mock.patch.object(
                direct,'retire_scheduler') as retire:
                direct.handoff()
                self.assertEqual(retire.call_args_list,[mock.call({'pid':2},watchdog=True),mock.call({'pid':1})])
                self.assertEqual(read(new/'COORDINATOR_HANDOFF_COMPLETE.json')['trainer_signals_sent'],0)

    def exercise(self,root,passed):
        import VLA.admission_r100.workflow as base
        cp=root/'checkpoint';atomic_json(cp/'FRESH_RELOAD.json',{'passed':passed,'max_errors':[0.]*8})
        with mock.patch.object(direct,'configure',return_value=base):cls=direct.owner_class()
        owner=cls.__new__(cls);owner.wait_for_fit=mock.Mock(return_value=cp)
        owner.gpu=mock.Mock();owner.evaluate=mock.Mock(return_value=root/'synthetic_progress.json');owner.status=mock.Mock()
        return owner

    def test_no_DEV_CONF_or_extra_finetune_before_direct800(self):
        import VLA.admission_r100.workflow as base
        from VLA.admission_r100.evaluate import FINAL
        with tempfile.TemporaryDirectory(prefix='r103_SYNTHETIC_') as tmp:
            root=Path(tmp);owner=self.exercise(root,True)
            with mock.patch.object(direct,'OUT',root),mock.patch.object(direct,'direct_lock',return_value=root/'frozen.json'),mock.patch.object(
                base,'audit_result',return_value={'per_action':{},'synthetic_NOT_actual_SR':True}):owner.run_direct()
            owner.evaluate.assert_called_once_with(root/'frozen.json','final800','fresh_secondary_final800',FINAL)
            self.assertEqual(len(FINAL),200)
            self.assertEqual(owner.gpu.call_count,1)
            self.assertIn('--op',owner.gpu.call_args.args[1]);self.assertIn('reload',owner.gpu.call_args.args[1])

    def test_direct_authorization_never_bypasses_reload_integrity(self):
        with tempfile.TemporaryDirectory(prefix='r103_SYNTHETIC_') as tmp:
            root=Path(tmp);owner=self.exercise(root,False)
            with mock.patch.object(direct,'OUT',root),mock.patch.object(direct,'direct_lock') as lock:
                with self.assertRaises(ValueError):owner.run_direct()
                owner.evaluate.assert_not_called();lock.assert_not_called()


if __name__=='__main__':unittest.main()
