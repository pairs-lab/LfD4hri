"""Fault-injection fixtures only; zero benchmark episodes."""
import unittest
from unittest import mock
from pathlib import Path
import tempfile
import numpy as np
from .common import depth_contract,geometry_contract,gpu_admissible,stop_reason
from .runner import stop_owned
from .common import atomic_json,read


class Contracts(unittest.TestCase):
    def test_actual_metric_depth_contract(self):
        d=np.ones((2,384,384),np.float32)*.5
        self.assertEqual(depth_contract(d,np.arange(2),2)['min_valid_pixels'],384**2)

    def test_zero_nan_empty_mask_depth_rejected(self):
        for value in (0.,np.nan,np.inf,4.):
            with self.assertRaises(ValueError):
                depth_contract(np.full((1,384,384),value,np.float32),np.array([0]),1)

    def test_depth_order_shape_and_future_index_rejected(self):
        d=np.ones((2,384,384),np.float32)
        for order in ([1,0],[0,2]):
            with self.assertRaises(ValueError):depth_contract(d,np.array(order),2)
        with self.assertRaises(ValueError):depth_contract(d[:,:200],np.arange(2),2)

    def test_camera_intrinsic_and_proper_extrinsic(self):
        K=np.diag([300.,300.,1.]);geometry_contract(K,np.eye(4))
        for bad in (-K,np.full((3,3),np.nan)):
            with self.assertRaises(ValueError):geometry_contract(bad,np.eye(4))
        cv=np.eye(4);cv[0,0]=-1
        with self.assertRaises(ValueError):geometry_contract(K,cv)
        cv=np.eye(4);cv[3,3]=0
        with self.assertRaises(ValueError):geometry_contract(K,cv)

    def test_vram_alone_does_not_admit_when_host_memory_low(self):
        self.assertFalse(gpu_admissible({'RAM_available_MiB':4000,'GPU_free_MiB':15000}))
        self.assertFalse(gpu_admissible({'RAM_available_MiB':8000,'GPU_free_MiB':5000}))
        self.assertTrue(gpu_admissible({'RAM_available_MiB':4608,'GPU_free_MiB':7168}))

    def test_primary_phase_change_and_heartbeat_cause_only_probe_to_defer(self):
        r={'RAM_available_MiB':5000,'GPU_free_MiB':8000}
        current={'family':'gr00t_n15','stage':'adapt2048_attempt00'}
        self.assertIsNone(stop_reason(r,current,1,True))
        self.assertIsNotNone(stop_reason(r,current,91,True))
        self.assertIsNotNone(stop_reason(r,{**current,'stage':'final800'},1,True))
        self.assertIsNotNone(stop_reason(r,{**current,'family':'falcon_esm'},1,True))
        self.assertIsNone(stop_reason(r,{**current,'stage':'final800'},500,False))

    def test_resource_pressure_stops_only_additional_work(self):
        current={'family':'gr00t_n15','stage':'adapt2048_attempt00'}
        for r in ({'RAM_available_MiB':1700,'GPU_free_MiB':8000},
                  {'RAM_available_MiB':5000,'GPU_free_MiB':2500}):
            self.assertIsNotNone(stop_reason(r,current,1,True))

    def test_never_signal_live_training_owner_or_pid_reuse(self):
        for module in ('VLA.admission_r100.train','VLA.admission_r100.workflow'):
            expected={'pid':123,'command':['python','-m',module]}
            with mock.patch('VLA.staging_r102.runner.same_owned_process',return_value=True),mock.patch(
                'VLA.staging_r102.runner.os.kill') as kill:
                with self.assertRaises(ValueError):stop_owned(expected)
                kill.assert_not_called()
        with mock.patch('VLA.staging_r102.runner.same_owned_process',return_value=False),mock.patch(
            'VLA.staging_r102.runner.os.kill') as kill:
            with self.assertRaises(ValueError):stop_owned({'pid':123})
            kill.assert_not_called()

    def test_live_index_reads_actual_training_not_stale_phase_timestamp(self):
        from .report import render
        with tempfile.TemporaryDirectory(prefix='r102_SYNTHETIC_') as tmp:
            base=Path(tmp);root=base/'R100';output=base/'staging'
            atomic_json(root/'WORKFLOW_STATUS.json',{'family':'gr00t_n15','stage':'adapt2048_attempt00'})
            atomic_json(root/'gr00t_n15/WORKFLOW_STATUS.json',{'family':'gr00t_n15',
                'stage':'adapt2048_attempt00','UTC':'old phase-start snapshot'})
            atomic_json(root/'gr00t_n15/adapt2048_attempt00/status.json',{'pid':123,'updates':7,'target_updates':20})
            ident={'command':['python','-m','VLA.admission_r100.train','--family','gr00t_n15']}
            with mock.patch('VLA.staging_r102.report.R100',root),mock.patch(
                'VLA.staging_r102.report.OUT',output),mock.patch(
                'VLA.staging_r102.report.alive',return_value=True),mock.patch(
                'VLA.staging_r102.report.identity',return_value=ident):render()
            value=read(output/'LIVE_READINESS.json')['families']['gr00t_n15']
            self.assertTrue(value['live_training']);self.assertEqual(value['actual_trainer_status']['updates'],7)
            self.assertEqual(value['certified_actual_episodes'],0)

    def test_live_index_rejects_unrelated_pid_as_training(self):
        from .report import render
        with tempfile.TemporaryDirectory(prefix='r102_SYNTHETIC_') as tmp:
            base=Path(tmp);root=base/'R100';output=base/'staging'
            atomic_json(root/'gr00t_n15/WORKFLOW_STATUS.json',{'stage':'adapt2048_attempt00'})
            atomic_json(root/'gr00t_n15/adapt2048_attempt00/status.json',{'pid':123,'updates':7,'target_updates':20})
            with mock.patch('VLA.staging_r102.report.R100',root),mock.patch(
                'VLA.staging_r102.report.OUT',output),mock.patch(
                'VLA.staging_r102.report.alive',return_value=True),mock.patch(
                'VLA.staging_r102.report.identity',return_value={'command':['unrelated']}):render()
            self.assertFalse(read(output/'LIVE_READINESS.json')['families']['gr00t_n15']['live_training'])


if __name__=='__main__':unittest.main()
