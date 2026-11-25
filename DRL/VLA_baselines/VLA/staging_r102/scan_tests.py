"""Real NPZ/receipt I/O on tiny synthetic data; no robot success trials."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest import mock
import numpy as np
from .common import atomic_json,read
from .depth_scan import main
from VLA.common.provenance import sha256_file


class ScanResume(unittest.TestCase):
    def test_middle_camera_fault_not_hidden_by_reset_terminal_samples(self):
        from .geometry_scan import camera_histories
        views=np.broadcast_to(np.eye(4).reshape(16),(5,16)).copy()
        P=np.zeros((4,4));P[0,0]=P[1,1]=1;P[3,2]=-1;P[2,2]=-1.01;P[2,3]=-.02
        projection=P.T.reshape(16)
        self.assertEqual(camera_histories(views,projection,5)['frames'],5)
        bad=views.copy();bad[2,0]=-1
        with self.assertRaises(ValueError):camera_histories(bad,projection,5)
        bad=np.broadcast_to(projection,(5,16)).copy();bad[2,0]=np.nan
        with self.assertRaises(ValueError):camera_histories(views,bad,5)

    def test_full_scan_resume_and_changed_payload_rejection(self):
        with tempfile.TemporaryDirectory(prefix='r102_SYNTHETIC_NOT_DATASET_') as tmp:
            root=Path(tmp);native=root/'R100';output=root/'staging';blob=root/'synthetic.npz'
            def payload(value):
                np.savez(blob,front_depth_z_m=np.full((1,384,384),value,np.float32),
                    source_frame_indices=np.array([0]),view_world_to_OpenGL_camera=np.eye(4).reshape(1,16),
                    projection_OpenGL=np.eye(4).reshape(16))
            payload(.5);digest=sha256_file(blob)
            sources=[{'frames':1,'source_sha256':'a'*64,'source_kind':'SYNTHETIC_NOT_DATASET',
                'task':task,'pick_name':'synthetic_target'+str(i)}
                for task in ('touch','pick','move','pick_place') for i in range(19)]
            rows=[{'file':str(blob),'sha256':digest,'source_sha256':'a'*64,'frames':1} for _ in sources]
            index=native/'falcon_esm/DEPTH_INDEX.json';atomic_json(index,rows)
            atomic_json(native/'falcon_esm/DEPTH_ADMISSION.json',{'passed':True,'episodes':76,'frames':76,
                'depth_index_sha256':sha256_file(index)})
            with mock.patch('VLA.staging_r102.depth_scan.OUT',output),mock.patch(
                'VLA.staging_r102.depth_scan.R100',native),mock.patch(
                'VLA.staging_r102.depth_scan.os.nice'),mock.patch(
                'VLA.refinement_r99.data.Data',return_value=SimpleNamespace(rows=sources)),mock.patch('builtins.print'):
                main()
                proof=read(output/'FULL_DEPTH_SCAN.json')
                self.assertTrue(proof['passed']);self.assertEqual(proof['frames'],76)
                with mock.patch('VLA.staging_r102.depth_scan.np.load',side_effect=AssertionError('must reuse valid receipts')):
                    main()
                payload(.6)
                with self.assertRaises(ValueError):main()


if __name__=='__main__':unittest.main()
