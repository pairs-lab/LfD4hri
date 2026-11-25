import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pybullet as p
from env_eval.camera_editions import SETTINGS,candidates,choose_layout,rectangles,compose_scene
from env_eval.handeye_camera import handeye_geometry


class CameraEditionTests(unittest.TestCase):
    def setUp(self):self.settings=json.loads(SETTINGS.read_text())

    def test_handeye_remains_rigid_under_translation_and_rotation(self):
        env=SimpleNamespace(robot_id=5,robot_spec=SimpleNamespace(mount_link=7))
        rotation=p.getQuaternionFromEuler((.4,-.7,1.1));position=(.3,-.5,.8)
        state=(position,rotation,None,None,(99,99,99),(0,0,0,1))
        with patch('env_eval.handeye_camera.p.getLinkState',return_value=state):
            cfg=handeye_geometry(env,self.settings['handeye'])
        r=np.array(p.getMatrixFromQuaternion(rotation)).reshape(3,3)
        np.testing.assert_allclose(cfg['eye'],position+r@self.settings['handeye']['eye_in_tool'],atol=1e-7)
        np.testing.assert_allclose(cfg['target'],position+r@self.settings['handeye']['target_in_tool'],atol=1e-7)
        np.testing.assert_allclose(cfg['up'],r@self.settings['handeye']['up_axis_in_tool'],atol=1e-7)

    def test_stacked_preserves_sensor_aspect_and_order(self):
        front,eye=rectangles('front_handeye',.24,'right',self.settings)
        self.assertEqual(front[1],eye[1]);self.assertEqual(front[3:],eye[3:])
        self.assertGreater(eye[2],front[2]+front[4]);self.assertLess(eye[2]+eye[4],1080)
        self.assertLess(abs(front[3]/front[4]-16/9),.02)

    def test_layout_checks_later_frames_before_locking_corner(self):
        cs=candidates(self.settings);rows=[]
        for frame in range(3):
            costs={key:[100 if frame==2 and c['side']=='left' else 0,0 if c['side']=='left' else 50]
                   for key,c in cs.items()}
            rows.append({'wide':[{'candidate_costs':costs}]})
        chosen=choose_layout(rows,'front_handeye',1,self.settings)
        self.assertEqual(chosen['side'],'right');self.assertEqual(chosen['fraction'],.24)

    def test_original_movies_are_protected_before_any_composition(self):
        with TemporaryDirectory() as temporary:
            target=Path(temporary);(target/'audit.json').write_text('{}')
            movie=target/'view_1.mp4';movie.write_bytes(b'original movie bytes')
            with self.assertRaises(FileExistsError):compose_scene(target/'missing_source',target,'front_handeye',self.settings)
            self.assertEqual(movie.read_bytes(),b'original movie bytes');self.assertFalse((target/'layout.json').exists())

    def test_changed_settings_cannot_silently_reuse_completed_edition(self):
        with TemporaryDirectory() as temporary:
            target=Path(temporary);(target/'camera_audit.json').write_text('{"pass":true}')
            old=json.loads(json.dumps(self.settings));old['handeye']['fov_degrees']=75
            (target/'camera_settings.json').write_text(json.dumps(old))
            with self.assertRaises(FileExistsError):compose_scene(target/'missing_source',target,'handeye',self.settings)


if __name__=='__main__':unittest.main()
