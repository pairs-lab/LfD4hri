"""Non-mutating contracts for temporal figure selection and provenance."""
import copy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from env_eval.case_demon.qualitative_figure import (
    MAIN,CASES,PILOT,OUTPUT,primary_event,project,selection,inventory,camera,decode)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={c['scene_id']:c for c in inventory(CASES)}

    def test_main_subset_and_four_groups(self):
        self.assertTrue(set(MAIN)<=set(CASES))
        self.assertEqual({self.cases[s]['group'] for s in MAIN},{1,2,3,4})
        self.assertEqual(len(CASES),12)

    def test_same_initial_world_and_crop(self):
        for case in self.cases.values():
            self.assertTrue(case['initial_world_pair_verified'])
            self.assertEqual(len(case['crop']),4)
            self.assertGreater(case['crop'][2],case['crop'][0])

    def test_monotone_distinct_event_frames(self):
        for case in self.cases.values():
            for method in case['methods']:
                picked=method['selection'];ids=[r['index'] for r in picked['main']]
                self.assertEqual(len(set(ids)),3)
                self.assertEqual(ids,sorted(ids))
                self.assertEqual(len(picked['detail_indices']),6)
                self.assertTrue(set(ids)<=set(picked['detail_indices']))

    def test_drop_requires_actual_prior_attachment(self):
        method=self.cases[3]['methods'][0];frames=copy.deepcopy(method['frames'])
        for row in frames:row['correct_attached']=False
        with self.assertRaises(ValueError):selection(frames,method['evidence'])

    def test_no_drop_label_for_empty_lift(self):
        method=self.cases[10]['methods'][1]
        self.assertEqual(method['selection']['outcome'],'Empty lift')
        self.assertEqual(method['evidence']['held_frames'],0)

    def test_slam_needs_velocity_force_gate(self):
        method=self.cases[12]['methods'][0];e=copy.deepcopy(method['evidence'])
        e['events'][0]['ground_slam_force_velocity_gate_pass']=False
        with self.assertRaises(ValueError):selection(method['frames'],e)

    def test_no_success_from_failed_outcome(self):
        for case in self.cases.values():
            for method in case['methods']:
                self.assertEqual(method['selection']['success'],method['evidence']['full_put_complete'])

    def test_success_includes_unheld_placed_state(self):
        for case in self.cases.values():
            for method in case['methods']:
                if not method['selection']['success']:continue
                i=method['selection']['main'][-1]['index']
                self.assertFalse(method['frames'][i]['correct_attached'])

    def test_cinematic_camera_optical_center(self):
        for view in (1,2):
            result=project(camera(view)['target'],camera(view),1920,1080)
            self.assertAlmostEqual(float(result[0]),960)
            self.assertAlmostEqual(float(result[1]),540)

    def test_target_miss_not_memory_loss(self):
        method=self.cases[13]['methods'][1]
        self.assertEqual(method['selection']['outcome'],'Target miss')
        self.assertFalse(method['evidence']['cognitive_misidentification_or_memory_loss_proven'])

    def test_empty_video_is_rejected_even_if_timing_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'blank.mp4'
            stream=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),30.,(64,64))
            for _ in range(3):stream.write(np.full((64,64,3),(170,175,190),np.uint8))
            stream.release()
            with self.assertRaisesRegex(ValueError,'Constant/empty'):
                decode(path,[0,2],3)

    def test_natural_recovery_has_hold_drop_reacquisition_and_stable_release(self):
        root=OUTPUT/'additional_recovery_source'
        if not (root/'XVLA_render_v2/render_complete.json').exists():
            self.skipTest('Optional recovery source has not been generated')
        case=inventory((14,),root)[0];method=case['methods'][1]
        self.assertEqual(method['selection']['outcome'],'Recovered success')
        before,dropped,final=[q['index'] for q in method['selection']['main']]
        self.assertTrue(method['frames'][before]['correct_attached'])
        self.assertFalse(method['frames'][dropped]['correct_attached'])
        self.assertTrue(any(f['correct_attached'] for f in method['frames'][dropped:final]))
        self.assertFalse(method['frames'][final]['correct_attached'])
        self.assertGreater(method['frames'][before]['wide'][1]['object_pixels'],0)


if __name__=='__main__':unittest.main()
