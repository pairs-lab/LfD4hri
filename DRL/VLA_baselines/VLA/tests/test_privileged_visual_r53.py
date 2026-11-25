import unittest
import tempfile
import numpy as np

from VLA.privileged_visual_r53.scene_map import render,geometry,project,TARGET,task_from_instruction


def measured_state():
    r=np.zeros(64,np.float32)
    r[7:10]=[0.,-.55,.4]
    r[11:14]=[.12,.1,-.35]
    r[14:17]=[-.20,.10,.15]
    r[18:24]=[-1,0,0,0,1,0]
    r[24:27]=[.045,.045,.05]
    r[27]=.08
    return r


class PrivilegedVisualTests(unittest.TestCase):
    def test_global_prefix_selection_does_not_pool_per_task_winners(self):
        from VLA.privileged_visual_r53.run import global_schedule
        from VLA.tests.test_feedback_r52 import FeedbackRepairTests
        helper=FeedbackRepairTests()
        with tempfile.TemporaryDirectory() as folder:
            paths=[helper.make_progress(folder,4,{'move':True}),
                   helper.make_progress(folder,6),
                   helper.make_progress(folder,30,{'touch':True,'pick':True})]
            result=global_schedule(paths,'xvla')
            self.assertEqual({r['selected_prefix'] for r in result['tasks'].values()},{30})

    def test_task_semantics_use_original_language_not_a_hidden_phase(self):
        for text,task in [('Touch the block','touch'),('Pick the pear up','pick'),
                          ('Move the egg left','move'),('Put the apple in the bowl','pick_place')]:
            self.assertEqual(task_from_instruction(text),task)
        with self.assertRaises(ValueError):task_from_instruction('')

    def test_current_and_goal_centers_are_exact_affine_geometry_not_EE_waypoints(self):
        r=measured_state()
        _,pad,target,goal,_=geometry(r,'move')
        np.testing.assert_allclose(target,pad+r[11:14])
        np.testing.assert_allclose(goal-target,r[14:17],atol=1e-8)

    def test_two_world_axes_are_not_flipped(self):
        center=project([0,-.5])
        self.assertGreater(project([.05,-.5])[0],center[0])
        self.assertLess(project([0,-.45])[1],center[1])

    def test_renderer_is_deterministic_and_does_not_mutate_sensor_state(self):
        r=measured_state();before=r.copy()
        a,b=render(r,'move'),render(r,'move')
        self.assertEqual(a.shape,(384,384,3));self.assertEqual(a.dtype,np.uint8)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(r,before)

    def test_requested_target_is_visually_grounded_at_the_correct_world_point(self):
        r=measured_state();_,_,target,_,_=geometry(r,'move')
        u,v=project(target[:2])
        np.testing.assert_array_equal(render(r,'move')[v,u],TARGET)

    def test_contacts_and_wrong_attachment_have_distinct_visible_feedback(self):
        r=measured_state();a=render(r,'move')
        r[30]=r[31]=1.
        b=render(r,'move')
        self.assertTrue(np.any(a!=b))

    def test_invalid_or_incomplete_raw_observation_is_not_silently_repaired(self):
        with self.assertRaises(ValueError):render(np.zeros(63),'move')
        r=measured_state();r[0]=np.nan
        with self.assertRaises(ValueError):render(r,'move')

    def test_touch_goal_is_surface_not_an_object_lift_destination(self):
        r=measured_state();_,_,target,goal,_=geometry(r,'touch')
        np.testing.assert_allclose(goal[:2],target[:2])
        self.assertAlmostEqual(goal[2]-target[2],r[26]/2)


if __name__=='__main__':unittest.main()
