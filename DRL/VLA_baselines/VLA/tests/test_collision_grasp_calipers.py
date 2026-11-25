import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from VLA.collision_grasp_planner import minimum_caliper_grasp, plan_body_grasp


def box_points(width, length, angle=0.):
    corners = np.array([(x,y,z) for x in (-width/2,width/2)
                        for y in (-length/2,length/2) for z in (0.,.03)])
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    corners[:,:2] = corners[:,:2] @ rotation.T
    return np.concatenate((corners,corners))


class CollisionCaliperTest(unittest.TestCase):
    def test_rotated_box_finds_full_width_not_covariance_axis(self):
        points = box_points(.074,.105,.41)+[.08,-.65,.005]
        plan = minimum_caliper_grasp(points)
        self.assertAlmostEqual(plan["cross_section_width_m"],.074,places=10)
        self.assertTrue(np.allclose(plan["point_world"],[.08,-.65,.02]))
        axis = plan["jaw_axis_world"]
        self.assertAlmostEqual(abs(axis @ np.array([np.cos(.41),np.sin(.41)])),1.,places=10)

    def test_duplicate_vertex_density_cannot_change_support(self):
        points = box_points(.074,.105,.41)
        weighted = np.concatenate((points,np.repeat(points[:1],100,axis=0)))
        self.assertAlmostEqual(minimum_caliper_grasp(points)["cross_section_width_m"],
                               minimum_caliper_grasp(weighted)["cross_section_width_m"],places=12)

    def test_degenerate_and_nonfinite_collision_mesh_rejected(self):
        for points in (np.zeros((12,3)),np.full((12,3),np.nan),np.zeros((3,2))):
            with self.assertRaises(ValueError):
                minimum_caliper_grasp(points)

    def plan(self, points, gap=.0885, profile=None):
        origin = np.array([.08,-.65,.005])
        world = points+origin
        gripper = SimpleNamespace(width=lambda:gap,
            pad_positions=lambda:(np.array([-.05,0,0]),np.array([.05,0,0])))
        with patch("VLA.collision_grasp_planner.p.getAABB",return_value=(world.min(0),world.max(0))), \
             patch("VLA.collision_grasp_planner.p.getMeshData",return_value=(len(points),points)), \
             patch("VLA.collision_grasp_planner.p.getBasePositionAndOrientation",return_value=(origin,[0,0,0,1])):
            return plan_body_grasp(SimpleNamespace(gripper=gripper, observation_profile=profile or {}),123)

    def test_recovery_closed_stroke_is_not_maximum_open_capacity(self):
        plan = self.plan(box_points(.074,.085,np.pi/4), gap=.0064,
                         profile={"gripper_controller":"body_contact_v9"})
        self.assertEqual(plan["method"], "collision_minimum_caliper_full_support")
        self.assertGreater(plan["open_gap_m"], .088)

    def test_near_round_covariance_does_not_allow_unfitting_current_yaw(self):
        points = box_points(.074,.085,np.pi/4)
        eig = np.linalg.eigvalsh(np.cov(points[:,:2].T))
        self.assertLess(eig[1]/eig[0],1.6)
        plan = self.plan(points)
        self.assertEqual(plan["method"],"collision_minimum_caliper_full_support")
        self.assertAlmostEqual(plan["cross_section_width_m"],.074)
        self.assertGreater(plan["current_yaw_support_width_m"],plan["open_gap_m"])

    def test_small_round_footprint_preserves_existing_yaw(self):
        angles = np.linspace(0,2*np.pi,24,endpoint=False)
        points = np.array([[.03*np.cos(a),.03*np.sin(a),z] for a in angles for z in (0,.03)])
        plan = self.plan(points)
        self.assertEqual(plan["method"],"central_aabb_near_round")
        self.assertIsNone(plan["jaw_axis_world"])

    def test_oversized_full_footprint_fails_before_downward_push(self):
        with self.assertRaisesRegex(ValueError,"No full near-round collision footprint fits"):
            self.plan(box_points(.095,.10,.3))


if __name__ == "__main__":
    unittest.main()
