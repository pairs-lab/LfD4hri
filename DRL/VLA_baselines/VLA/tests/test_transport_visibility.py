import unittest
import numpy as np
from VLA.audit_transport_visibility import transport_visibility


class TransportVisibilityTest(unittest.TestCase):
    def arrays(self):
        actions = np.zeros((20, 7))
        actions[15:19, 0] = .03
        return {"canonical_actions": actions, "next_grasped_object": np.ones(20, bool),
                "next_verified_grasp": np.ones(20, bool),
                **{camera+"_"+role+"_mask": np.ones((20, 8, 8), bool)
                   for camera in ("front", "wrist") for role in ("pick", "place")}}

    def test_hidden_goal_during_pickup_is_not_transport_failure(self):
        arrays = self.arrays()
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][5:10] = False
        result, failures = transport_visibility(arrays, {"place_name": "gray bowl"})
        self.assertFalse(failures)
        self.assertEqual(result["roles"]["place"]["visible_xy_transport_fraction"], 1)

    def test_front_only_policy_cannot_claim_wrist_visibility(self):
        arrays = self.arrays()
        arrays["front_place_mask"][15:19] = False
        _, multi = transport_visibility(arrays, {"place_name": "gray bowl"})
        _, front = transport_visibility(arrays, {"place_name": "gray bowl"}, camera_views=("front",))
        self.assertFalse(multi)
        self.assertTrue(front)
        with self.assertRaises(ValueError):
            transport_visibility(arrays, {"place_name": "gray bowl"}, camera_views=())

    def test_global_average_cannot_conceal_one_hidden_carry_step(self):
        arrays = self.arrays()
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][17] = False
        result, failures = transport_visibility(arrays, {"place_name": "gray bowl"})
        self.assertTrue(failures)
        self.assertEqual(result["roles"]["place"]["visible_whole_episode_fraction"], .95)
        self.assertEqual(result["roles"]["place"]["visible_xy_transport_fraction"], .75)

    def test_virtual_goal_has_no_physical_mask(self):
        arrays = self.arrays()
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][:] = False
        result, failures = transport_visibility(arrays, {"place_name": "top left corner"})
        self.assertFalse(failures)
        self.assertFalse(result["roles"]["place"]["physical_body"])

    def test_one_frame_protocol_is_opt_in_and_never_allows_persistent_gaps(self):
        arrays = self.arrays()
        arrays["canonical_actions"][10:19, 0] = .03
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][17] = False
        _, strict = transport_visibility(arrays, {"place_name": "gray bowl"})
        self.assertTrue(strict)
        result, bounded = transport_visibility(arrays, {"place_name": "gray bowl"}, max_transient_frames=1)
        self.assertFalse(bounded)
        self.assertTrue(result["roles"]["place"]["single_frame_outage_disclosed_not_full_observability"])
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][18] = False
        _, failed = transport_visibility(arrays, {"place_name": "gray bowl"}, max_transient_frames=1)
        self.assertTrue(failed)

    def test_long_carry_average_cannot_hide_persistent_gap_in_bounded_protocol(self):
        arrays = self.arrays()
        arrays = {key: np.concatenate([value]*3) for key, value in arrays.items()}
        arrays["canonical_actions"][1:, 0] = .03
        for camera in ("front", "wrist"):
            arrays[camera+"_place_mask"][20:22] = False
        result, failed = transport_visibility(arrays, {"place_name": "gray bowl"}, max_transient_frames=1)
        self.assertGreater(result["roles"]["place"]["visible_xy_transport_fraction"], .95)
        self.assertEqual(result["roles"]["place"]["longest_consecutive_invisible_transport_frames"], 2)
        self.assertTrue(failed)
