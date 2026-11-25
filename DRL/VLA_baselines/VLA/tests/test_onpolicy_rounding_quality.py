import unittest
from unittest.mock import patch

from VLA.audit_onpolicy_sensor_rounding import correction_environment


class OnPolicyRoundingQualityTests(unittest.TestCase):
    @patch("VLA.audit_onpolicy_sensor_rounding.NearSurfaceTouchCollection")
    def test_touch_preserves_the_collection_quality_wrapper(self, wrapper):
        env = object()
        self.assertIs(correction_environment(env,"touch"),wrapper.return_value)
        wrapper.assert_called_once_with(env,None,None)

    @patch("VLA.audit_onpolicy_sensor_rounding.NearSurfaceTouchCollection")
    def test_grasp_tasks_keep_original_env(self, wrapper):
        env = object()
        for task in ("pick","move","pick_place"):
            self.assertIs(correction_environment(env,task),env)
        wrapper.assert_not_called()
