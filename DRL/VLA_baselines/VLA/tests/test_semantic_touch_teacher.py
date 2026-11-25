from types import SimpleNamespace
import unittest
from unittest.mock import patch
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection


class SemanticTouchTeacherTest(unittest.TestCase):
    def wrapper(self,strict=True,step=4,safety=None):
        info={"success":strict,"raw_success":strict,"strict_raw_success":strict,"safety_violation":safety}
        env=SimpleNamespace(_last_success=strict,step_count=step,task=SimpleNamespace(max_steps=80),
            step=lambda action:(None,1.,True,info))
        return NearSurfaceTouchCollection(env,None,None),info

    @patch.object(NearSurfaceTouchCollection,"terminal_geometry",return_value=(.00015,0.))
    def test_near_surface_is_explicit_proximity_not_required_force(self,geometry):
        wrapper,original=self.wrapper()
        self.assertTrue(wrapper.step(None)[3]["success"])
        self.assertTrue(wrapper._last_success)
        self.assertTrue(original["raw_success"])

    @patch.object(NearSurfaceTouchCollection,"terminal_geometry",return_value=(.00015,0.))
    def test_cannot_promote_original_strict_failure(self,geometry):
        wrapper,_=self.wrapper(strict=False)
        self.assertFalse(wrapper.step(None)[3]["success"])

    @patch.object(NearSurfaceTouchCollection,"terminal_geometry",return_value=(.0008,0.))
    def test_further_cache_contact_continues_but_never_overrides_safety_or_budget(self,geometry):
        wrapper,_=self.wrapper()
        self.assertFalse(wrapper.step(None)[2])
        for kwargs in ({"step":80},{"safety":"mount_drift"}):
            wrapper,_=self.wrapper(**kwargs)
            self.assertTrue(wrapper.step(None)[2])

    @patch.object(NearSurfaceTouchCollection,"terminal_geometry",return_value=(-.0001,20.))
    def test_force_quality_failure_is_preserved_without_modifying_raw_reward(self,geometry):
        wrapper,_=self.wrapper()
        _,_,done,info=wrapper.step(None)
        self.assertTrue(done)
        self.assertFalse(info["success"])
        self.assertTrue(info["raw_success"])
