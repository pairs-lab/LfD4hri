"""Expert-only collision-normal motion bound; NOT a VLA control fallback."""
import numpy as np
import pybullet as p
from VLA.robust_touch_teacher import RobustTouchTeacher
from VLA.collect_touch_replay_correction import ForceVerifiedTouchCollection


def surface_limited_motion(action, contacts):
    action=np.asarray(action).copy()
    scale=1.
    for contact in contacts:
        gap=float(contact[8])
        closing=-float(np.dot(action[:3],contact[7]))
        if closing > 0:
            allowance=.5*gap if gap > .0001 else .000020
            scale=min(scale,max(0.,allowance)/closing)
    action[:3] *= scale
    return action


class SurfaceLimitedTouchTeacher(RobustTouchTeacher):
    def predict(self, observation):
        action=super().predict(observation)
        if self.task.name=="touch" and self.phase=="descend":
            contacts=p.getClosestPoints(self.env.gripper.body,self.env.pick_id,distance=.01)
            action=surface_limited_motion(action,contacts)
        return action


class SurfaceLimitedTouchCollection(ForceVerifiedTouchCollection):
    @property
    def _last_success(self):
        # After first force-bearing contact, HOLD to obtain the second quality
        # sample, rather than deliberately applying another inward increment.
        return self.env._last_success and self.positive_contact()
