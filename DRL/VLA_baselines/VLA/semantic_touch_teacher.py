"""Expert-only gentle near-surface touch, consistent with first-hit semantics.

Near-surface success is NOT described as force-verified physical contact.
The original raw reward predicate is additionally required and remains intact.
"""
import pybullet as p
from VLA.collect_paired_grounding_candidate import FrozenRetargetReset
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher

SURFACE_MARGIN_M=.0002
BOUNDARY_FORCE_QUALITY_CAP_N=5.


class NearSurfaceTouchCollection(FrozenRetargetReset):
    def terminal_geometry(self):
        closest=p.getClosestPoints(self.env.gripper.body,self.env.pick_id,distance=.025)
        actual=p.getContactPoints(self.env.gripper.body,self.env.pick_id)
        return min((c[8] for c in closest),default=None),max((c[9] for c in actual),default=0.)

    @property
    def _last_success(self):
        gap,force=self.terminal_geometry()
        return self.env._last_success and gap is not None and gap<=SURFACE_MARGIN_M and force<=BOUNDARY_FORCE_QUALITY_CAP_N

    def step(self,action):
        observation,reward,done,info=self.env.step(action)
        gap,force=self.terminal_geometry()
        self.last_surface_gap=gap
        self.last_contact_force=force
        self.last_quality_failure="boundary_normal_force_above_5N" if force>BOUNDARY_FORCE_QUALITY_CAP_N else None
        info=dict(info,expert_terminal_surface_gap_m=gap,expert_terminal_max_normal_force_n=force)
        if self.last_quality_failure:
            return observation,reward,True,dict(info,success=False,expert_quality_failure=self.last_quality_failure)
        acceptable=(info["strict_raw_success"] and gap is not None and gap<=SURFACE_MARGIN_M)
        if not acceptable:
            info["success"]=False
            if not info.get("safety_violation") and self.env.step_count<self.env.task.max_steps:
                done=False
        return observation,reward,done,info
