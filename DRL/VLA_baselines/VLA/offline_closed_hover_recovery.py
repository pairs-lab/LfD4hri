"""Collection-only closed/empty mid-height starts; no runtime policy helper.

Ordinary guarded actions, no object/joint teleport or model inputs added. This
adds measured closed-empty -> OPEN labels, not natural-slip/wrong-object or
near-body recovery certification. Startup controls are NEVER expert labels.
"""
import numpy as np
import pybullet as p


def closed_hover_start(env):
    from VLA.common.physics_safety import surface_clearance
    from VLA.staged_acquisition_teacher import INITIAL_VERTICAL_STEP_M
    height = max(.22, max(p.getAABB(body)[1][2] for body in env.object_ids.values())+.08)
    controls = []
    initial = {name: np.asarray(p.getBasePositionAndOrientation(body)[0]) for name, body in env.object_ids.items()}
    def execute(action):
        _, _, done, info = env.step(action)
        controls.append(np.asarray(info.get("executed_canonical_action", action), np.float32))
        if done or env.gripper.grasped_object is not None or p.getContactPoints(bodyA=env.gripper.body):
            raise ValueError("Closed-hover startup touched/held/terminated; preserve failure")
    for _ in range(35):
        dz = height-float(env.gripper.grasp_center()[2])
        if abs(dz) < .006:
            break
        execute(np.asarray([0, 0, np.clip(.8*dz, -INITIAL_VERTICAL_STEP_M, INITIAL_VERTICAL_STEP_M), 0, 0, 0, -1], np.float32))
    else:
        raise ValueError("Bounded closed-hover vertical startup did not converge")
    for _ in range(15):
        execute(np.asarray([0, 0, 0, 0, 0, 0, 1], np.float32))
        if env.measured_gripper_closure() >= .95:
            break
    else:
        raise ValueError("CLOSE request did not reach actual95% FK closure")
    if surface_clearance(env) < .003:
        raise ValueError("Closed-hover start has insufficient table clearance")
    points = [c for body in env.object_ids.values()
              for c in p.getClosestPoints(env.gripper.body, body, distance=.25)]
    clearance = min((c[8] for c in points), default=.25)
    if clearance < .003:
        raise ValueError("Closed-hover start lacks positive object clearance")
    return controls, {"kind": "mid_height_contact_free_closed_empty_v1",
        "actual_closure": env.measured_gripper_closure(), "grasped_object": None,
        "contact_count": 0, "surface_clearance_m": float(surface_clearance(env)),
        "object_clearance_m": float(clearance), "hover_height_m": height,
        "source_object_translation_drift_m": {name: float(np.linalg.norm(
            np.asarray(p.getBasePositionAndOrientation(body)[0])-initial[name])) for name, body in env.object_ids.items()},
        "object_poses_teleported": False, "startup_controls_are_training_labels": False,
        "near_body_or_natural_failure_recovery_claimed": False}


class ClosedHoverRecoveryReset:
    def __init__(self, base):
        self.base = base
        self.env = getattr(base, "env", base)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        self.base.reset(task)
        self.startup_actions, self.startup_evidence = closed_hover_start(self.env)
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        position = np.asarray(p.getBasePositionAndOrientation(self.env.pick_id)[0])
        self.env.initial_pick_z, self.env.initial_pick_position = position[2], position
        self.env.initial_pick_top, self.env.initial_pick_bottom = self.env._pick_top(), self.env._pick_bottom()
        return self.env.observe()
