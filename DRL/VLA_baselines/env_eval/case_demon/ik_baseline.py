"""Baseline: Cartesian straight lines between the same waypoints, IK every tick.

Like an industrial MoveL without blending: constant tool speed on each segment,
a full stop at every waypoint, tool held vertical. Contact detection, suction,
joint controller, gains, holds and scoring are shared with the Oracle, so only
the way the trajectory is generated differs.
"""
from dataclasses import asdict

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation, Slerp

from env_eval.case_demon.rollout import PHYSICS_HZ, Failed, Rollout


def straight_line(start, end, speed, q_start=None, q_end=None):
    """Constant-speed samples (one per physics tick) from start to end, inclusive."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    ticks = max(1, int(np.ceil(np.linalg.norm(end-start)/speed*PHYSICS_HZ)))
    fractions = np.arange(1, ticks+1)/ticks
    points = start+np.outer(fractions, end-start)
    if q_start is None:
        return points, None
    slerp = Slerp([0, 1], Rotation.from_quat([q_start, q_end]))
    return points, slerp(fractions).as_quat()


class IKRollout(Rollout):
    def __init__(self, env, speed):
        super().__init__(env)
        self.speed = speed

    def move(self, end, quat, phase, stop=None, start_quat=None):
        e = self.env
        self.phase = phase
        points, quats = straight_line(e.tip(), end, self.speed, start_quat, quat)
        q_previous = e.q.copy()
        for index, point in enumerate(points):
            q = e.solve_ik(point, quat if quats is None else quats[index], seed=e.q)
            e.tick(q, (q-q_previous)*PHYSICS_HZ)
            q_previous = q
            self.monitor()
            self.measure_firmness()
            if stop and stop():
                return True
        return False

    def run(self):
        e = self.env
        result = {'scene': asdict(e.scene), 'controller': 'straight-line IK baseline (MoveL, stop at waypoints)'}
        try:
            down = e.down_quaternion()
            start = e.tip()
            home_quat = e.link_pose()[1]
            surface = e.surface_point()
            def touching():
                points = e.tip_contacts()
                return bool(points) and all(c[2] == e.pick for c in points)
            self.move(surface+[0, 0, .06], down, 'approach', start_quat=home_quat)
            self.move(surface+[0, 0, -.006], down, 'descend', touching)
            if not touching():
                raise Failed('Suction tip never contacted the target')
            if not e.grip():
                raise Failed('Contact did not create a suction attachment')
            self.attached_tick = e.ticks
            self.hold(.28, 'attach')  # same dwell as the Oracle's settle-and-seal
            self.object_reference = self._object_relative()

            tip = e.tip()
            obj = np.array(p.getBasePositionAndOrientation(e.pick)[0])
            low = np.array(p.getAABB(e.pick)[0])
            bowl_low, bowl_high = map(np.asarray, p.getAABB(e.place))
            goal_xy = (bowl_low[:2]+bowl_high[:2])/2
            goal = np.r_[goal_xy, bowl_high[2]+(obj[2]-low[2])+.012]
            target = tip+goal-obj
            tops = [p.getAABB(b)[1][2] for b in e.ids.values() if b != e.pick]
            safe_z = max(max(tops)+.05+(tip[2]-low[2]), tip[2]+.11)  # Oracle's apex rule
            self.move(np.r_[tip[:2], safe_z], down, 'carry')
            self.move(np.r_[target[:2], safe_z], down, 'carry')
            self.move(target, down, 'lower')
            self.phase = 'release'
            e.ungrip()
            self.hold(.12, 'release')
            self.move(target+[0, 0, .17], down, 'retract')
            self.move(start, home_quat, 'retract', start_quat=down)
            self.hold(.7, 'settle')
            result.update(self.outcome(goal_xy))
        except (Failed, RuntimeError) as error:
            result.update(success=False, safety_pass=False, overall_pass=False, failure=str(error))
        return self.finish(result)
