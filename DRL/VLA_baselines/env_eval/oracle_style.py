"""Presentation-oriented Oracle paths; no claim of a learned controller."""
from math import comb
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from env_eval.oracle_trajectory import minimum_jerk


class StyledPoseCurve:
    """C2 pose interpolation with arc-length timing and rounded 3D geometry."""

    def __init__(self, points, quaternions, duration, rotation_times=None):
        self.points = np.asarray(points, dtype=float)
        self.duration = float(duration)
        if self.duration <= 0 or not np.isfinite(self.points).all():
            raise ValueError('Finite points and positive duration required')
        times = np.linspace(0, 1, len(quaternions)) if rotation_times is None else rotation_times
        self.orientation = RotationSpline(times, Rotation.from_quat(quaternions))
        parameter = np.linspace(0, 1, 2001)
        path = self.geometry(parameter)
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        self.length = float(distances[-1])
        self.inverse_length = (CubicSpline(distances / distances[-1], parameter)
                               if self.length > 1e-8 else None)

    def geometry(self, parameter):
        parameter = np.asarray(parameter)
        degree = len(self.points) - 1
        weights = np.array([comb(degree, i) * parameter**i * (1-parameter)**(degree-i)
                            for i in range(degree+1)])
        return np.moveaxis(weights, 0, -1) @ self.points

    def sample(self, elapsed):
        clock = float(minimum_jerk(elapsed / self.duration))
        parameter = float(self.inverse_length(clock)) if self.inverse_length is not None else clock
        return self.geometry(np.clip(parameter, 0, 1)), self.orientation(clock).as_quat()


def offset_quaternion(quaternion, yaw=0., pitch=0., roll=0.):
    return (Rotation.from_euler('xyz', [roll, pitch, yaw]) * Rotation.from_quat(quaternion)).as_quat()


def blend_quaternion(start, goal, fraction):
    origin=Rotation.from_quat(start)
    turn=(origin.inv()*Rotation.from_quat(goal)).as_rotvec()
    return (origin*Rotation.from_rotvec(fraction*turn)).as_quat()


def styled_approach(start, goal, q0, q1, *, duration, side=.015, lean=.035):
    start, goal = np.asarray(start), np.asarray(goal)
    delta = goal-start
    shoulder = start + .32*delta
    shoulder[2] = max(start[2]-.015, goal[2]+.15)
    shoulder[0] += side
    entry = goal + np.array([side*.6, .008, .12])
    soft_entry = goal + np.array([side*.25, .003, .055])
    final_entry = goal + np.array([.001, 0., .020])
    curve = StyledPoseCurve([start, shoulder, entry, soft_entry, final_entry, goal],
                           [q0, q1], duration)
    origin = Rotation.from_quat(q0)
    turn = (origin.inv()*Rotation.from_quat(q1)).as_rotvec()
    def approach_orientation(clock):
        # A global cubic through a large initial turn and repeated terminal
        # poses can overshoot during contact. Finish a C2 geodesic turn before
        # the final approach, with a small, smoothly vanishing pitch accent.
        progress = float(minimum_jerk(np.clip(clock/.75, 0., 1.)))
        accent = lean*np.sin(np.pi*np.clip(clock/.82, 0., 1.))**4
        return Rotation.from_euler('y',accent)*origin*Rotation.from_rotvec(progress*turn)
    curve.orientation = approach_orientation
    return curve


def styled_carry(start, goal, quaternion, *, clearance, duration, side=.014, lean=.025, goal_quaternion=None):
    start, goal = np.asarray(start), np.asarray(goal)
    delta = goal-start
    horizontal = np.r_[delta[:2], 0.]
    height = max(float(clearance), start[2]+.095, goal[2]+.055)
    p1 = start + .055*horizontal + [side*.15, 0., .070]
    p2 = start + .34*horizontal + [0., side, 0.]
    p3 = goal - .30*horizontal + [0., side*.65, 0.]
    p4 = goal - .045*horizontal + [-side*.10, 0., .035]
    p2[2] = p3[2] = height+.065
    if np.linalg.norm(delta[:2]) < .01:
        p1[0] += side*.2
        p2[0] += side
        p3[0] += side*.7
    goal_quaternion=quaternion if goal_quaternion is None else goal_quaternion
    return StyledPoseCurve([start, p1, p2, p3, p4, goal],
                           [quaternion, offset_quaternion(blend_quaternion(quaternion,goal_quaternion,.45), yaw=lean*.7, pitch=lean),
                            offset_quaternion(blend_quaternion(quaternion,goal_quaternion,.85), yaw=-lean*.3, pitch=lean*.4), goal_quaternion],
                           duration, rotation_times=[0., .34, .7, 1.])


def styled_retract(start, goal, quaternion, *, duration=1.35):
    start, goal = np.asarray(start), np.asarray(goal)
    return StyledPoseCurve([start, start+[.005, 0., .035], goal+[-.008, .012, -.025], goal],
                           [quaternion, offset_quaternion(quaternion, pitch=-.025), quaternion], duration)


def styled_lift(start, goal, quaternion, *, duration=2.7, side=.012, lean=.025, goal_quaternion=None):
    """A small lateral sweep with monotonically increasing height; no overshoot."""
    start, goal = np.asarray(start), np.asarray(goal)
    delta = goal-start
    points = [start,
              start+.12*delta+[side*.12, side*.05, 0.],
              start+.33*delta+[side, side*.3, 0.],
              start+.64*delta+[side*.65, side*.15, 0.],
              start+.88*delta+[-side*.08, 0., 0.], goal]
    goal_quaternion=quaternion if goal_quaternion is None else goal_quaternion
    return StyledPoseCurve(points,
                           [quaternion, offset_quaternion(blend_quaternion(quaternion,goal_quaternion,.5), pitch=lean), goal_quaternion], duration)
