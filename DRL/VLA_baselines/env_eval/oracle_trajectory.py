"""Continuous privileged pose trajectories, independent of learned policies."""
from __future__ import annotations

from math import comb
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def minimum_jerk(t):
    t = np.clip(t, 0., 1.)
    return t**3 * (10. + t * (-15. + 6. * t))


class PoseCurve:
    """Bezier geometry and quintic timing; orientation follows shortest SLERP."""

    def __init__(self, control_points, start_quaternion, end_quaternion, duration):
        self.points = np.asarray(control_points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 2:
            raise ValueError("Bezier control points must be Nx3")
        if not np.isfinite(self.points).all() or duration <= 0:
            raise ValueError("Finite points and positive duration required")
        self.duration = float(duration)
        self.orientation = Slerp([0., 1.], Rotation.from_quat([start_quaternion, end_quaternion]))

    def sample(self, elapsed):
        s = float(minimum_jerk(elapsed / self.duration))
        degree = len(self.points) - 1
        weights = [comb(degree, i) * s**i * (1-s)**(degree-i) for i in range(degree+1)]
        return np.asarray(weights) @ self.points, self.orientation(s).as_quat()


def approach_curve(start, goal, q0, q1, duration=4., bend=.02):
    """A single curve ends tangent to a careful downward approach."""
    start, goal = np.asarray(start), np.asarray(goal)
    first = start + .25 * (goal-start)
    first[2] = max(start[2], goal[2]+.20)
    first[0] += bend
    before = goal + [bend, 0., .18]
    return PoseCurve([start, first, before, goal], q0, q1, duration)


def carry_curve(start, goal, quaternion, clearance, duration=5., bend=.015):
    """One lift/transport/descent arc, with no intermediate stop or corner."""
    start, goal = np.asarray(start), np.asarray(goal)
    # Vertical tangents near contact; horizontal travel grows continuously.
    apex = max(clearance, start[2]+.22, goal[2]+.18)
    c1, c2 = start.copy(), goal.copy()
    c1[2] = c2[2] = apex + .10
    c1[1] += bend
    c2[1] += bend
    return PoseCurve([start, c1, c2, goal], quaternion, quaternion, duration)
