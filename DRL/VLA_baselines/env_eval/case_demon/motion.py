"""Continuous joint-space plans: C2 splines through a few sparse IK keyframes."""
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

MAX_VELOCITY, MAX_ACCELERATION = 1.6, 2.8  # rad/s, rad/s^2


class JointPlan:
    """Clamped spline through keyframes; time stretches until joint limits hold.

    v0 is the actual velocity at the start, so a new plan continues the previous
    motion without a kink. The end velocity is zero. degree=3 is C2 (jerk jumps
    at knots, used for the CASE_demon videos); degree=5 is C4 with zero end
    acceleration, so jerk is continuous, as in minimum-jerk motion.
    """
    def __init__(self, knots, times, v0=None, max_velocity=MAX_VELOCITY, max_acceleration=MAX_ACCELERATION,
                 degree=3):
        knots = np.asarray(knots, dtype=float)
        times = np.asarray(times, dtype=float)
        v0 = np.zeros(knots.shape[1]) if v0 is None else np.asarray(v0, dtype=float)
        for _ in range(12):
            zero = np.zeros(knots.shape[1])
            if degree == 5:
                spline = make_interp_spline(times, knots, k=5, bc_type=([(1, v0), (2, zero)], [(1, zero), (2, zero)]))
            else:
                spline = CubicSpline(times, knots, bc_type=((1, v0), (1, zero)))
            grid = np.linspace(times[0], times[-1], 400)
            velocity = float(np.abs(spline(grid, 1)).max())
            acceleration = float(np.abs(spline(grid, 2)).max())
            scale = max(1., velocity/max_velocity, np.sqrt(acceleration/max_acceleration))
            if scale <= 1.001:
                break
            times = times[0]+(times-times[0])*scale*1.01
        self.spline, self.times = spline, times
        self.duration = float(times[-1]-times[0])
        self.peak_velocity, self.peak_acceleration = velocity, acceleration

    def sample(self, elapsed):
        t = self.times[0]+elapsed
        return self.spline(t), self.spline(t, 1)
