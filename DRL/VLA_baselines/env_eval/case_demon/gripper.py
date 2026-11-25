"""Robotiq 2F-85 on all five arms: environment, shared task geometry, Ours and IK.

Both controllers receive the same task geometry from task_geometry(): grasp
point and jaw yaw, place point, safe transport height. They differ only in how
the trajectory between those waypoints is generated.
"""
from dataclasses import asdict

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from VLA.common.body_environment import DERIVED_ASSETS
from VLA.common.body_gripper import BodyContactRobotiq
from env_eval.case_demon.ik_baseline import IKRollout
from env_eval.case_demon.motion import JointPlan
from env_eval.case_demon.rollout import PHYSICS_HZ, TABLE_TOP, Failed, Rollout, SuctionEnv, world_rotation_aligning

# URDF flange links (fixed, massless) carry the mount; the last moving link's
# mesh gives the physical flange face the gripper must sit on.
FLANGE = {'Panda': 7, 'KUKA': 7, 'Kinova': 7, 'UR5e': 9, 'UF850': 6}
WRIST = {'Panda': 6, 'KUKA': 6, 'Kinova': 6, 'UR5e': 7, 'UF850': 5}
COUPLER_DEPTH = .07  # stock UR mount: Robotiq link-0 origin this far past tool0
SEAT_OVERLAP = .004  # coupler top tucked this far into the wrist to hide the seam
CLOSE_S, GRASP_TIMEOUT_S, OPEN_S, IDLE_S = .55, 1.2, .40, 1.10  # release + idle = 1.5 s


class GripperEnv(SuctionEnv):
    def _attach_tool(self, dynamic):
        self.flange = FLANGE[self.spec.label]
        self.gripper = BodyContactRobotiq(self.robot, self.flange, DERIVED_ASSETS, dynamic,
                                          assist_grasp=True, mimic_gear=False)
        if self.spec.label == 'UF850':
            # The repository's UF850 mount fix (utils/Robotiq2F85UF850.py).
            p.changeConstraint(self.gripper.mount_constraint, maxForce=50000., erp=.9)
            p.changeDynamics(self.robot, self.flange, mass=.1, localInertiaDiagonal=[.1, .1, .1])
            p.changeDynamics(self.gripper.body, -1, mass=.2, localInertiaDiagonal=[.0002, .0002, .00018],
                             linearDamping=0., angularDamping=0.)
        self._settle(120)
        # Seat the coupler on the flange face: its top just overlaps the
        # wrist's lowest point (the stock UR offset leaves a gap on other arms).
        wrist_low = p.getAABB(self.robot, WRIST[self.spec.label])[0][2]
        coupler_top = p.getAABB(self.gripper.body, -1)[1][2]
        self.mount_shift_m = float(wrist_low+SEAT_OVERLAP-coupler_top)
        p.changeConstraint(self.gripper.mount_constraint, jointChildPivot=(0, 0, -COUPLER_DEPTH+self.mount_shift_m))
        self._settle(240)
        self.closure_profile = self._closure_profile()

    def _lowest(self):
        return min(p.getAABB(self.gripper.body, link)[0][2] for link in range(-1, p.getNumJoints(self.gripper.body)))

    def _closure_profile(self):
        """Robotiq fingertips sweep down as they close: measure (width, pad-centre
        drop, fingertip depth) below the OPEN pad centre, then reopen."""
        open_tip = self.tip()[2]
        rows = []
        for angle in np.linspace(0., self.gripper.CLOSED_ANGLE, 9):
            self.gripper.command = 2*angle/self.gripper.CLOSED_ANGLE-1
            self._settle(90)
            rows.append((self.gripper.width(), open_tip-self.tip()[2], open_tip-self._lowest()))
        self.gripper.open()
        self._settle(240)
        return np.array(sorted(rows))

    def closing_drop(self, width):
        """(pad-centre drop, fingertip depth) once the pads close on this width."""
        widths, drops, depths = self.closure_profile.T
        return float(np.interp(width, widths, drops)), float(np.interp(width, widths, depths))

    def _settle(self, ticks):
        for _ in range(ticks):  # tool and open fingers settle before the episode
            self.gripper.apply_motor_control()
            p.stepSimulation()
            self.gripper.post_step()

    @property
    def tool_bodies(self):
        return (self.gripper.body,)

    def holding(self):
        return self.gripper.grasped_object == self.pick

    def tip(self):
        return np.asarray(self.gripper.grasp_center(), dtype=float)

    def link_pose(self):
        state = p.getLinkState(self.robot, self.spec.ee_link, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def down_quaternion(self):
        flange = Rotation.from_quat(p.getLinkState(self.robot, self.flange, computeForwardKinematics=True)[5])
        link = Rotation.from_quat(self.link_pose()[1])
        align = world_rotation_aligning(flange.apply((0, 0, 1)), (0, 0, -1))
        return (align*link).as_quat()

    def pad_axis(self):
        right, left = self.gripper.pad_positions()
        axis = np.asarray(left-right, float)
        axis[2] = 0.
        return axis/np.linalg.norm(axis)

    def tick(self, q, velocity):
        self.q, self.velocity = np.asarray(q), np.asarray(velocity)
        self._motors(self.q, self.velocity)
        self.gripper.apply_motor_control()
        p.stepSimulation()
        self.gripper.post_step()
        self.ticks += 1
        if self.tick_hook:
            self.tick_hook(self)


def task_geometry(env):
    """Waypoints shared by both controllers. Privileged simulator state only."""
    low, high = map(np.asarray, p.getAABB(env.pick))
    centre = (low+high)/2
    # Close on the narrow side. Pads should meet the object near mid-height,
    # with the fingertips (which sweep down while closing) 3 mm above the table.
    width = float(min(high[:2]-low[:2]))
    drop, depth = env.closing_drop(width)
    open_z = max(centre[2]+drop, TABLE_TOP+.003+depth)
    closed_fraction = (open_z-drop-low[2])/(high[2]-low[2])
    if closed_fraction > .75:
        raise Failed(f'Object too short for a table-clear pinch ({closed_fraction:.2f} of height)')
    grasp = np.r_[centre[:2], open_z]
    # Jaw yaw: square to a box's faces; round objects keep the current yaw.
    yaw = 0.
    if env.scene.target.endswith(('block', 'gelatin_box')):
        heading = Rotation.from_quat(p.getBasePositionAndOrientation(env.pick)[1]).as_euler('xyz')[2]
        axis = env.pad_axis()
        pads = np.arctan2(axis[1], axis[0])
        yaw = (heading-pads+np.pi/4) % (np.pi/2)-np.pi/4
    down = env.down_quaternion()
    grasp_quat = (Rotation.from_euler('z', yaw)*Rotation.from_quat(down)).as_quat()
    bowl_low, bowl_high = map(np.asarray, p.getAABB(env.place))
    hang = grasp[2]-low[2]
    place = np.r_[(bowl_low[:2]+bowl_high[:2])/2, bowl_high[2]+hang+.015]
    tops = max(p.getAABB(b)[1][2] for b in env.ids.values() if b != env.pick)
    safe_z = max(tops+.06+hang, grasp[2]+.12)
    return {'down': down, 'yaw': yaw, 'grasp_quat': grasp_quat, 'grasp': grasp, 'hover': grasp+[0, 0, .10],
            'place': place, 'safe_z': safe_z, 'start': env.tip(), 'home_quat': env.link_pose()[1]}


class PolicyStyleExecution:
    """Synthetic BC/DAgger-style execution of the reference spline (labelled as such).

    Not a trained policy. It reproduces how learned visuomotor policies run:
    - actions at 10 Hz in chunks of 4 (ACT / Diffusion Policy style), linearly
      interpolated and low-pass filtered by the joint servo: a faint 10 Hz ripple
      and slight kinks where a new chunk replaces the old one;
    - open-loop error grows inside a chunk (compounding error);
    - each new chunk, predicted from a fresh observation, pulls the error back
      towards the reference (the correction DAgger trains), leaving a
      time-correlated AR(1) drift;
    - the error is scaled by an envelope that vanishes in precision phases.
    Deterministic for a given seed.
    """
    CONTROL_HZ, CHUNK, SERVO_TAU = 10, 4, .045

    def __init__(self, rng, joints, drift=.0045, step=.0007, pull=.35):
        self.rng, self.n = rng, joints
        self.drift, self.step, self.pull = drift, step, pull
        self.error = np.zeros(joints)
        self.filtered = None

    def targets(self, plan, envelope):
        """Control-rate targets for one plan: reference + chunked compounding error."""
        count = int(np.ceil(plan.duration*self.CONTROL_HZ))+1
        times = np.minimum(np.arange(count)/self.CONTROL_HZ, plan.duration)
        out, within = [], np.zeros(self.n)
        for k, t in enumerate(times):
            if k % self.CHUNK == 0:
                # New chunk from a fresh observation: partial correction + new error.
                self.error = (1-self.pull)*(self.error+within)+self.rng.normal(0, self.drift*self.pull**.5, self.n)
                within = np.zeros(self.n)
            else:
                within = within+self.rng.normal(0, self.step, self.n)
            out.append(plan.sample(t)[0]+envelope(t/plan.duration)*(self.error+within))
        return times, np.array(out)

    def command(self, times, targets, t, dt):
        """Servo: linear interpolation of the 10 Hz targets, then first-order lag."""
        raw = np.array([np.interp(t, times, targets[:, j]) for j in range(self.n)])
        previous = self.filtered
        self.filtered = raw if previous is None else previous+(1-np.exp(-dt/self.SERVO_TAU))*(raw-previous)
        velocity = np.zeros(self.n) if previous is None else (self.filtered-previous)/dt
        return self.filtered, velocity


def window(s, start, rise, stop, fall):
    """Smooth 0..1 envelope: rises over [start, start+rise], falls to 0 at stop."""
    up = np.clip((s-start)/rise, 0, 1)
    down = np.clip((stop-s)/fall, 0, 1)
    smooth = lambda x: x*x*(3-2*x)
    return smooth(up)*smooth(down)


class GripperRollout(Rollout):
    """Ours: quintic joint-space splines through sparse keyframes, plus policy-like jitter."""
    controller = 'Ours: privileged Oracle, quintic joint-space keyframe splines + synthetic jitter (NOT a trained policy)'
    firmness_gate = False  # a pinch may shift slightly; reported as a metric instead
    jitter = True

    def follow(self, plan, phases, stop=None, envelope=None):
        """Execute a plan; with jitter, run it through PolicyStyleExecution."""
        if not self.jitter:
            return super().follow(plan, phases, stop)
        if not hasattr(self, 'policy'):
            self.policy = PolicyStyleExecution(np.random.default_rng(self.rng.integers(1 << 31)), len(self.env.q))
        envelope = envelope or (lambda s: 0.)
        times, targets = self.policy.targets(plan, envelope)
        ticks = round(plan.duration*PHYSICS_HZ)
        for tick in range(1, ticks+1):
            fraction = tick/ticks
            self.phase = [name for f, name in phases if fraction >= f][-1]
            self.env.tick(*self.policy.command(times, targets, tick/PHYSICS_HZ, 1/PHYSICS_HZ))
            self.monitor()
            self.measure_firmness()
            if stop and stop():
                return True
        return False

    def hold(self, seconds, phase):
        """Holds keep the servo state continuous when the policy model is active."""
        if not self.jitter or getattr(self, 'policy', None) is None:
            return super().hold(seconds, phase)
        self.phase = phase
        target = self.env.q.copy() if self.policy.filtered is None else self.policy.filtered.copy()
        times, targets = np.array([0., seconds]), np.vstack([target, target])
        for tick in range(1, round(seconds*PHYSICS_HZ)+1):
            self.env.tick(*self.policy.command(times, targets, tick/PHYSICS_HZ, 1/PHYSICS_HZ))
            self.monitor()
            self.measure_firmness()

    def offset(self, scale=(.005, .010)):
        """A small miss of the intermediate target, corrected on the way down."""
        angle = self.rng.uniform(0, 2*np.pi)
        return self.rng.uniform(*scale)*np.array([np.cos(angle), np.sin(angle), 0.]) if self.jitter else np.zeros(3)

    def close_and_wait(self):
        e = self.env
        self.phase = 'grasp'
        e.gripper.close()
        for tick in range(round(GRASP_TIMEOUT_S*PHYSICS_HZ)):
            self.hold(1/PHYSICS_HZ, 'grasp')
            if tick >= CLOSE_S*PHYSICS_HZ and e.holding():
                break
        if not e.holding():
            raise Failed(f'No verified bilateral grasp (width {e.gripper.width():.4f} m)')
        self.attached_tick = e.ticks
        self.object_reference = self._object_relative()

    def release(self):
        e = self.env
        self.phase = 'release'
        e.gripper.open()
        self.hold(OPEN_S, 'release')
        self.hold(IDLE_S, 'settle')  # idle so the drop can be seen; no return home

    def run(self):
        e = self.env
        rng = self.rng
        result = {'scene': asdict(e.scene), 'controller': self.controller}
        try:
            g = task_geometry(e)
            self.geometry = g
            down, q_home = g['down'], e.q.copy()
            pace = rng.uniform(.95, 1.08)
            side = rng.choice((-1., 1.))*rng.uniform(.02, .04)
            lean = rng.uniform(.10, .16)
            start, hover, grasp = g['start'], g['hover'], g['grasp']
            heading = (hover-start)[:2]/max(np.linalg.norm((hover-start)[:2]), 1e-9)
            normal = np.array([-heading[1], heading[0], 0.])
            mid = start+.55*(hover-start)+normal*side
            mid[2] = max(start[2], hover[2])+.02
            yaw_quat = lambda f: (Rotation.from_euler('z', f*g['yaw'])*Rotation.from_quat(down)).as_quat()
            q_mid = e.keyframe(mid, self.pose(yaw_quat(.5), 0., lean, heading), q_home)
            span = np.clip(1.2+1.7*np.linalg.norm(hover-start), 1.7, 3.2)*pace
            knots, times = [q_home, q_mid], [0, .5*span]
            if self.jitter:
                # Hesitate above the object: overshoot, partial correction, settle.
                miss = self.offset()
                for point, dt in ((hover+miss, span), (hover-.45*miss, span+.38), (hover+.15*miss, span+.70)):
                    knots.append(e.keyframe(point, g['grasp_quat'], knots[-1]))
                    times.append(dt)
                lead = span+.70
            else:
                knots.append(e.keyframe(hover, g['grasp_quat'], q_mid))
                times.append(span)
                lead = span
            q_near = e.keyframe(grasp+[0, 0, .035], g['grasp_quat'], knots[-1])
            q_grasp = e.keyframe(grasp, g['grasp_quat'], q_near)
            plan = JointPlan(knots+[q_near, q_grasp], times+[lead+.6, lead+1.25], degree=5)
            hover_at = span/plan.times[-1]
            self.follow(plan, [(0, 'approach'), (hover_at, 'descend')],
                        envelope=lambda s: window(s, 0., .12, (lead+.15)/plan.times[-1], .3))
            self.close_and_wait()

            place = g['place']
            travel = (place-grasp)[:2]
            heading = travel/max(np.linalg.norm(travel), 1e-9)
            normal = np.array([-heading[1], heading[0], 0.])
            lift = grasp+[0, 0, .09]
            apex = grasp+.5*(place-grasp)-rng.choice((-1., 1.))*normal*rng.uniform(.015, .035)
            apex[2] = max(g['safe_z'], lift[2]+.03)
            q_lift = e.keyframe(lift, self.pose(g['grasp_quat'], 0., .3*lean, heading), e.q)
            q_apex = e.keyframe(apex, self.pose(g['grasp_quat'], 0., lean, heading), q_lift)
            q_over = e.keyframe(place+[0, 0, .07]+self.offset((.004, .008)), g['grasp_quat'], q_apex)
            q_near = e.keyframe(place+[0, 0, .025], g['grasp_quat'], q_over)
            q_place = e.keyframe(place, g['grasp_quat'], q_near)
            sweep = np.clip(1.1+2.5*np.linalg.norm(travel), 1.6, 3.2)*pace
            times = [0, .65, .65+.5*sweep, .65+sweep, .65+sweep+.45, .65+sweep+.9]
            plan = JointPlan([e.q, q_lift, q_apex, q_over, q_near, q_place], times, degree=5)
            over_at = (.65+sweep)/plan.times[-1]
            self.follow(plan, [(0, 'carry'), (over_at, 'lower')],
                        envelope=lambda s: window(s, .65/plan.times[-1], .15, over_at+.04, .25))
            self.release()

            result.update(self.outcome(place[:2]))
        except (Failed, RuntimeError) as error:
            result.update(success=False, safety_pass=False, overall_pass=False, failure=str(error))
        return self.finish(result)


class SmoothOnlyRollout(GripperRollout):
    """Ablation: Ours without the synthetic jitter."""
    controller = 'Ours without jitter (ablation)'
    jitter = False


class GripperIKRollout(IKRollout, GripperRollout):
    """Baseline: straight Cartesian lines between the SAME waypoints, stop at each."""
    jitter = False
    controller = 'IK baseline: straight Cartesian segments (MoveL), stop at every waypoint'

    def run(self):
        e = self.env
        result = {'scene': asdict(e.scene), 'controller': self.controller}
        try:
            g = task_geometry(e)
            self.geometry = g
            self.move(g['hover'], g['grasp_quat'], 'approach', start_quat=g['home_quat'])
            self.move(g['grasp'], g['grasp_quat'], 'descend')
            self.close_and_wait()
            tip = e.tip()
            self.move(np.r_[tip[:2], g['safe_z']], g['grasp_quat'], 'carry')
            self.move(np.r_[g['place'][:2], g['safe_z']], g['grasp_quat'], 'carry')
            self.move(g['place'], g['grasp_quat'], 'lower')
            self.release()
            result.update(self.outcome(g['place'][:2]))
        except (Failed, RuntimeError) as error:
            result.update(success=False, safety_pass=False, overall_pass=False, failure=str(error))
        return self.finish(result)
