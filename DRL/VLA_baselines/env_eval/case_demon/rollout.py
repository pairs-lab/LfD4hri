"""Privileged suction Oracle at 240 Hz: scripted control, NOT a trained policy.

Motion is a joint-space spline through sparse IK keyframes, so the whole arm
flows instead of the tip tracing a Cartesian line. All attachment goes through
the repository's Suction class after a measured tip/target contact. No object
is teleported during an episode.
"""
import importlib.util
from dataclasses import asdict

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation

from config import COLORS
from env_eval.case_demon.motion import JointPlan
from env_eval.case_demon.robots import DRL, ROBOTS
from utils.suction import Suction

PHYSICS_HZ = 240
GHOST_OFFSET = np.array([0., 0., -5.])
TABLE_TOP = .004


def world_rotation_aligning(a, b):
    a, b = np.asarray(a)/np.linalg.norm(a), np.asarray(b)/np.linalg.norm(b)
    axis, c = np.cross(a, b), float(a @ b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return Rotation.identity() if c > 0 else Rotation.from_rotvec(np.pi*np.array([1., 0., 0.]))
    return Rotation.from_rotvec(axis/s*np.arctan2(s, c))


class Failed(RuntimeError):
    pass


class SuctionEnv:
    def __init__(self, scene, egl=True):
        self.scene, self.spec = scene, ROBOTS[scene.robot]
        self.client = p.connect(p.DIRECT)
        self.renderer, self.plugin = p.ER_TINY_RENDERER, None
        if egl:
            spec = importlib.util.find_spec('eglRenderer')
            if spec:
                self.plugin = p.loadPlugin(spec.origin, '_eglRendererPlugin')
                if self.plugin >= 0:
                    self.renderer = p.ER_BULLET_HARDWARE_OPENGL
        self.tick_hook = None
        self.ticks = 0
        self._load()

    def close(self):
        if self.plugin is not None and self.plugin >= 0:
            p.unloadPlugin(self.plugin)
        p.disconnect(self.client)

    def _spawn(self, name, xy, fixed):
        color, kind = name.split(' ', 1)
        if kind == 'block':
            shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.02, .02, .02))
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(.02, .02, .02), rgbaColor=COLORS[color])
            body = p.createMultiBody(.05, shape, visual, basePosition=(*xy, TABLE_TOP+.021))
        else:
            body = p.loadURDF(str(DRL / f'asset/{kind}/{kind}.urdf'), (*xy, .08), useFixedBase=fixed)
            if kind == 'bowl':
                p.changeVisualShape(body, -1, rgbaColor=COLORS[color])
            # Rest the mesh on the table without an initial drop.
            low = p.getAABB(body)[0][2]
            pos, quat = p.getBasePositionAndOrientation(body)
            p.resetBasePositionAndOrientation(body, (pos[0], pos[1], pos[2]-low+TABLE_TOP+.001), quat)
        p.changeDynamics(body, -1, lateralFriction=1.2, spinningFriction=.02, rollingFriction=.01,
                         restitution=0., linearDamping=.5, angularDamping=.5)
        return body

    def _load(self):
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/PHYSICS_HZ)
        p.setPhysicsEngineParameter(numSolverIterations=150, contactERP=.9,
                                    solverResidualThreshold=1e-7, enableFileCaching=0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.plane = p.loadURDF(str(DRL / 'asset/plane.urdf'), (0, 0, -.004), useFixedBase=True)
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.45, .35, .004))
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(.45, .35, .004), rgbaColor=(.32, .35, .38, 1))
        self.table = p.createMultiBody(0, shape, visual, basePosition=(0, -.5, 0))
        self.robot = p.loadURDF(str(self.spec.urdf), (0, 0, 0),
                                p.getQuaternionFromEuler((0, 0, self.spec.base_yaw)), useFixedBase=True,
                                flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        # Offline kinematic copy for FK/Jacobians: below the floor, no collisions.
        self.ghost = p.loadURDF(str(self.spec.urdf), GHOST_OFFSET.tolist(),
                                p.getQuaternionFromEuler((0, 0, self.spec.base_yaw)), useFixedBase=True)
        for link in range(-1, p.getNumJoints(self.ghost)):
            p.setCollisionFilterGroupMask(self.ghost, link, 0, 0)
        self.joints = [i for i in range(p.getNumJoints(self.robot))
                       if p.getJointInfo(self.robot, i)[2] == p.JOINT_REVOLUTE]
        if len(self.joints) != len(self.spec.home):
            raise ValueError(f'{self.spec.label}: joint count differs from home')
        for joint, value in zip(self.joints, self.spec.home):
            p.resetJointState(self.robot, joint, value)
        for link, rgba in self.spec.colors:
            p.changeVisualShape(self.robot, link, rgbaColor=rgba)
        info = [p.getJointInfo(self.robot, j) for j in self.joints]
        self.lower = np.array([i[8] if i[8] < i[9] else -2*np.pi for i in info])
        self.upper = np.array([i[9] if i[8] < i[9] else 2*np.pi for i in info])
        self.ids = {}
        for index, (name, xy) in enumerate(self.scene.objects):
            self.ids[name] = self._spawn(name, xy, name.endswith(' bowl'))
        self.pick, self.place = self.ids[self.scene.target], self.ids[self.scene.goal]
        self._attach_tool([b for n, b in self.ids.items() if not n.endswith(' bowl')])
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.q = np.array([p.getJointState(self.robot, j)[0] for j in self.joints])
        self.velocity = np.zeros(len(self.q))
        for _ in range(PHYSICS_HZ):
            self._motors(self.q, self.velocity)
            p.stepSimulation()
        self.initial = {n: np.array(p.getBasePositionAndOrientation(b)[0]) for n, b in self.ids.items()}
        self.local_tip = self._local_tip()

    def _attach_tool(self, dynamic):
        self.suction = Suction(self.robot, self.spec.ee_link, dynamic)
        self._seat_suction()

    @property
    def tool_bodies(self):
        return (self.suction.body, self.suction.base)

    def holding(self):
        return self.suction.check_grasp()

    def _seat_suction(self):
        # Suction.__init__ spawns at a hard-coded pose; seat it on the flange
        # first so the constraint cannot sweep it through the scene.
        com, quat = p.getLinkState(self.robot, self.spec.ee_link, computeForwardKinematics=True)[:2]
        rotation = Rotation.from_quat(quat)
        p.resetBasePositionAndOrientation(self.suction.base, np.array(com)-rotation.apply((0, 0, .01)), quat)
        p.resetBasePositionAndOrientation(self.suction.body, np.array(com)+rotation.apply((0, 0, .08)), quat)
        for link in range(-1, p.getNumJoints(self.robot)):
            for body in (self.suction.base, self.suction.body):
                for own in range(-1, p.getNumJoints(body)):
                    p.setCollisionFilterPair(self.robot, body, link, own, 0)
        # The URDFs give this 0.1 kg tool an inertia of 1 kg m2, which makes it
        # lag and swing on its constraints. Use a realistic value and stiff mounts.
        for body in (self.suction.base, self.suction.body):
            for own in range(-1, p.getNumJoints(body)):
                p.changeDynamics(body, own, localInertiaDiagonal=(2e-4, 2e-4, 2e-4), linearDamping=0., angularDamping=0.)
        for index in range(p.getNumConstraints()):
            constraint = p.getConstraintUniqueId(index)
            if p.getConstraintInfo(constraint)[2] in (self.suction.base, self.suction.body):
                p.changeConstraint(constraint, maxForce=1e7, erp=.95)

    def link_pose(self):
        state = p.getLinkState(self.robot, self.spec.ee_link, computeForwardKinematics=True)
        return np.array(state[4]), np.array(state[5])

    def tip(self):
        return np.array(p.getLinkState(self.suction.body, 0, computeForwardKinematics=True)[0])

    def _local_tip(self):
        # Expressed in the link inertial frame, as calculateJacobian expects.
        state = p.getLinkState(self.robot, self.spec.ee_link, computeForwardKinematics=True)
        return Rotation.from_quat(state[1]).inv().apply(self.tip()-np.array(state[0]))

    def down_quaternion(self):
        """Link orientation with the measured suction axis vertically down."""
        _, link = self.link_pose()
        head = Rotation.from_quat(p.getBasePositionAndOrientation(self.suction.body)[1])
        align = world_rotation_aligning(head.apply((0, 0, 1)), (0, 0, -1))
        return (align*Rotation.from_quat(link)).as_quat()

    def _motors(self, q, velocity):
        p.setJointMotorControlArray(self.robot, self.joints, p.POSITION_CONTROL, targetPositions=q,
                                    targetVelocities=velocity, forces=[250]*len(q),
                                    positionGains=[.16]*len(q), velocityGains=[1.]*len(q))

    def _ghost_state(self, q):
        for joint, value in zip(self.joints, q):
            p.resetJointState(self.ghost, joint, float(value))
        state = p.getLinkState(self.ghost, self.spec.ee_link, computeForwardKinematics=True)
        return (np.array(state[0])-GHOST_OFFSET, Rotation.from_quat(state[1]), Rotation.from_quat(state[5]))

    def solve_ik(self, tip, quat, iterations=4, seed=None):
        """Damped least squares with a home nullspace pull, on an offline ghost.

        PyBullet's nullspace IK mishandles these yawed bases, so all five arms
        use this one solver; joint limits are enforced by clipping.
        """
        q = (self.q if seed is None else np.asarray(seed)).copy()
        target = Rotation.from_quat(quat)
        home = np.asarray(self.spec.home)
        n = len(q)
        for _ in range(iterations):
            com, com_rotation, link_rotation = self._ghost_state(q)
            position = com+com_rotation.apply(self.local_tip)
            error = np.r_[np.asarray(tip)-position, (target*link_rotation.inv()).as_rotvec()]
            zero = [0.]*n
            linear, angular = p.calculateJacobian(self.ghost, self.spec.ee_link, self.local_tip.tolist(),
                                                  q.tolist(), zero, zero)
            # PyBullet returns these Jacobians in the base frame, not world.
            yaw = Rotation.from_euler('z', self.spec.base_yaw).as_matrix()
            J = np.vstack((yaw@np.array(linear), yaw@np.array(angular)))
            pinv = J.T@np.linalg.inv(J@J.T+1e-4*np.eye(6))
            step = pinv@error+(np.eye(n)-pinv@J)@(.05*(home-q))
            q = np.clip(q+np.clip(step, -.25, .25), self.lower, self.upper)
        return q

    def ik_error(self, q, tip, quat):
        com, com_rotation, link_rotation = self._ghost_state(q)
        position = com+com_rotation.apply(self.local_tip)
        angle = (Rotation.from_quat(quat)*link_rotation.inv()).magnitude()
        return float(np.linalg.norm(position-np.asarray(tip))), float(angle)

    def tick(self, q, velocity):
        self.q, self.velocity = np.asarray(q), np.asarray(velocity)
        self._motors(self.q, self.velocity)
        p.stepSimulation()
        self.ticks += 1
        if self.tick_hook:
            self.tick_hook(self)

    def keyframe(self, tip, quat, seed):
        solution = self.solve_ik(tip, quat, iterations=120, seed=seed)
        error, angle = self.ik_error(solution, tip, quat)
        if error > .003 or angle > .05:
            raise Failed(f'Keyframe unreachable: {error*1e3:.1f} mm, {angle:.3f} rad at {np.round(tip, 3)}')
        return solution

    def grip(self):
        """Attach after contact: stiff constraint, no tip/object contact fight."""
        self.suction.activate()
        if self.suction.contact_constraint is None:
            return False
        p.changeConstraint(self.suction.contact_constraint, maxForce=2000, erp=.95)
        self._tip_collision(0)
        return True

    def ungrip(self):
        self.suction.release()
        self._tip_collision(1)

    def _tip_collision(self, enabled):
        for own in range(-1, p.getNumJoints(self.suction.body)):
            p.setCollisionFilterPair(self.suction.body, self.pick, own, -1, enabled)

    def surface_point(self):
        """Top of the requested object, measured by a simulator ray."""
        low, high = map(np.asarray, p.getAABB(self.pick))
        center = (low+high)/2
        hit = p.rayTest((center[0], center[1], high[2]+.03), (center[0], center[1], low[2]-.01))[0]
        if hit[0] != self.pick:
            raise Failed('Ray from above does not reach the target centre')
        return np.array(hit[3])

    def tip_contacts(self):
        return p.getContactPoints(bodyA=self.suction.body, linkIndexA=0)


class Rollout:
    firmness_gate = True  # suction: a loose hold fails the episode

    def __init__(self, env):
        self.env = env
        self.events = []
        self.unwanted = set()
        self.pairs = self._self_pairs()
        self.self_baseline = {}
        self.self_violation = 0.
        self.phase = 'start'
        self.attached_tick = None
        self.firm = {'head_mm': 0., 'head_deg': 0., 'object_mm': 0., 'object_deg': 0.}
        self.tool_reference = self._tool_relative()
        self.object_reference = None
        self.rng = np.random.default_rng(7700+1000*env.scene.scene_id+env.scene.candidate)

    def _tool_relative(self):
        e = self.env
        com, quat = p.getLinkState(e.robot, e.spec.ee_link, computeForwardKinematics=True)[:2]
        frame = Rotation.from_quat(quat)
        position, orientation = p.getBasePositionAndOrientation(e.tool_bodies[0])
        return frame.inv().apply(np.array(position)-com), frame.inv()*Rotation.from_quat(orientation)

    def _object_relative(self):
        e = self.env
        tool, orientation = p.getBasePositionAndOrientation(e.tool_bodies[0])
        frame = Rotation.from_quat(orientation)
        position, quat = p.getBasePositionAndOrientation(e.pick)
        return frame.inv().apply(np.array(position)-tool), frame.inv()*Rotation.from_quat(quat)

    def measure_firmness(self):
        position, orientation = self._tool_relative()
        self.firm['head_mm'] = max(self.firm['head_mm'], 1e3*float(np.linalg.norm(position-self.tool_reference[0])))
        self.firm['head_deg'] = max(self.firm['head_deg'], float(np.degrees((self.tool_reference[1].inv()*orientation).magnitude())))
        if self.object_reference is not None and self.env.holding():
            position, orientation = self._object_relative()
            self.firm['object_mm'] = max(self.firm['object_mm'], 1e3*float(np.linalg.norm(position-self.object_reference[0])))
            self.firm['object_deg'] = max(self.firm['object_deg'], float(np.degrees((self.object_reference[1].inv()*orientation).magnitude())))

    def _self_pairs(self):
        e = self.env
        parents = {i: p.getJointInfo(e.robot, i)[16] for i in range(p.getNumJoints(e.robot))}
        links = [i for i in range(-1, p.getNumJoints(e.robot)) if p.getCollisionShapeData(e.robot, i)]
        def chain(a):
            out = [a]
            while out[-1] != -1:
                out.append(parents[out[-1]])
            return out
        def distance(a, b):
            ca, cb = chain(a), chain(b)
            return min(ca.index(x)+cb.index(x) for x in set(ca) & set(cb))
        return [(a, b) for i, a in enumerate(links) for b in links[i+1:] if distance(a, b) > 2]

    def monitor(self):
        e = self.env
        objects = set(e.ids.values())
        for c in p.getContactPoints(bodyA=e.robot):
            if c[2] in (e.plane,) and c[3] == -1:
                continue
            if c[2] not in e.tool_bodies:
                self.unwanted.add(('arm', int(c[3]), int(c[2]), self.phase))
        held = e.pick if e.holding() else None
        for c in [c for body in e.tool_bodies for c in p.getContactPoints(bodyA=body)]:
            if c[2] in objects and c[2] != e.pick:
                self.unwanted.add(('suction', int(c[2]), self.phase))
            if c[2] == e.table:
                self.unwanted.add(('suction_table', self.phase))
        if held is not None:
            for c in p.getContactPoints(bodyA=held):
                if c[2] in objects:
                    self.unwanted.add(('carried', int(c[2]), self.phase))
        if e.ticks % 8 == 0:
            for a, b in self.pairs:
                hits = p.getClosestPoints(e.robot, e.robot, .002, a, b)
                d = min((h[8] for h in hits), default=.002)
                base = self.self_baseline.setdefault((a, b), d)
                self.self_violation = max(self.self_violation, base-d if d < 0 else 0.)

    def follow(self, plan, phases, stop=None):
        """phases: (start_fraction, name); stop may end the motion on contact."""
        ticks = round(plan.duration*PHYSICS_HZ)
        for tick in range(1, ticks+1):
            elapsed = tick/PHYSICS_HZ
            self.phase = [name for fraction, name in phases if tick/ticks >= fraction][-1]
            self.env.tick(*plan.sample(elapsed))
            self.monitor()
            self.measure_firmness()
            if stop and stop():
                return True
        return False

    def hold(self, seconds, phase):
        self.phase = phase
        q = self.env.q.copy()
        for _ in range(round(seconds*PHYSICS_HZ)):
            self.env.tick(q, np.zeros(len(q)))
            self.monitor()
            self.measure_firmness()

    def pose(self, q_down, yaw=0., lean=0., direction=(1., 0., 0.)):
        """Tool orientation: yaw about vertical, leaning into the travel direction."""
        direction = np.r_[np.asarray(direction, dtype=float)[:2], 0.]
        direction /= max(np.linalg.norm(direction), 1e-9)
        tilt = Rotation.from_rotvec(np.cross(direction, (0, 0, 1))*lean)
        return (tilt*Rotation.from_euler('z', yaw)*Rotation.from_quat(q_down)).as_quat()

    def outcome(self, goal_xy):
        """Physical result and safety, shared by every controller compared."""
        e = self.env
        final = np.array(p.getBasePositionAndOrientation(e.pick)[0])
        support = bool(p.getContactPoints(bodyA=e.pick, bodyB=e.place))
        speed = float(np.linalg.norm(p.getBaseVelocity(e.pick)[0]))
        offset = float(np.linalg.norm(final[:2]-goal_xy))
        moved = {n: float(np.linalg.norm(np.array(p.getBasePositionAndOrientation(b)[0])-e.initial[n]))
                 for n, b in e.ids.items() if b not in (e.pick, e.place)}
        released = not e.holding()
        physical = bool(released and support and offset < .045 and speed < .025)
        firm = (not self.firmness_gate or self.firm['head_mm'] <= 1. and self.firm['head_deg'] <= 1.
                and self.firm['object_mm'] <= 1.5 and self.firm['object_deg'] <= 1.5)
        safe = not self.unwanted and self.self_violation < .001 and max(moved.values(), default=0) < .02 and firm
        return dict(success=physical, safety_pass=safe, overall_pass=bool(physical and safe),
                    released=released, bowl_support=support, final_xy_offset_m=offset,
                    final_speed_m_s=speed, distractor_motion_m=moved, final_object=final.tolist())

    def finish(self, result):
        e = self.env
        result.update(phase_reached=self.phase, ticks=e.ticks, seconds=e.ticks/PHYSICS_HZ,
                      attached_after_seconds=None if self.attached_tick is None else self.attached_tick/PHYSICS_HZ,
                      unwanted_contacts=sorted(map(str, self.unwanted)),
                      self_collision_excess_m=self.self_violation, suction_firmness=self.firm)
        return result

    def run(self):
        e = self.env
        scene = e.scene
        rng = self.rng
        result = {'scene': asdict(scene), 'controller': 'scripted privileged suction Oracle; NOT a trained policy'}
        try:
            down = e.down_quaternion()
            pace = rng.uniform(.92, 1.1)
            side = rng.choice((-1., 1.))*rng.uniform(.018, .04)
            lean = rng.uniform(.09, .17)
            yaw = rng.uniform(-.25, .25)
            yaw_carry = yaw+rng.uniform(-.2, .2)
            start, q_home = e.tip(), e.q.copy()

            # Reach: one sweep from home, slowing into a guarded descent.
            surface = e.surface_point()
            hover = surface+[0, 0, .06]
            flat = (hover-start)[:2]
            heading = flat/max(np.linalg.norm(flat), 1e-9)
            normal = np.array([-heading[1], heading[0], 0.])
            mid = start+.55*(hover-start)+normal*side
            mid[2] = max(start[2], hover[2])+.015
            q_mid = e.keyframe(mid, self.pose(down, .5*yaw, lean, heading), q_home)
            q_hover = e.keyframe(hover, self.pose(down, yaw), q_mid)
            q_near = e.keyframe(surface+[0, 0, .022], self.pose(down, yaw), q_hover)
            q_touch = e.keyframe(surface+[0, 0, -.006], self.pose(down, yaw), q_near)
            span = np.clip(1.15+1.7*np.linalg.norm(hover-start), 1.6, 3.2)*pace
            times = [0, .52*span, span, span+.55, span+1.15]
            plan = JointPlan([q_home, q_mid, q_hover, q_near, q_touch], times)
            def touching():
                points = e.tip_contacts()
                return bool(points) and all(c[2] == e.pick for c in points)
            self.follow(plan, [(0, 'approach'), (span/plan.times[-1], 'descend')], touching)
            if not touching():
                raise Failed('Suction tip never contacted the target')
            result['contact_xy_error_m'] = float(np.linalg.norm((e.tip()-surface)[:2]))
            # Seal on first contact and come to rest at that height. Actively
            # pressing flexes the tool mount and makes the object spring on lift-off.
            if not e.grip():
                raise Failed('Contact did not create a suction attachment')
            self.attached_tick = e.ticks
            q_press = e.keyframe(e.tip(), self.pose(down, yaw), e.q)
            self.follow(JointPlan([e.q, q_press], [0, .18], v0=e.velocity), [(0, 'attach')])
            self.hold(.10, 'attach')
            self.object_reference = self._object_relative()

            # Carry: lift, arc over the clutter, slow down into the bowl.
            tip = e.tip()
            obj = np.array(p.getBasePositionAndOrientation(e.pick)[0])
            low = np.array(p.getAABB(e.pick)[0])
            bowl_low, bowl_high = map(np.asarray, p.getAABB(e.place))
            goal_xy = (bowl_low[:2]+bowl_high[:2])/2
            hang = obj[2]-low[2]
            goal = np.r_[goal_xy, bowl_high[2]+hang+.012]
            target = tip+goal-obj
            tops = [p.getAABB(b)[1][2] for b in e.ids.values() if b != e.pick]
            apex_z = max(tops)+.05+(tip[2]-low[2])
            lift = tip+[0, 0, .08]
            travel = (target-tip)[:2]
            distance = float(np.linalg.norm(travel))
            heading = travel/max(distance, 1e-9)
            normal = np.array([-heading[1], heading[0], 0.])
            apex = tip+.5*(target-tip)-rng.choice((-1., 1.))*normal*rng.uniform(.015, .035)
            apex[2] = max(apex_z, lift[2]+.03)
            over = target+[0, 0, .06]
            near = target+[0, 0, .02]
            q_lift = e.keyframe(lift, self.pose(down, yaw, .3*lean, heading), q_press)
            q_apex = e.keyframe(apex, self.pose(down, .5*(yaw+yaw_carry), lean, heading), q_lift)
            q_over = e.keyframe(over, self.pose(down, yaw_carry, 0., heading), q_apex)
            q_near = e.keyframe(near, self.pose(down, yaw_carry), q_over)
            q_goal = e.keyframe(target, self.pose(down, yaw_carry), q_near)
            sweep = np.clip(1.05+2.5*distance, 1.5, 3.1)*pace
            times = [0, .62, .62+.5*sweep, .62+sweep, .62+sweep+.4, .62+sweep+.8]
            plan = JointPlan([e.q, q_lift, q_apex, q_over, q_near, q_goal], times)
            self.follow(plan, [(0, 'carry'), ((.62+sweep)/plan.times[-1], 'lower')])
            self.phase = 'release'
            e.ungrip()
            self.hold(.12, 'release')

            # Rise clear of the bowl, then flow back to the home pose.
            q_up = e.keyframe(target+[0, 0, .05], self.pose(down, yaw_carry), e.q)
            q_clear = e.keyframe(target+[0, 0, .17], self.pose(down, .5*yaw_carry), q_up)
            back = np.clip(1.2+1.3*np.linalg.norm(target-start), 1.8, 3.)*pace
            plan = JointPlan([e.q, q_up, q_clear, q_home], [0, .5, 1.05, 1.05+back])
            self.follow(plan, [(0, 'retract')])
            self.hold(.7, 'settle')

            result.update(self.outcome(goal_xy))
        except (Failed, RuntimeError) as error:
            result.update(success=False, safety_pass=False, overall_pass=False, failure=str(error))
        return self.finish(result)
