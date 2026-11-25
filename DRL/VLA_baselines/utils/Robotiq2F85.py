import os
import threading
import time
import numpy as np
import pybullet

''' Index: 0, Name: base_link_robotiq_2f_85_base_joint
Index: 1, Name: robotiq_2f_85_right_driver_joint (main)
Index: 2, Name: robotiq_2f_85_right_coupler_joint
Index: 3, Name: robotiq_2f_85_right_follower_joint (cover)
Index: 4, Name: robotiq_2f_85_right_pad_joint (inside cover)
Index: 5, Name: robotiq_2f_85_right_spring_link_joint
Index: 6, Name: robotiq_2f_85_left_driver_joint  
Index: 7, Name: robotiq_2f_85_left_coupler_joint
Index: 8, Name: robotiq_2f_85_left_follower_joint
Index: 9, Name: robotiq_2f_85_left_pad_joint
Index: 10, Name: robotiq_2f_85_left_spring_link_joint
'''

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def root_path(target):
    return os.path.join(ROOT_PATH, 'asset', target)

# @markdown **Gripper class:** adds a gripper to the robot and runs a parallel thread to simulate single-actuator behavior.

class Robotiq2F85:
  """Gripper handling for Robotiq 2F85."""

  JOINTS = (1, 3, 5, 6, 8, 10)
  SIGNS = np.array([1, -1, 1, 1, -1, 1])
  PAD_LINKS = (4, 9)
  OPEN_ANGLE = 0.0
  CLOSED_ANGLE = 0.78
  DRIVER_FORCE = 30.
  FOLLOWER_FORCE = 60.
  # Preserve the original UR mount. UF850 overrides these in its own class.
  MOUNT_OFFSET = (0., 0., .015)
  MOUNT_PARENT_POSITION = (0., 0., 0.)
  MOUNT_CHILD_LINK = 0
  MOUNT_CHILD_POSITION = (0., 0., -.07)

  def __init__(self, robot, tool, *, start_thread=True, urdf=None, object_ids=(), assist_grasp=False):
    """Original articulated gripper; robot-specific mount settings are isolated.

    The optional step-driven API avoids a background-thread race in demos.
    Existing Robotiq2F85(robot, tool) callers retain the original UR mount.
    """
    self.robot = self.robot_id = robot
    self.tool = self.mount_link = tool
    self.object_ids = list(object_ids)
    self.assist_grasp = bool(assist_grasp)
    self.command = -1.0
    self.grasp_constraint = self.grasped_object = None
    self._latched = None
    self._contact_streak = {}
    self._lost_contact_ticks = 0
    self.motor_joint = 1
    self.mount_offset = np.array(self.MOUNT_OFFSET)
    self.mount_rotation = pybullet.getQuaternionFromEuler([0, 0, -np.pi/2])
    state = pybullet.getLinkState(robot, tool, computeForwardKinematics=True)
    pos, rot = pybullet.multiplyTransforms(state[4], state[5], self.mount_offset, self.mount_rotation)
    self.urdf_path = urdf or root_path('robotiq_2f_85/robotiq_2f_85.urdf')
    self.body = pybullet.loadURDF(str(self.urdf_path), pos, rot)
    self.n_joints = pybullet.getNumJoints(self.body)
    self.activated = False
    self.constraint_id = self.mount_constraint = pybullet.createConstraint(
      parentBodyUniqueId=robot, parentLinkIndex=tool,
      childBodyUniqueId=self.body, childLinkIndex=self.MOUNT_CHILD_LINK,
      jointType=pybullet.JOINT_FIXED, jointAxis=[0, 0, 0],
      parentFramePosition=self.MOUNT_PARENT_POSITION, childFramePosition=self.MOUNT_CHILD_POSITION,
      childFrameOrientation=pybullet.getQuaternionFromEuler([0, 0, np.pi/2]))
    self._configure_mount_dynamics()
    for i in range(self.n_joints):
      pybullet.changeDynamics(self.body, i, lateralFriction=10., spinningFriction=1.,
        rollingFriction=1., frictionAnchor=True)
    for grip_link in range(-1, self.n_joints):
      pybullet.setCollisionFilterPair(robot, self.body, tool, grip_link, 0)
    self.max_mount_translation_error = self.max_mount_rotation_error = 0.
    self._load_pad_geometry()
    self.running = bool(start_thread)
    self.constraints_thread = None
    if start_thread:
      self.constraints_thread = threading.Thread(target=self.step, daemon=True)
      self.constraints_thread.start()
    self.release()

  def _configure_mount_dynamics(self):
    """UR5e retains the original constraint/dynamics defaults."""
    pass

  def _load_pad_geometry(self):
    import xml.etree.ElementTree as ET
    import trimesh
    from pathlib import Path
    path = Path(self.urdf_path)
    root = ET.parse(path).getroot()
    self.vertices = {}
    for side, link in zip(('right', 'left'), self.PAD_LINKS):
      descriptor = root.find(f"link[@name='robotiq_2f_85_{side}_pad']/collision/geometry/mesh")
      mesh = trimesh.load(path.parent / descriptor.get('filename'), force='mesh', process=False)
      self.vertices[link] = np.asarray(mesh.vertices) * np.array(list(map(float, descriptor.get('scale').split())))

  def shutdown(self):
    self.running = False
    if self.constraints_thread is not None:
      self.constraints_thread.join(timeout=1.)

  def open(self):
    self.command = -1.
    self.activated = False
    self._release_assisted_grasp()

  def close(self):
    self.command = 1.
    self.activated = True

  @property
  def target_angle(self):
    return self.OPEN_ANGLE + (self.command+1)*.5*(self.CLOSED_ANGLE-self.OPEN_ANGLE)

  def apply_motor_control(self):
    """Same gripper joints; each finger chain follows its measured actuator."""
    targets = [self.target_angle, self.target_angle]
    limit = getattr(self, 'pregrasp_force_limit', None)
    contact_object = getattr(self, 'pregrasp_object', None)
    if limit is not None and contact_object is not None and self.grasp_constraint is None and getattr(self, 'pregrasp_force_enabled', False):
      # Optional feedback-demo mode. Legacy callers retain their original
      # motor commands. Avoid forceful squeezing while the second pad is
      # still approaching; the command still expresses intent to grasp.
      contacts = pybullet.getContactPoints(self.body, contact_object)
      dt = pybullet.getPhysicsEngineParameters()['fixedTimeStep']
      for index, driver, links in ((0, 1, (1,2,3,4,5)), (1, 6, (6,7,8,9,10))):
        force = sum(c[9] for c in contacts if c[3] in links)
        actual = pybullet.getJointState(self.body, driver)[0]
        if force > limit:
          targets[index] = min(targets[index], actual - min(.25, .08*(force-limit))*dt)
        elif force > .35:
          targets[index] = min(targets[index], actual + .06*dt)
    if self._latched is not None and self.grasp_constraint is not None:
      # Maintain a gentle pinch as compliant contact settles. Fixed encoders
      # alone can unload one pad during a curved, rotating carry.
      contacts = pybullet.getContactPoints(self.body, self.grasped_object)
      dt = pybullet.getPhysicsEngineParameters()['fixedTimeStep']
      for index, pad in zip((0, 3), self.PAD_LINKS):
        force = sum(c[9] for c in contacts if c[3] == pad)
        rate = .12 if force < .8 else (-.04 if force > 6. else 0.)
        self._latched[index] = np.clip(self._latched[index] + rate*dt, 0., self.CLOSED_ANGLE)
      targets = [float(self._latched[0]), float(self._latched[3])]
    for driver, followers, target in ((1, ((3,-1),(5,1)), targets[0]), (6, ((8,-1),(10,1)), targets[1])):
      pybullet.setJointMotorControl2(self.body, driver, pybullet.POSITION_CONTROL,
        targetPosition=float(target), force=self.DRIVER_FORCE, positionGain=.3, velocityGain=1., maxVelocity=1.5)
      actual = pybullet.getJointState(self.body, driver)[0]
      for joint, sign in followers:
        pybullet.setJointMotorControl2(self.body, joint, pybullet.POSITION_CONTROL,
          targetPosition=float(sign*actual), force=self.FOLLOWER_FORCE, positionGain=.8,
          velocityGain=1., maxVelocity=1.91986177778)

  def world_pad_vertices(self):
    output = []
    for link in self.PAD_LINKS:
      state = pybullet.getLinkState(self.body, link, computeForwardKinematics=True)
      rotation = np.array(pybullet.getMatrixFromQuaternion(state[5])).reshape(3,3)
      output.append(self.vertices[link] @ rotation.T + state[4])
    return output

  def pad_positions(self):
    return tuple(v.mean(0).astype(np.float32) for v in self.world_pad_vertices())

  def grasp_center(self):
    return np.mean(self.pad_positions(), axis=0)

  def width(self):
    right,left = self.world_pad_vertices()
    axis = left.mean(0)-right.mean(0);axis /= max(np.linalg.norm(axis), 1e-12)
    return float(max(0., (left@axis).min()-(right@axis).max()))

  def _bilateral(self, body):
    return set(self.PAD_LINKS) <= {c[3] for c in pybullet.getContactPoints(self.body,body) if c[9]>.1}

  def is_touching(self, body):
    return bool(pybullet.getContactPoints(self.body, body))

  def _create_assisted_grasp(self, body):
    self._latched = np.array([pybullet.getJointState(self.body,j)[0] for j in self.JOINTS])+.005*self.SIGNS
    self._lost_contact_ticks = 0
    parent = pybullet.getLinkState(self.body,0,computeForwardKinematics=True)[:2]
    child = pybullet.getBasePositionAndOrientation(body)
    inverse = pybullet.invertTransform(*parent)
    relative = pybullet.multiplyTransforms(*inverse,*child)
    self.grasp_constraint = pybullet.createConstraint(self.body,0,body,-1,pybullet.JOINT_FIXED,[0,0,0],
      relative[0],[0,0,0],parentFrameOrientation=relative[1],childFrameOrientation=[0,0,0,1])
    pybullet.changeConstraint(self.grasp_constraint,maxForce=250.)
    self.grasped_object = body

  def _release_assisted_grasp(self):
    if self.grasp_constraint is not None:
      pybullet.removeConstraint(self.grasp_constraint)
    self.grasp_constraint = self.grasped_object = self._latched = None
    self._lost_contact_ticks = 0
    self._contact_streak = {}

  def post_step(self):
    from scipy.spatial.transform import Rotation
    tool = pybullet.getLinkState(self.robot,self.tool,computeForwardKinematics=True)
    base = pybullet.getBasePositionAndOrientation(self.body)
    rotation = Rotation.from_quat(tool[5])
    relative = rotation.inv().apply(np.asarray(base[0])-tool[4])
    angle = (Rotation.from_quat(self.mount_rotation).inv()*rotation.inv()*Rotation.from_quat(base[1])).magnitude()
    self.relative_mount_position = relative.tolist()
    self.relative_mount_quaternion = (rotation.inv()*Rotation.from_quat(base[1])).as_quat().tolist()
    self.max_mount_translation_error = max(self.max_mount_translation_error,float(np.linalg.norm(relative-self.mount_offset)))
    self.max_mount_rotation_error = max(self.max_mount_rotation_error,float(np.degrees(angle)))
    if self.grasped_object is not None:
      self._lost_contact_ticks = 0 if self._bilateral(self.grasped_object) else self._lost_contact_ticks+1
      if self._lost_contact_ticks>=12:self._release_assisted_grasp()
    if self.command>.2 and self.grasp_constraint is None and self.assist_grasp:
      options = []
      for body in self.object_ids:
        self._contact_streak[body] = self._contact_streak.get(body,0)+1 if self._bilateral(body) else 0
        if self._contact_streak[body]<3:continue
        low,high = map(np.asarray,pybullet.getAABB(body));centre = self.grasp_center()
        fraction = (centre[2]-low[2])/max(high[2]-low[2],1e-6)
        if .25<=fraction<=.8:options.append((np.linalg.norm(centre-(low+high)/2),body))
      if options:self._create_assisted_grasp(min(options)[1])

  # Control joint positions by enforcing hard contraints on gripper behavior.
  # Set one joint as the open/close motor joint (other joints should mimic).
  def step(self):
    while self.running:
      try:
        currj = [pybullet.getJointState(self.body, i)[0] for i in range(self.n_joints)]
        indj = [6, 3, 8, 5, 10]
        targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
        pybullet.setJointMotorControlArray(self.body, indj, pybullet.POSITION_CONTROL, targj, positionGains=np.ones(5))
      except:
        return
      time.sleep(0.001)

  # Close gripper fingers.
  def activate(self):
    pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=1, force=10)
    self.activated = True
    self.command = 1.

  # Open gripper fingers.
  def release(self):
    pybullet.setJointMotorControl2(self.body, self.motor_joint, pybullet.VELOCITY_CONTROL, targetVelocity=-1, force=10)
    self.activated = False
    self.command = -1.
    self._release_assisted_grasp()

  def move_angle(self, angle: float):
    # directly set motor joint position
    pybullet.setJointMotorControl2(
      self.body,
      self.motor_joint,
      pybullet.POSITION_CONTROL,
      targetPosition=angle,
      force=100
    )

  # If activated and object in gripper: check object contact.
  # If activated and nothing in gripper: check gripper contact.
  # If released: check proximity to surface (disabled).
  def detect_contact(self):
    obj, _, ray_frac = self.check_proximity()
    if self.activated:
      empty = self.grasp_width() < 0.01
      cbody = self.body if empty else obj
      if obj == self.body or obj == 0:
        return False
      return self.external_contact(cbody)
  #   else:
  #     return ray_frac < 0.14 or self.external_contact()

  # Return if body is in contact with something other than gripper
  def external_contact(self, body=None):
    if body is None:
      body = self.body
    pts = pybullet.getContactPoints(bodyA=body)
    pts = [pt for pt in pts if pt[2] != self.body]
    return len(pts) > 0  # pylint: disable=g-explicit-length-test

  def check_grasp(self):
    while self.moving():
      time.sleep(0.001)
    success = self.grasp_width() > 0.01
    return success

  def grasp_width(self):
    lpad = np.array(pybullet.getLinkState(self.body, 4)[0])
    rpad = np.array(pybullet.getLinkState(self.body, 9)[0])
    dist = np.linalg.norm(lpad - rpad) - 0.047813
    return dist

  def check_proximity(self):
    ee_pos = np.array(pybullet.getLinkState(self.robot, self.tool)[0])
    tool_pos = np.array(pybullet.getLinkState(self.body, 0)[0])
    vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
    ee_targ = ee_pos + vec
    ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]
    obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
    return obj, link, ray_frac
