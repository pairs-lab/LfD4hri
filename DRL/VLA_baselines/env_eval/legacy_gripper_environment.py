"""Use the user's existing utils.Robotiq2F85 class; no merged robot/gripper."""
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from VLA.common.environment import ASSET_ROOT
from VLA.common.spatial import FIXED_DESTINATION
from utils.Robotiq2F85 import Robotiq2F85
from utils.Robotiq2F85UF850 import Robotiq2F85UF850
from env_eval.oracle_demo import DemoEnv


class LegacyGripperEnvironment(DemoEnv):
    def __init__(self,robot,asset_dir,video=False):
        super().__init__(robot,asset_dir,video)
        self.max_mount_translation_error=0.
        self.max_mount_rotation_error=0.

    def _spawn_object(self,name,position,fixed):
        body=super()._spawn_object(name,position,fixed)
        if fixed and name.endswith(' bowl'):
            # Task positions are spawn heights for falling objects. Static
            # bowls cannot fall, so seat their collision bottom on the table.
            pos,quat=p.getBasePositionAndOrientation(body)
            grounded=np.asarray(pos,dtype=float)
            grounded[2]+=p.getAABB(self.table_id)[1][2]-p.getAABB(body)[0][2]
            p.resetBasePositionAndOrientation(body,grounded,quat)
        return body

    def reset(self, task):
        self.task = task
        self.step_count = 0
        self._last_success = False
        self.success_streak = 0
        self.object_ids = {}
        self._last_segmentation = {}
        self.pick_id = None
        self.place_id = None
        self.initial_place_top = None

        p.resetSimulation(physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / self.physics_hz)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            contactERP=0.9,
            solverResidualThreshold=1e-7,
            enableFileCaching=0,
        )

        p.loadURDF(str(ASSET_ROOT / "plane.urdf"), (0, 0, -0.004), useFixedBase=True)
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(0.45, 0.35, 0.004))
        table_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=(0.45, 0.35, 0.004), rgbaColor=(0.2, 0.2, 0.2, 1.0)
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=(0, -0.5, 0),
        )

        self.robot_id = p.loadURDF(
            str(self.robot_spec.urdf),
            (0, 0, 0),
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        )
        for joint_id, home in zip(self.robot_spec.arm_joint_ids, self.robot_spec.home_joints):
            p.resetJointState(self.robot_id, joint_id, home)

        self.pick_id = self._spawn_object(task.pick_name, task.pick_position, fixed=False)
        self.object_ids[task.pick_name] = self.pick_id
        if (
            task.place_name is not None
            and task.place_name not in FIXED_DESTINATION
            and task.place_position is not None
        ):
            self.place_id = self._spawn_object(
                task.place_name,
                task.place_position,
                fixed=task.place_name.endswith(" bowl"),
            )
            self.object_ids[task.place_name] = self.place_id
        for scene_object in task.scene_objects:
            if scene_object.name in self.object_ids:
                raise ValueError(f"Duplicate scene object name {scene_object.name!r}")
            self.object_ids[scene_object.name] = self._spawn_object(
                scene_object.name,
                scene_object.position,
                fixed=scene_object.fixed,
            )

        dynamic=[body for body in self.object_ids.values() if p.getDynamicsInfo(body,-1)[0]>0]
        gripper_class=Robotiq2F85UF850 if self.robot_spec.name=='uf850' else Robotiq2F85
        self.gripper=gripper_class(self.robot_id,self.robot_spec.mount_link,start_thread=False,
            urdf=self.gripper_asset_root/'robotiq_2f_85/robotiq_2f_85.urdf',object_ids=dynamic,assist_grasp=True)
        # Adjacent flange geometry is intentionally overlapping; nonadjacent
        # arm/gripper collision pairs are enabled below for the audit.
        for arm in range(-1,p.getNumJoints(self.robot_id)):
            for finger in range(-1,p.getNumJoints(self.gripper.body)):
                p.setCollisionFilterPair(self.robot_id,self.gripper.body,arm,finger,0)
        # The native cover meshes overlap the rubber contact surface. Rigid
        # cover contacts fight the compliant pads and unload a finger.
        for link in range(p.getNumJoints(self.gripper.body)):
            p.changeDynamics(self.gripper.body,link,contactStiffness=3000.,contactDamping=20.)

        self._apply_arm_targets(np.asarray(self.robot_spec.home_joints, dtype=np.float32))
        self.gripper.open()
        for _ in range(self.physics_hz):
            self.gripper.apply_motor_control()
            p.stepSimulation()
        self.initial_pick_z = p.getBasePositionAndOrientation(self.pick_id)[0][2]
        self.initial_pick_position = np.asarray(
            p.getBasePositionAndOrientation(self.pick_id)[0], dtype=np.float64
        )
        self.initial_pick_top = self._pick_top()
        self.initial_pick_bottom = self._pick_bottom()
        self.initial_place_top = self._body_top(self.place_id) if self.place_id is not None else None
        base_dist, base_obj = self._read_reward_thresholds(task.task_id)
        self.dist_threshold = base_dist * self.success_threshold_scale
        self.obj_threshold = base_obj * self.success_threshold_scale
        self.q=np.array([p.getJointState(self.robot_id,j)[0] for j in self.robot_spec.arm_joint_ids])
        self.q_velocity=np.zeros(6)
        self.filtered_tcp_offset=None;self.touch_reference=None;self.pressure_constraint=None
        self.self_pairs=[];self.arm_gripper_pairs=[]
        parents={i:p.getJointInfo(self.robot_id,i)[16] for i in range(p.getNumJoints(self.robot_id))}
        def distance(a,b):
            ca=[a];cb=[b]
            while ca[-1]!=-1:ca.append(parents[ca[-1]])
            while cb[-1]!=-1:cb.append(parents[cb[-1]])
            return min(ca.index(x)+cb.index(x) for x in set(ca)&set(cb))
        links=[i for i in range(-1,p.getNumJoints(self.robot_id)) if p.getCollisionShapeData(self.robot_id,i)]
        for k,a in enumerate(links):
            for b in links[k+1:]:
                if distance(a,b)>2:
                    self.self_pairs.append((a,b));p.setCollisionFilterPair(self.robot_id,self.robot_id,a,b,1)
        self.arm_gripper_pairs=[link for link in links if distance(link,self.robot_spec.mount_link)>2]
        for link in self.arm_gripper_pairs:
            for finger in range(-1,p.getNumJoints(self.gripper.body)):
                p.setCollisionFilterPair(self.robot_id,self.gripper.body,link,finger,1)
        p.changeVisualShape(self.table_id,-1,rgbaColor=(.32,.35,.38,1.))
        if self.robot_spec.name=='uf850':
            for link in links:
                p.changeVisualShape(self.robot_id,link,rgbaColor=(.80,.83,.87,1.))
        # UR5e uses its original URDF/MTL material palette, without recoloring.
        self.initial_objects={name:np.array(p.getBasePositionAndOrientation(body)[0]) for name,body in self.object_ids.items()}
        return self.observe()

    def tick(self,center,quaternion,command):
        # Avoid an implicit full-open command on the exact release tick.
        if command<-.2 and self.gripper.grasp_constraint is not None:
            self.gripper._release_assisted_grasp()
        if self.gripper.grasp_constraint is not None and self.gripper.grasp_constraint!=self.pressure_constraint:
            # A small preload maintains contact as compliant pads settle.
            self.gripper._latched+=.006*self.gripper.SIGNS
            self.pressure_constraint=self.gripper.grasp_constraint
        current,qcur=self.pose()
        raw=Rotation.from_quat(qcur).inv().apply(self.gripper.grasp_center()-current)
        if self.filtered_tcp_offset is None:self.filtered_tcp_offset=raw.copy()
        else:self.filtered_tcp_offset+=.2*(raw-self.filtered_tcp_offset)
        if self.touch_reference is not None:
            reference=self.touch_reference
            mount=reference['mount']+np.asarray(center)-reference['tcp']
            center=mount+Rotation.from_quat(quaternion).apply(raw)
        else:
            center=np.asarray(center)+Rotation.from_quat(quaternion).apply(raw-self.filtered_tcp_offset)
        super().tick(center,quaternion,command)
        self.max_mount_translation_error=self.gripper.max_mount_translation_error
        self.max_mount_rotation_error=self.gripper.max_mount_rotation_error

