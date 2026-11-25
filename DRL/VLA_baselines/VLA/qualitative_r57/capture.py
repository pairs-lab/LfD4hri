"""Read-only real-tick telemetry, shared by verified VLA replay and Oracle."""
import numpy as np
import pybullet as p
from VLA.qualitative_recorder_r51 import StateRecorder
from VLA.common.artifact_io import atomic_json


class EvidenceRecorder(StateRecorder):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.dense=[]
        self.phase='VLA_native_action_replay'

    def record_tick(self,tick,control):
        env=self.env
        target_contacts=p.getContactPoints(env.gripper.body,env.pick_id)
        ground=[c for body in env.ground_bodies for c in p.getContactPoints(env.gripper.body,body)]
        self.dense.append({'physics_tick':tick,'simulation_time_s':tick/env.physics_hz,'control':control,
            'control_source':self.phase,'object_position':list(p.getBasePositionAndOrientation(env.pick_id)[0]),
            'object_velocity':list(p.getBaseVelocity(env.pick_id)[0]),
            'pad_center':env.gripper.grasp_center().tolist(),
            'target_force_N':sum(max(0.,c[9]) for c in target_contacts),
            'ground_force_N':sum(max(0.,c[9]) for c in ground),
            'correct_verified_hold':bool(env.gripper.grasped_object==env.pick_id and env.gripper._bilateral(env.pick_id))})

    def capture(self,tick,control):
        super().capture(tick,control)
        env=self.env
        target=env.pick_id;robot=env.gripper.body
        row=self.frames[-1]
        position,quat=p.getBasePositionAndOrientation(target)
        linear,angular=p.getBaseVelocity(target)
        contacts=p.getContactPoints(bodyA=target)
        support=[c for c in contacts if c[2] in ([env.place_id] if env.place_id is not None else env.ground_bodies)]
        ground=[c for body in env.ground_bodies for c in p.getContactPoints(robot,body)]
        pad=env.gripper.grasp_center()
        row.update(simulation_time_s=tick/env.physics_hz,control_source=self.phase,object_position=list(position),
            object_orientation=list(quat),object_linear_velocity=list(linear),object_angular_velocity=list(angular),
            pad_center=pad.tolist(),goal_position=env._object_goal().tolist(),
            correct_verified_hold=bool(row['correct_attached'] and row['bilateral']),
            gripper_command=float(env.gripper.command),
            destination_support_force_N=sum(max(0.,c[9]) for c in support),
            gripper_ground_force_N=sum(max(0.,c[9]) for c in ground),
            requested_object=env.task.pick_name,
            held_object=next((name for name,body in env.object_ids.items() if body==env.gripper.grasped_object),None),
            other_object_positions={name:list(p.getBasePositionAndOrientation(body)[0])
                for name,body in env.object_ids.items() if body not in (target,env.place_id)},
            capture_is_read_only_no_images_or_policy_cache_mutation=True)

    def close(self):
        super().close()
        atomic_json(self.directory/'physics_event_ticks.json',self.dense)


def visibility(env):
    result={}
    for role,seg in env._last_segmentation.items():
        ids=seg&((1<<24)-1);ys,xs=np.where(ids==env.pick_id)
        result[role]={'target_pixels':len(xs),'target_box':
            [int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
            'scope':'actual_policy_observation_segmentation_at_10Hz_NOT_presentation_handeye'}
    return result
