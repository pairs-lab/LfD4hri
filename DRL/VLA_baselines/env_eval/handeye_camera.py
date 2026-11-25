"""Rigid hand-eye extrinsics using the same mount COM frame as VLA."""
import pybullet as p
from VLA.common.environment import CAMERA_CONFIG


def handeye_geometry(env,settings=None):
    cfg=settings or CAMERA_CONFIG['wrist']
    state=p.getLinkState(env.robot_id,env.robot_spec.mount_link,computeForwardKinematics=True)
    def point(local):return p.multiplyTransforms(state[0],state[1],local,(0,0,0,1))[0]
    up=p.multiplyTransforms((0,0,0),state[1],cfg['up_axis_in_tool'],(0,0,0,1))[0]
    return dict(eye=point(cfg['eye_in_tool']),target=point(cfg['target_in_tool']),up=up,fov=cfg['fov_degrees'])
