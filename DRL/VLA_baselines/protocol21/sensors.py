"""Read-only robot/FK/contact capture. No target, goal or attachment identity is read."""
import numpy as np
from .state21_reference import pack
from .inputs import Frame


def capture(env, observation, *, depth=False):
    import pybullet as p
    from VLA.common.body_gripper import BodyContactRobotiq
    if not isinstance(env.gripper, BodyContactRobotiq):
        raise ValueError('Calibrated surface-pad BodyContactRobotiq is required; centroid width is not jaw gap')
    client = env.client_id
    g = env.gripper
    right, left = g.pad_positions()  # PAD_LINKS=(right4,left9), not left/right!
    gap = g.width()
    tool = p.getLinkState(env.robot_id, env.robot_spec.mount_link,
                          computeForwardKinematics=True, physicsClientId=client)
    R = np.asarray(p.getMatrixFromQuaternion(tool[5])).reshape(3, 3)
    forces = {link: 0. for link in g.PAD_LINKS}
    # Query all bodies. Robot/gripper self contacts are excluded, not wrong objects/table.
    for c in p.getContactPoints(bodyA=g.body, physicsClientId=client):
        if c[2] not in (g.body, env.robot_id) and c[3] in forces:
            forces[c[3]] += max(0., float(c[9]))
    raw = pack(q=observation.joint_positions,
               closure01=1 - np.clip((gap-.0063742693)/(.0885956060-.0063742693), 0, 1),
               pad_midpoint_world=(left+right)/2, previous_signed_close=g.command,
               tool_to_world_rotation=R, surface_jaw_gap_m=gap,
               left_external_normal_force_N=forces[9], right_external_normal_force_N=forces[4])
    optional = {}
    if depth:
        # Independent read-only camera capture; does not overwrite observation segmentation.
        from VLA.common.observation_profile import camera_geometry
        cfg = env.observation_profile['cameras']['front']
        eye, target, up = camera_geometry(env, cfg, 'front')
        w, h = observation.front_rgb.shape[1], observation.front_rgb.shape[0]
        view = p.computeViewMatrix(eye, target, up)
        proj = p.computeProjectionMatrixFOV(cfg['fov_degrees'], w/h, .01, 3.)
        im = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER,
                              flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                              physicsClientId=client)
        rgb = np.asarray(im[2], np.uint8).reshape(h, w, 4)[..., :3]
        if not np.array_equal(rgb, observation.front_rgb):
            raise ValueError('Depth capture RGB differs from policy front; verify camera near/far and renderer')
        z = np.asarray(im[3], np.float64).reshape(h, w)
        metric = .01*3./(3.-(3.-.01)*z)
        from VLA.falcon_esm_r91.data import geometry
        K, E = geometry(view, proj, w, h)
        optional = dict(depth_z_m=metric.astype(np.float32), intrinsics=K, extrinsics_OpenCV=E)
    return Frame(raw, np.asarray(tool[4], np.float32), observation.front_rgb,
                 observation.wrist_rgb, observation.prompt, env.step_count/env.control_hz, **optional).checked()
