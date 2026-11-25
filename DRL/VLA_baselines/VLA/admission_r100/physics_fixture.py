"""Actual same-robot actuator and RGB-D geometry checks, not policy trials."""
import numpy as np
import pybullet as p
from .common import OUT,read,atomic_json,utc
from .contracts import decode_seer,measured_pose
from VLA.common.environment_factory import environment_class_for_profile
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.scene64_conditioning import model_input
from VLA.falcon_esm_r91.depth import capture


def main():
    protocol=read('VLA/evaluation_protocol_v9_postfit_10_per_action.json')
    profile=read(protocol['observation_profile_file']);rng=np.random.default_rng(20309999)
    task=randomize_rl_scene(select_tasks('all')[0],rng)
    env=environment_class_for_profile(profile)(observation_profile=profile,robot='ur5e',image_size=384,
        control_hz=10,assist_grasp=True,seed=20309999,required_success_steps=10000,
        use_semantic_success=False,render_observations=True)
    values=[];geometry=[]
    try:
        obs=env.reset(task);raw=model_input(env,obs)['scene64'];initial=raw.copy()
        # Same canonical actuator, native Seer OPEN polarity conversion.
        for native_open in (-1.,1.):
            for _ in range(8):
                pose=measured_pose(obs.ee_position,obs.ee_rotation_6d)
                local=np.zeros((3,7));local[:,6]=native_open
                command,_=decode_seer(local,pose,pose)
                obs,_,_,info=env.step(command[0])
                if info.get('safety_violation'):raise ValueError('Free-space gripper fixture unsafe')
            raw=model_input(env,obs)['scene64']
            values.append({'native_signed_OPEN':native_open,'canonical_signed_CLOSE':float(command[0,6]),
                'measured_closure':float(raw[6]),'jaw_gap_m':float(raw[27])})
        if values[0]['jaw_gap_m']+.025>=values[1]['jaw_gap_m']:
            raise ValueError('Native polarity did not produce actual physical closing then opening')
        # Render same actual front camera and verify backprojected target
        # surface lies in its measured world AABB (half-pixel tolerance noted).
        actual=env.observe();depth,view,projection,K,cv=capture(env,actual)
        result=p.getCameraImage(384,384,viewMatrix=view.tolist(),projectionMatrix=projection.tolist(),
            renderer=p.ER_TINY_RENDERER,flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,physicsClientId=env.client_id)
        segmentation=np.asarray(result[4]).reshape(384,384);visible=(segmentation&((1<<24)-1))==env.pick_id
        ys,xs=np.nonzero(visible)
        if len(xs)<8:raise ValueError('Geometry fixture target not visible')
        z=depth[ys,xs];camera=np.c_[(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z,np.ones(len(z))]
        world=(np.linalg.inv(cv)@camera.T).T[:,:3]
        lower,upper=np.asarray(p.getAABB(env.pick_id,physicsClientId=env.client_id))
        outside=np.maximum(np.maximum(lower-world,world-upper),0)
        if float(outside.max())>.008:raise ValueError('OpenGL/OpenCV/depth geometry lies outside actual target AABB')
        geometry={'target_pixels':len(xs),'maximum_outside_AABB_m':float(outside.max()),
            'projection_pixel_center_convention':'(width-1)/2; tolerance includes raster/half-pixel boundary',
            'metric_depth_K_extrinsics_same_actual_camera':True}
    finally:env.close()
    atomic_json(OUT/'ACTUAL_PHYSICS_FIXTURE.json',{'passed':True,'UTC':utc(),'physical_gripper_checks':values,
        'actual_RGBD_geometry':geometry,'actual_policy_episodes':0,'new_success_rate_claimed':False,
        'robot_assets_controller_or_primary_protocol_changed':False})
    print({'passed':True,'gripper':values,'geometry':geometry},flush=True)


if __name__=='__main__':main()
