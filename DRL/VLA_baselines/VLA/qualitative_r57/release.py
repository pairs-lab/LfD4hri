"""User-authorized qualitative-only terminal release, NEVER VLA transport.

Only holds the measured arm pose and opens. No horizontal/vertical correction,
target waypoint, force injection or teleport. Unsafe alignment remains failure.
"""
import numpy as np
import pybullet as p


def landing_gate(object_xy,goal_xy,place_name,place_bounds=None,object_extent=None):
    xy=np.asarray(object_xy,float);goal=np.asarray(goal_xy,float)
    error=float(np.linalg.norm(xy-goal))
    if place_bounds is None:
        return error<=.045,{'kind':'table_region','maximum_center_error_m':.045,'actual_center_error_m':error}
    low,high=map(lambda x:np.asarray(x,float),place_bounds)
    if 'bowl' in (place_name or ''):
        radius=.72*float(min((high-low)[:2])/2)
        size=0. if object_extent is None else float(np.linalg.norm(np.asarray(object_extent)[:2])/2)
        # Conservative72% of outer radius, object bounding radius AND2mm
        # clearance. Final support/settle is measured; gate approval alone
        # never becomes a success label.
        allowed=max(0.,radius-size-.002)
        return error<=allowed,{'kind':'conservative_bowl_opening','allowed_center_error_m':allowed,'actual_center_error_m':error}
    inside=bool(np.all(xy>=low[:2]+.003)&np.all(xy<=high[:2]-.003))
    return inside,{'kind':'support_center_inside_destination_AABB_3mm_margin','actual_center_error_m':error,
        'destination_xy_low':low[:2].tolist(),'destination_xy_high':high[:2].tolist()}


def physical_metrics(env):
    position=np.asarray(p.getBasePositionAndOrientation(env.pick_id)[0])
    goal=env._object_goal()
    supporters=[env.place_id] if env.place_id is not None else env.ground_bodies
    contacts=[c for body in supporters for c in p.getContactPoints(env.pick_id,body)]
    return {'object_position':position.tolist(),'goal':goal.tolist(),
        'goal_XY_m':float(np.linalg.norm(position[:2]-goal[:2])),
        'speed_m_per_s':float(np.linalg.norm(p.getBaseVelocity(env.pick_id)[0])),
        'support_force_N':sum(max(0.,c[9]) for c in contacts),
        'correct_attached':bool(env.gripper.grasped_object==env.pick_id),
        'gripper_command':float(env.gripper.command)}


def complete(metrics,held_history):
    if not held_history or len(metrics)<5:return False
    return all(not r['correct_attached'] and r['goal_XY_m']<=.045 and r['support_force_N']>.005
        and r['speed_m_per_s']<.025 and r['gripper_command']<-.2 for r in metrics[-5:])


def run(env,held_history,before_control=None):
    bounds=p.getAABB(env.place_id) if env.place_id is not None else None
    lo,hi=map(np.asarray,p.getAABB(env.pick_id))
    ready,evidence=landing_gate(p.getBasePositionAndOrientation(env.pick_id)[0][:2],env._object_goal()[:2],
        env.task.place_name,bounds,hi-lo)
    verified=bool(env.gripper.grasped_object==env.pick_id and env.gripper._bilateral(env.pick_id))
    if env.safety_latched or not ready or not verified or not held_history:
        return {'attempted':False,'landing_gate':evidence,'verified_correct_hold_at_start':verified,
            'reason':'no_safe_aligned_verified_hold; no transport correction','qualitative_put_complete':False,'controls':[]}
    controls=[];metrics=[]
    for i in range(20):
        if before_control:before_control(i)
        action=np.zeros(7,np.float32);action[6]=1-2*min((i+1)/8,1.)
        obs,_,done,info=env.step(action)
        actual=physical_metrics(env);metrics.append(actual)
        controls.append({'control_source':'deterministic_terminal_release_NOT_neural',
            'requested_action':action.tolist(),'executed_action':info['executed_canonical_action'],
            'original_benchmark_done_flag':bool(done),'physical_metrics':actual,'safety_violation':info.get('safety_violation')})
        if info.get('safety_violation'):break
    return {'attempted':True,'landing_gate':evidence,'verified_correct_hold_at_start':verified,
        'maximum_extension_seconds':2.,'arm_translation_or_rotation_commanded':False,
        'qualitative_put_complete':complete(metrics,held_history),'controls':controls,
        'method_label':'X-VLA + terminal release supervisor; NOT pure VLA; NOT quantitative method'}
