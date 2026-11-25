"""Restore official VLANeXt gripper binarization before common Robotiq adapter.

R84 emitted continuous signed-close into a +/-0.2 hysteresis actuator; official
parallel_eval_utils._denormalize_action uses >0 -> +1, else -1. Pose predictions,
weights, native history/preprocessing and PyBullet physics stay unchanged.
"""
import argparse
from pathlib import Path
import numpy as np


def native_gripper(actions):
    actions=np.asarray(actions,np.float32).copy()
    if actions.ndim!=2 or actions.shape[1]!=7 or not np.isfinite(actions).all():
        raise ValueError('Finite native canonical chunk required')
    actions[:,6]=np.where(actions[:,6]>0,1.,-1.)
    return actions


def main(a):
    import VLA.replacement_r84.server as old
    runtime=old.Runtime
    class OfficialGripperRuntime(runtime):
        def predict(self,s,seed):
            actions=native_gripper(super().predict(s,seed))
            self.last_input={**self.last_input,'gripper_postprocess':'official_sign_gt0_+1_else_-1',
                'new_target_based_action_or_physics_change':False}
            return actions
    old.Runtime=OfficialGripperRuntime
    try:old.main(a)
    finally:old.Runtime=runtime


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
