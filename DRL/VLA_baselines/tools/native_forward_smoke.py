"""One genuine native-policy forward on a measured PyBullet observation; no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from protocol21.specs import SPECS
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=tuple(SPECS),required=True)
    p.add_argument('--resources',type=Path,default=ROOT)
    p.add_argument('--foundation',type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    from VLA.common.body_environment import BodyGraspEnvironment
    from VLA.common.tasks import TASKS
    from protocol21.sensors import capture
    from protocol21.inputs import Inputs
    from protocol21.state21_reference import fit_train_stats
    from protocol21.native import create
    from protocol21.execution import prediction
    env=BodyGraspEnvironment(observation_profile=ROOT/'VLA/profiles/body_contact_v9_wrist65_f80_encoder.json',
                            robot='ur5e',image_size=384,physics_hz=240,control_hz=10,seed=20261126)
    try:
        obs=env.reset(TASKS['pick'])
        f=capture(env,obs,depth=SPECS[a.model].depth)
        # Test-only normalizers make this a wiring check, never a reported policy result.
        action={'mean':[0.]*7,'std':[1.]*7,'q01':[-.05]*6+[-1.],'q99':[.05]*6+[1.]}
        stats={'robot21':fit_train_stats([f.raw_state21]),'actions':action,'action':action,
               'adapter':{'proprio':{'q01':[-1.]*8,'q99':[1.]*8,'mask':[True]*8},
                          'action':{'q01':[-.05]*6+[0.],'q99':[.05]*6+[1.],
                                    'min':[-.05]*6+[0.],'max':[.05]*6+[1.],
                                    'mask':[True]*6+[False]}},
               'local_action_scale6':[.05]*6}
        inputs=Inputs(a.model,stats);inputs.push(f)
        runtime=create(a.model,stats,resources=a.resources,foundation=a.foundation,device=a.device)
        result=prediction(runtime,a.model,inputs.sample(),stats,seed=20261126)
        report={'mode':'actual_native_forward_smoke','family':a.model,'device':a.device,
                'real_PyBullet_RGB_and_robot21':True,'actual_native_policy_inference':True,
                'native_prediction_shape':list(result.shape),'finite':bool(np.isfinite(result).all()),
                'optimizer_steps':0,'task_actions_executed':0,'success_rate_measured':False,
                'normalization':'smoke-only fixture; not eligible for benchmark or training',
                'native_input_receipt':getattr(getattr(runtime,'model',None),'last_input_shapes',None)}
    finally:env.close()
    message=json.dumps(report,indent=2,allow_nan=False)
    if a.output:
        with a.output.open('x') as stream:stream.write(message+'\n')
    print(message)


if __name__=='__main__':main()
