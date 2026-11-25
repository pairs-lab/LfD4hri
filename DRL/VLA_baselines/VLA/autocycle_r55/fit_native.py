"""Process-local sampler continuation of exact native X/Adapter/Adam state."""
import argparse
from pathlib import Path
from .common import read,hashes
from .sampling import draw,contract


def main(a):
    import VLA.fit_feedback_r52 as feedback
    import VLA.new_vlas.train as native
    from VLA.privileged_visual_r53.fit import main as fit
    parent=read(a.resume/'config.json');before=parent['actual_optimizer_updates']
    if a.mode=='gate' and a.updates-before!=2:raise ValueError('Exactly two NEW real Adam updates required')
    if a.updates-before>1500 or a.updates<=before:raise ValueError('Explicit finite continuation budget')
    profile=a.sampling_profile
    def selected_draw(data,rng):return draw(data,rng,profile)
    feedback.draw=selected_draw;feedback.SAMPLER=contract(profile)
    atomic=native.atomic_json
    def save(path,value):
        if Path(path).name in ('config.json','plan.json'):
            value={**value,'r55_continuation':{'parent':str(a.resume.resolve()),'parent_actual_updates':before,
                'profile':contract(profile),'all_motor_commands_remain_actual_native_VLA':True}}
            if Path(path).name=='config.json':value['code'].update(hashes())
        atomic(path,value)
    native.atomic_json=save
    fit(a)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('xvla','vla_adapter'),required=True)
    for name in ('data','output','resume','input-proof'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reload-proof',type=Path);p.add_argument('--mode',choices=('gate','long'),required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--total-updates',type=int,required=True)
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--sampling-profile',choices=('acquisition','transport'),required=True);main(p.parse_args())
