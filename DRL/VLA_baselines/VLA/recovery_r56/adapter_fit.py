"""Native Adapter continuation with an explicit stage-relative LR restart.

Real cumulative optimizer counters remain unchanged. Only the learning-rate
schedule is restarted for this declared training stage, not the model,
observations, loss, action representation or controller.
"""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import read


def stage_scale(local_update, budget=1500, warmup=100):
    if not 0 <= local_update <= budget:
        raise ValueError('Stage-relative schedule outside actual finite budget')
    if local_update < warmup:
        return .1+.9*(local_update+1)/warmup
    fraction=(local_update-warmup)/(budget-warmup)
    return .1+.9*.5*(1+math.cos(math.pi*fraction))


def native_cosine(global_angle, origin, budget=1500):
    actual_step=(global_angle/math.pi)*(origin+budget-200)+200
    local=min(float(budget),max(0.,actual_step-origin))
    return 2*(stage_scale(local,budget)-.1)/.9-1


def main(a):
    import VLA.new_vlas.train as native
    from VLA.autocycle_r55.fit_native import main as fit
    parent=read(a.resume/'config.json')
    origin=parent.get('r56_stage_relative_LR',{}).get('origin_actual_optimizer_update',parent['actual_optimizer_updates'])
    if a.family!='vla_adapter' or a.total_updates!=origin+1500 or a.updates>a.total_updates:
        raise ValueError('One explicitly bounded native Adapter1500-update training stage')
    # The pinned trainer normally derives cosine phase from the CUMULATIVE
    # counter. This process-local math namespace rebases only that expression.
    # It does not modify Python math globally or reset Adam/RNG/counters.
    native.math=SimpleNamespace(pi=math.pi,cos=lambda angle:native_cosine(angle,origin))
    original=native.atomic_json
    def atomic(path,value):
        if Path(path).name in ('config.json','plan.json'):
            value={**value,'r56_stage_relative_LR':{
                'origin_actual_optimizer_update':origin,'budget_new_actual_updates':1500,
                'warmup_new_updates':100,'peak_lr':a.learning_rate,'floor_ratio':.1,
                'Adam_RNG_and_cumulative_counters_resumed_NOT_reset':True,
                'native_L1_objective_and_architecture_unchanged':True}}
            if Path(path).name=='config.json':
                value['code'][str(Path(__file__).resolve())]=sha256_file(__file__)
        original(path,value)
    native.atomic_json=atomic
    fit(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('vla_adapter',),default='vla_adapter')
    for name in ('data','output','resume','input-proof'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reload-proof',type=Path)
    p.add_argument('--mode',choices=('gate','long'),required=True)
    p.add_argument('--updates',type=int,required=True)
    p.add_argument('--total-updates',type=int,required=True)
    p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--sampling-profile',choices=('acquisition',),default='acquisition')
    main(p.parse_args())
