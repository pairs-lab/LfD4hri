"""Same76 TRAIN strata/228 map cases in each actual native NumPy environment."""
import argparse,hashlib
import numpy as np
from .common import OUT,read,atomic_json,utc
from VLA.recovery_quality_r90.audit import BANK


def main(a):
    if a.family=='simulation_reference':from VLA.receipt_recovery_r88.map_reference import render
    else:from VLA.privileged_visual_r53.scene_map import render
    seen=set();cases=[];digest=hashlib.sha256()
    for ei,row in enumerate(read(BANK/'index.json')):
        key=row['task'],row['pick_name']
        if row['source_kind']!='complete_original' or key in seen:continue
        seen.add(key)
        with np.load(BANK/row['cache'],allow_pickle=False) as values:
            raw=values['scene64']
            for fi in sorted({0,len(raw)//2,len(raw)-1}):
                rgb=render(raw[fi],row['task']);digest.update(rgb.tobytes())
                cases.append({'episode':ei,'frame':fi,'task':row['task'],'target':row['pick_name'],
                    'sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
    if len(seen)!=76 or len(cases)!=228:raise ValueError('Full TRAIN rendering strata required')
    value={'passed':True,'UTC':utc(),'family':a.family,'numpy':np.__version__,'cases':cases,
        'combined_sha256':digest.hexdigest(),'observations_or_eval_outcomes_changed':False}
    path=OUT/(a.family+'_MAP_TRANSPORT.json');atomic_json(path,value)
    if a.family!='simulation_reference':
        reference=read(OUT/'simulation_reference_MAP_TRANSPORT.json')
        if reference['cases']!=cases:raise ValueError('Native/simulation renderer byte discrepancy')
    print({'family':a.family,'actual_cases':len(cases),'sha256':digest.hexdigest()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=('seer','gr00t_n15','falcon_esm','simulation_reference'),required=True)
    main(p.parse_args())
