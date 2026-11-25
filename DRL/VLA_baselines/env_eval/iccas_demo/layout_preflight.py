"""Cheap all-state geometry/layout audit before the HD render queue."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from VLA.common.artifact_io import atomic_json
from .assets import ICCASEnvironment
from .qualify import OUT,ROBOTS,digest
from .scenes import get_scene
from .render import choose_layouts


def main(out):
    for robot,label in ROBOTS.items():
        with ICCASEnvironment(robot,out/'_derived_assets'/label,False) as env:
            for sid in range(2,11):
                folder=out/label/f'scene{sid:02d}';p=json.loads((folder/'PHYSICS.json').read_text())
                source=Path(p['qualified_directory']);schema=json.loads((source/'REPLAY_SCHEMA.json').read_text())
                samples=json.loads((source/'trajectory.json').read_text());arrays=np.load(source/'frame_states.npz')
                env.reset(get_scene(sid).task());start=time.monotonic()
                layouts=choose_layouts(env,schema,arrays,samples)
                atomic_json(folder/'LAYOUT_PREFLIGHT.json',{'passed':True,'layouts':layouts,
                    'frame_states_sha256':digest(source/'frame_states.npz'),'checked_frames':len(samples),
                    'all_objects_protected_over_complete_trajectory':True,'seconds':time.monotonic()-start})
                print({'robot':robot,'scene':sid,'frames':len(samples),
                    'layout':{v:(l['fraction'],l['side']) for v,l in layouts.items()}},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUT);main(p.parse_args().output)
