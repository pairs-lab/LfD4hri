"""Run once per NumPy environment; verify exact pixel equality across versions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.privileged_visual_r53.scene_map import render as original
from .map_reference import render

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args()
    root=Path('VLA/audits/receipt_recovery_r88_20261021')
    cases=[]
    failure=json.loads((root/'REPRODUCED_REQUEST.json').read_text())
    cases.append(('pick_place',failure['scene64']))
    # Transport qualification only. No final outcomes, actions or expert phase
    # are fitted/selected; only render the already measured input geometry.
    source=Path('VLA/audits/dita800_user_r87_20261021/final800')
    progress=json.loads((source/'progress.json').read_text())
    for row in progress['rows'][::4]:
        trace=json.loads(Path(row['trajectory']).read_text())[row['task']]
        for frame in trace[::max(1,len(trace)//4)]:cases.append((row['task'],frame['pre_action_scene64']))
    digest=hashlib.sha256();different=0
    for task,raw in cases:
        im=render(raw,task);digest.update(im.tobytes())
        different+=int(not np.array_equal(im,original(raw,task)))
    atomic_json(root/(a.label+'_MAP_REFERENCE.json'),{'numpy':np.__version__,'cases':len(cases),
        'reference_images_combined_sha256':digest.hexdigest(),'reference_vs_original_differing_cases':different,
        'failing_reference_sha256':hashlib.sha256(render(failure['scene64'],'pick_place').tobytes()).hexdigest(),
        'model_inputs_or_physics_changed':False})
