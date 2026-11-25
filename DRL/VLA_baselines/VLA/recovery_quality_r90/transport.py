"""Compare native/reference map bytes on fixed TRAIN states, no model/GPU."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from VLA.common.artifact_io import atomic_json
from .audit import BANK, OUT, TASKS, read
from .evaluate import renderer


def main(a):
    if a.original:
        from VLA.privileged_visual_r53.scene_map import render
    else:
        render = renderer(a.family)
    seen, cases, combined = set(), [], hashlib.sha256()
    for i, row in enumerate(read(BANK/'index.json')):
        key = row['task'], row['pick_name']
        if row['source_kind'] != 'complete_original' or key in seen: continue
        seen.add(key)
        with np.load(BANK/row['cache'], allow_pickle=False) as data:
            raw = data['scene64']
            for fi in sorted({0, len(raw)//2, len(raw)-1}):
                rgb = render(raw[fi], row['task'])
                combined.update(rgb.tobytes())
                cases.append({'task':row['task'],'target':row['pick_name'],'episode':i,'frame':fi,
                    'sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
    if len(seen) != 76 or {t for t, _ in seen} != set(TASKS):
        raise ValueError('Complete76 TRAIN strata required')
    atomic_json(a.output, {'family':a.family,'numpy':np.__version__,'original_policy_renderer':a.original,
        'TRAIN_cases':len(cases),'case_hashes':cases,'combined_sha256':combined.hexdigest(),
        'model_loaded_or_policy_observation_changed':False,'evaluation_states_or_outcomes_used':False})
    print({'family':a.family,'numpy':np.__version__,'TRAIN_cases':len(cases),'digest':combined.hexdigest()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=('dita','vlanext_s'),required=True)
    p.add_argument('--original',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
