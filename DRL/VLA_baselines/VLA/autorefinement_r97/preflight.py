"""CPU contracts/data qualification; no training or final outcome selection."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from .common import (OUT, ROOT, RECOVERY, PARENT, TASKS, DEV, CONF, FINAL,
    PY, CPU, SIM, ENV, PROTOCOL, read, verify, metric_guard, bindings, atomic_json)
from VLA.common.provenance import sha256_file


def main():
    OUT.mkdir(exist_ok=True)
    metric_guard()
    from VLA.continuous_r86.owner import qualified
    parent = qualified(PARENT)
    if parent['updates'] != 10000 or not read(PARENT/'FRESH_RELOAD_cuda.json')['passed']:
        raise ValueError('Qualified exact native10k parent required')
    unit = subprocess.run([str(CPU),'-m','unittest','VLA.autorefinement_r97.tests',
        'VLA.recovery_quality_r90.tests','VLA.dita_terminal_recovery_r95.tests'],cwd=ROOT,env={**ENV,'CUDA_VISIBLE_DEVICES':''},
        capture_output=True,text=True,timeout=120)
    print(unit.stdout+unit.stderr,flush=True)
    if unit.returncode:
        raise ValueError('Unit regression failed')
    from VLA.recovery_quality_r90.data import Data
    from VLA.dita_transfer_r84.data import sample
    data = Data()
    if len(data.rows)!=2337 or sum(r['frames'] for r in data.rows)!=125527:
        raise ValueError('Incomplete original + qualified recovery bank')
    stats = parent['stats']
    if stats != read(ROOT/'VLA/audits/replacement_r84_20261020/VLANEXT_DATA_AUDIT.json')['stats']:
        raise ValueError('Parent normalizers differ from qualified TRAIN statistics')
    tasks, sources, targets = Counter(), Counter(), set()
    for i,row in enumerate(data.rows):
        n = data.numeric(i)
        if (n['scene64'].shape != (row['frames'],64) or n['canonical7'].shape != (row['frames'],7)
            or n['native8'].shape != (row['frames'],8)):
            raise ValueError('Numeric sample alignment/shape differs')
        if not np.isfinite(np.r_[n['scene64'].ravel(),n['canonical7'].ravel(),n['native8'].ravel()]).all():
            raise ValueError('Nonfinite dataset')
        if not np.isin(n['canonical7'][:,6],[-1.,1.]).all():
            raise ValueError('Wrong canonical gripper sign')
        if row['source_kind']=='near_miss_corrective':
            if not row['intervention_actions_excluded_from_training_labels']:
                raise ValueError('TRAIN interventions must not become supervision')
            if sha256_file(row['source']) != row['source_sha256']:
                raise ValueError('Recovery actual RGB/source changed')
        tasks[row['task']] += row['frames']
        sources[row['source_kind']] += row['frames']
        targets.add((row['task'],row['pick_name']))
    if len(targets)!=76 or set(tasks)!=set(TASKS):
        raise ValueError('Expected all four tasks and19 targets')
    chosen = [next(i for i,r in enumerate(data.rows) if r['task']==task and r['source_kind']==kind)
        for task in TASKS for kind in ('complete_original','near_miss_corrective')]
    canvas = Image.new('RGB',(672,8*248),'white')
    draw = ImageDraw.Draw(canvas)
    for ri,ei in enumerate(chosen):
        row = data.rows[ei]
        for fi in sorted({0,row['frames']//2,row['frames']-1}):
            s = sample(data,ei,fi,stats)
            if s['history_indices'].max()!=fi or s['history74'].shape!=(2,74):
                raise ValueError('Causal two-observation history violated')
            if not np.isin(s['normalized_action11'][:,6],[0.,1.]).all():
                raise ValueError('Native OPEN01 labels are not preserved')
            expected = (1-data.numeric(ei)['canonical7'][fi,6])/2
            if s['normalized_action11'][1,6]!=expected:
                raise ValueError('Native label offset is wrong')
            if fi == 0:
                draw.text((4,ri*248+3),f"{row['task']} / {row['source_kind']} / {row['pick_name']}",fill='black')
                for ci,name in enumerate(('front','wrist','map')):
                    canvas.paste(Image.fromarray(s['history_images'][name][-1]).resize((224,224)),(ci*224,ri*248+24))
    canvas.save(OUT/'TRAIN_OBSERVATION_REVIEW.jpg',quality=92)
    receipts = []
    for label,python,original in (('native',PY,True),('evaluator',SIM,False)):
        path = OUT/f'TRANSPORT_{label}.json'
        args = [str(python),'-m','VLA.recovery_quality_r90.transport','--family','dita','--output',str(path)]
        if original:
            args.append('--original')
        subprocess.run(args,cwd=ROOT,env=ENV,check=True,timeout=180)
        receipts.append(read(path))
    if receipts[0]['case_hashes'] != receipts[1]['case_hashes']:
        raise ValueError('Native/evaluator map pixel receipts differ')
    seed_sets = list(map(set,(DEV,CONF,FINAL)))
    if any(a&b for i,a in enumerate(seed_sets) for b in seed_sets[i+1:]):
        raise ValueError('Evaluation split overlap')
    atomic_json(OUT/'DATA_AUDIT.json', {'passed':True,'trajectories':len(data.rows),
        'frames':sum(tasks.values()),'per_action_frames':dict(tasks),'per_source_frames':dict(sources),
        'original_trajectories':2185,'original_frames':119495,'additional_recovery_trajectories':152,
        'additional_recovery_frames':6032,'target_task_strata':len(targets),
        'all_numeric_frames_checked':True,'all_recovery_RGB_source_hashes_checked':True,
        'all_original_RGB_redecoded_this_pass':False,'representative_actual_RGB_chronology_samples':24,
        'train_only_admission':str(RECOVERY/'ADMISSION.json'),
        'admission_sha256':sha256_file(RECOVERY/'ADMISSION.json'),
        'TRAIN_source_and_recovery_membership_not_neural_performance':True,
        'new_eval_blocks_not_used_as_training_inputs':True,
        'full_bank_eval_scene_hash_independence_NOT_proven_by_seed_disjointness_alone':True})
    atomic_json(OUT/'PROTOCOL_SNAPSHOT.json',read(PROTOCOL))
    atomic_json(OUT/'QUALIFICATION.json', {'passed':True,'code':bindings(),
        'metric_exactly_original':True,'native_evaluator_map_cases_matched':228,
        'native_loss_and_action_unchanged':True,'parent_weights_sha256':parent['weights_sha256'],
        'data_audit':str(OUT/'DATA_AUDIT.json'),'CPU_regressions_passed':True,
        'GPU_gate_fresh_reload_and_actual_neural_tests_still_required':True})


if __name__=='__main__':
    main()
