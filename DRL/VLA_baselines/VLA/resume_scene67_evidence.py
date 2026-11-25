"""Recover terminal neural episodes written before a statistics-writer crash.

Recovery never simulates a previously completed task or chooses among runs.
Incomplete or inconsistent evidence is an error, not a replacement seed.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.tasks import select_tasks,randomize_rl_scene
from VLA.run_cross_vla120 import SEEDS


def terminal(trace,max_steps):
    if not trace or [r['step'] for r in trace]!=list(range(1,len(trace)+1)):
        raise ValueError('Contiguous original executed controls required')
    last=trace[-1]
    return bool(last['success'] or last.get('safety_violation') or len(trace)==max_steps)


def recover(previous,output,checkpoint,protocol):
    previous,output,checkpoint=map(Path,(previous,output,checkpoint))
    child_path=previous.parent/'active_child.json';child=json.loads(child_path.read_text())
    command=child['command']
    if (command[command.index('--checkpoint')+1]!=str(checkpoint)
        or Path(command[command.index('--output')+1]).resolve()!=previous.resolve()):
        raise ValueError('Previous evaluator used another checkpoint or output')
    # Do not recover while an old writer/episode is still in flight.
    proc=Path('/proc')/str(child['pid'])
    if proc.exists() and (proc/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z':
        raise ValueError('Original evaluator must have exited before evidence-only recovery')
    old=json.loads((previous/'progress.json').read_text());stored={(r['seed'],r['task']):r for r in old['rows']}
    rows=[];preserved={};sources=[]
    for seed in SEEDS:
        rng=np.random.default_rng(seed);tasks=[randomize_rl_scene(t,rng) for t in select_tasks('all')]
        folder=previous/f'test_seed{seed}';path=folder/'trajectory.json'
        if not path.exists():continue
        traces=json.loads(path.read_text())
        if set(traces)-{t.name for t in tasks}:raise ValueError('Unexpected task/seed trajectory')
        for task in tasks:
            trace=traces.get(task.name)
            if trace is None:continue
            if not terminal(trace,task.max_steps):raise ValueError('Partial episode cannot silently be rerolled')
            if any(r['task']!=task.name or len(r['action'])!=7 or len(r['actual_numeric67_input'])!=67 for r in trace):
                raise ValueError('Original native model control trace is incomplete')
            row={'seed':seed,'task':task.name,'success':bool(trace[-1]['success']),'steps':len(trace),
                'safety_violation':trace[-1].get('safety_violation'),'trajectory':str((output/f'test_seed{seed}'/'trajectory.json').resolve()),
                'preserved_original_neural_episode':True,'original_trajectory_sha256':sha256_file(path)}
            if (seed,task.name) in stored:
                for key in ('success','steps','safety_violation'):
                    if stored[seed,task.name][key]!=row[key]:raise ValueError('Progress contradicts terminal trajectory')
            rows.append(row);preserved[seed,task.name]=row
        shutil.copytree(folder,output/folder.name)
        sources.append({'path':str(path.resolve()),'sha256':sha256_file(path)})
    if set(stored)-set(preserved):raise ValueError('A previously recorded episode went missing')
    atomic_json(output/'preserved_episode_evidence.json',{'checkpoint':str(checkpoint),'previous':str(previous.resolve()),
        'previous_progress_sha256':sha256_file(previous/'progress.json'),'previous_child_sha256':sha256_file(child_path),
        'episodes':len(rows),'sources':sources,'old_controls_rerolled':False,'scorer_or_safety_changed':False,
        'episode_noise': 'unchanged adapter private per-seed/per-prompt generator; tasks independent'})
    return rows,preserved
