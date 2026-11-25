import json
import os
from pathlib import Path
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/audits/cross_vla800_r62_20261018'
R59=ROOT/'VLA/audits/xvla800_r59_20261017'
R56=ROOT/'VLA/audits/recovery_r56_20261017'
MAIN=R56/'automatic_v1'
ADAPTER=R56/'adapter_stage_restart_v1'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PI=ROOT/'VLA/third_party/openpi/.venv/bin/python'
TASKS=('touch','pick','move','pick_place')
SEEDS=list(range(20271001,20271201))
MODELS=('smolvla','pi0','vla_adapter')
ENV={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'.82'}


def read(p):return json.loads(Path(p).read_text())


def utc():return datetime.now(timezone.utc).isoformat()


def alive(pid):
    if not pid:return False
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def blockers():
    result=[]
    for folder in (MAIN,ADAPTER):
        file=folder/'status.json'
        if not file.exists():
            result.append({'source':str(folder),'reason':'source owner status missing'});continue
        s=read(file)
        active={k:s.get(k) for k in ('pid','active_child_pid','active_server_pid') if alive(s.get(k))}
        if active:result.append({'source':str(folder),'stage':s.get('stage'),'active':active})
    return result


def resources():
    mem={r.split(':')[0]:int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()}
    v=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=15).strip().splitlines()
    if len(v)!=1:raise ValueError('This admission policy was qualified for the one-GPU machine')
    return {'RAM_available_MiB':mem['MemAvailable']/1024,'GPU_free_MiB':int(v[0]),
            'disk_free_GiB':__import__('shutil').disk_usage(ROOT).free/1024**3}


def admissible(r):
    return r['RAM_available_MiB']>=4500 and r['GPU_free_MiB']>=14000 and r['disk_free_GiB']>=30


def method_for(model):
    # Only the already predeclared development/confirmation selector decides
    # weights. No old120/new800 outcomes are read here.
    if model=='smolvla':
        file=MAIN/'smolvla_selection.json';method=read(file)['selected']
    elif model=='pi0':
        file=MAIN/'status.json';method=read(file)['selected_methods']['pi0']
    elif model=='vla_adapter':
        file=ADAPTER/'FROZEN_BEFORE_FINAL.json'
        method=read(file)['selected']
    else:raise ValueError('Unknown selected family')
    from VLA.autocycle_r55.decision import candidate_key
    score=candidate_key(method)  # Reject partial/final/unknown selection cohorts.
    cp=Path(method['checkpoint']).resolve()
    if not cp.is_dir():raise ValueError('Selected checkpoint missing')
    if model in ('smolvla','pi0'):
        if not 1<=method['prefix']<=5:raise ValueError('Invalid native5 execution prefix')
        if method['mode'] not in (('baseline','map3') if model=='smolvla' else ('map3',)):
            raise ValueError('Wrong native input contract')
    else:
        schedule=read(method['schedule'])
        if schedule['family']!=model or Path(schedule['checkpoint']).resolve()!=cp:
            raise ValueError('Schedule/checkpoint mismatch')
        if any(not 4<=schedule['tasks'][t]['selected_prefix']<=8 for t in TASKS):
            raise ValueError('Invalid Adapter native execution prefix')
    return {**method,'checkpoint':str(cp)},file,score


def checkpoint_hashes(cp):
    from VLA.common.provenance import sha256_file
    cp=Path(cp).absolute();paths=[];seen=set()
    # Orbax/HF checkpoints can contain symlinks. Bind the logical inference
    # paths too, so changing a symlink cannot evade a later weight check.
    for directory,dirs,names in os.walk(cp,followlinks=True):
        actual=Path(directory).resolve()
        if actual in seen:raise ValueError('Aliased/cyclic checkpoint directory requires explicit qualification')
        seen.add(actual);dirs.sort()
        paths.extend(Path(directory)/name for name in sorted(names) if (Path(directory)/name).is_file())
    if not paths:raise ValueError('Empty checkpoint')
    return {str(p):sha256_file(p) for p in sorted(paths)}


def verify(bound):
    from VLA.common.provenance import sha256_file
    for path,h in bound.items():
        if sha256_file(path)!=h:raise ValueError('Frozen dependency or checkpoint changed: '+path)


def matrix(rows):
    import numpy as np
    expected={(s,t) for s in SEEDS for t in TASKS}
    keys=[(r['seed'],r['task']) for r in rows]
    if len(rows)!=800 or len(set(keys))!=800 or set(keys)!=expected:
        raise ValueError('Require all800 unique locked trials; no partial forecast or seed replacement')
    if any(type(r['success']) is not bool for r in rows):raise ValueError('Require actual boolean outcomes')
    lookup={(r['seed'],r['task']):r['success'] for r in rows}
    return np.asarray([[lookup[s,t] for t in TASKS] for s in SEEDS],float)
