import json
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.recovery_r85.common import ROOT,utc
from VLA.common.provenance import sha256_file

OUT=ROOT/'VLA/audits/benchmark_guardian_r89_20261021'
TASKS=('touch','pick','move','pick_place')
EXPECTED={(s,t) for s in range(20284001,20284201) for t in TASKS}
ORDERED=[(s,t) for s in range(20284001,20284201) for t in TASKS]


def read(p):return json.loads(Path(p).read_text())


def complete_rows(path):
    data=read(path);rows=data['rows'];keys=[(r['seed'],r['task']) for r in rows]
    if (len(keys)!=len(set(keys)) or not set(keys)<=EXPECTED or data['completed_trials']!=len(rows)
        or data['planned_trials']!=800):raise ValueError('Committed ledger corruption/duplicate/coverage mismatch')
    if keys!=ORDERED[:len(rows)]:raise ValueError('Not the original fixed-cohort chronological prefix')
    if any(not isinstance(r['success'],bool) or type(r['steps']) is not int or r['steps']<1
           or not Path(r['trajectory']).exists() for r in rows):
        raise ValueError('Invalid actual completed outcome')
    from VLA.autonomy_r83.common import metrics
    if metrics(rows)!=data['per_action'] or data['motor_controls']<sum(r['steps'] for r in rows):
        raise ValueError('Committed metrics/control ledger mismatch')
    return data,rows,{(r['seed'],r['task']):r for r in rows}


def identity(pid):
    try:
        folder=Path('/proc')/str(int(pid));stat=(folder/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid),'start_ticks':stat[19],
            'command':[s.decode() for s in (folder/'cmdline').read_bytes().split(b'\0') if s]}
    except (FileNotFoundError,ProcessLookupError):return None


def alive(pid):
    if not pid:return False
    try:return (Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except (FileNotFoundError,ProcessLookupError):return False


def parent_pid(pid):
    try:return int((Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()[1])
    except (FileNotFoundError,ProcessLookupError):return None


def live_children(pid):
    """Read children of every thread; never assume an idle status file is enough."""
    result=set()
    for task in (Path('/proc')/str(int(pid))/'task').glob('*'):
        try:result.update(int(p) for p in (task/'children').read_text().split() if alive(int(p)))
        except (FileNotFoundError,ProcessLookupError):pass
    return result


def bindings():return {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
