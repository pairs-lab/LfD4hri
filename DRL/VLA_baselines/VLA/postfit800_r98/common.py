from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import time

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from VLA.continuous_r86.owner import identity, alive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VLA/audits/vlanext_postfit800_r98_20261024'
SOURCE=ROOT/'VLA/audits/dita_terminal800_r95_20261023/other_models'
CP=SOURCE/'vlanext_s/recovery_fit9216'
CPU=Path('/home/uet/miniconda3/envs/robot/bin/python')
PY=ROOT/'VLA/.venvs/vlanext_r84/bin/python'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PORT=5999
SMOKE=[20298001]
FINAL=list(range(20294001,20294201))
TASKS=('touch','pick','move','pick_place')
ENV={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
    'MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
    'TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled',
    'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.now(timezone.utc).isoformat()


def verify(files):
    for path,digest in files.items():
        if sha256_file(path)!=digest:raise ValueError('Bound file changed: '+str(path))


def bindings():
    from VLA.autorefinement_r97.common import bindings as existing
    result=existing()
    for folder in ('postfit800_r98','replacement_r84'):
        result.update({str(p.resolve()):sha256_file(p) for p in (ROOT/'VLA'/folder).glob('*.py')})
    return result


def approved(method,cohort):
    return bool(cohort['role']=='fresh_secondary_final800' and cohort['seeds']==FINAL
        and method.get('family')=='vlanext_s' and Path(method.get('checkpoint','')).resolve()==CP
        and method.get('R98_frozen_before_any_evaluation') is True
        and method.get('actual_checkpoint_optimizer_updates')==9216
        and method.get('frozen_for_user_requested_benchmark') is True
        and method.get('benchmark_authorization',{}).get('user_requested_direct_postfit800') is True)


@contextmanager
def native_s_callbacks():
    """Family/reference/eligibility callbacks ONLY, in this separate process.

    R97 resumed Dita and therefore imported a NumPy1 reference. S actually
    uses NumPy2 in BOTH native and evaluator environments. Substitute its
    real reference, never the native model image, input, actions or physics.
    Historical source files and the running trainer remain untouched.
    """
    import VLA.autorefinement_r97.evaluate as evaluator
    import VLA.autorefinement_r97.audit as auditor
    import VLA.receipt_recovery_r88.evaluate as bookkeeping
    import VLA.receipt_recovery_r88.map_reference as reference
    import VLA.recovery_r85.evaluate as writer
    from VLA.privileged_visual_r53.scene_map import render
    saved=(evaluator.approved_final,auditor.approved_final,bookkeeping.render,reference.render,writer.atomic_json)
    evaluator.approved_final=auditor.approved_final=approved
    bookkeeping.render=reference.render=render
    writer.atomic_json=durable_json
    try:yield
    finally:
        evaluator.approved_final,auditor.approved_final,bookkeeping.render,reference.render,writer.atomic_json=saved


def durable_json(path,value):
    """Fsync committed traces/progress and their native query journal."""
    from VLA.recovery_r85.evaluate import json_metadata
    path=Path(path)
    # The query evidence must be durable BEFORE its outcome is committed.
    if path.name=='progress.json':
        journal=path.parent/'NEURAL_QUERIES.jsonl'
        if journal.exists():
            with journal.open('rb') as stream:os.fsync(stream.fileno())
    atomic_json(path,json_metadata(value))
    with path.open('rb') as stream:os.fsync(stream.fileno())
    directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory)
    finally:os.close(directory)


def tail(path,limit=24000):
    path=Path(path)
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-limit))
        return f.read().decode(errors='replace')


def retryable(text,abrupt=False,stalled=False):
    fatal=('ValueError:','TypeError:','KeyError:','AssertionError:','AttributeError:',
        'receipt mismatch','prefix differs','scene differs','Bound file changed','Malformed',
        'Nonfinite','Native map pixel','Source mismatch','hash mismatch')
    if any(value in text for value in fatal):return False
    transient=('ConnectionError','ReadTimeout','Connection reset','Connection refused',
        'RemoteDisconnected','out of memory','Killed','BrokenPipeError','Resource interruption')
    return abrupt or stalled or any(value in text for value in transient)


class Heartbeat:
    def __init__(self,now):self.changed=now;self.signature=None
    def observe(self,signature,now):
        if signature!=self.signature:self.signature=signature;self.changed=now
        return now-self.changed


def progress_signature(folder):
    values=[]
    for name in ('progress.json','NEURAL_QUERIES.jsonl','RESTORE_PROGRESS.json'):
        path=Path(folder)/name
        values.append((path.stat().st_size,path.stat().st_mtime_ns) if path.exists() else None)
    return tuple(values)
