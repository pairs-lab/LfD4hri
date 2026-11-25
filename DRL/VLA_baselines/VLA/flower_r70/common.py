import hashlib
import json
import os
from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.common.artifact_io import atomic_json

ROOT=Path(__file__).resolve().parents[2]
R69=ROOT/'VLA/audits/flower_privileged_r69_20261019'
OUT=ROOT/'VLA/audits/flower_postfit_r70_20261019'
PY=ROOT/'VLA/.venvs/flower_r69/bin/python'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
DEVELOPMENT=ROOT/'VLA/audits/recovery_r56_20261017/automatic_v1/development40.json'
CONFIRMATION=DEVELOPMENT.with_name('confirmation40.json')
ENV={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
     'TOKENIZERS_PARALLELISM':'false','WANDB_MODE':'disabled'}


def read(path):return json.loads(Path(path).read_text())


def verify(binding):
    for path,digest in binding.items():
        if sha256_file(path)!=digest:raise ValueError('Changed bound source: '+path)


def identity(path):
    s=Path(path).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def snapshot_rolling(source,dest):
    """Pin an atomically-replaced rolling master inode, never copy a live write.

    R69 writes masters then optimizer, serially inside one snapshot(). Bookend
    identities and write order reject capture during an in-progress pair save.
    The writer never modifies either published inode in place.
    """
    import torch
    source,dest=Path(source),Path(dest)
    if dest.exists():raise FileExistsError('Preserve previous diagnostic checkpoint')
    master,opt=source/'trainables.pt',source/'optimizer.pt'
    before={str(p):identity(p) for p in (master,opt)}
    if before[str(opt)][3]<before[str(master)][3]:
        raise ValueError('Snapshot pair still being written; wait for optimizer publication')
    state=torch.load(opt,weights_only=True,mmap=True,map_location='cpu')
    step=int(state['actual_new_optimizer_updates'])
    if step<=128 or step%512:raise ValueError('Expected a completed rolling512 snapshot')
    del state
    if before!={str(p):identity(p) for p in (master,opt)}:
        raise ValueError('Rolling pair changed during inspection; retry later')
    plan=read(source/'PLAN.json');verify(plan['code'])
    dest.mkdir(parents=True)
    os.link(master,dest/'trainables.pt')
    if before!={str(p):identity(p) for p in (master,opt)}:
        atomic_json(dest/'CAPTURE_REJECTED.json',{'error':'Source snapshot pair changed after hardlink'})
        raise ValueError('Capture rejected; do not use this pinned file')
    atomic_json(dest/'stats.json',read(source/'stats.json'))
    result={'kind':'rolling_development_NOT_final','checkpoint':str(dest.resolve()),
        'actual_new_optimizer_updates':step,'source':str(source.resolve()),'source_file_identity':before,
        'trainables_sha256':sha256_file(dest/'trainables.pt'),'code':plan['code'],
        'stats_sha256':sha256_file(dest/'stats.json'),'no_training_process_stopped':True,
        'pinned_master_inode_not_modified_by_future_atomic_saves':True}
    atomic_json(dest/'CHECKPOINT.json',result)
    return result


def checkpoint(path, *, allow_rolling=False):
    path=Path(path).resolve()
    if (path/'CAPTURE_REJECTED.json').exists():raise ValueError('Rejected capture cannot be used')
    if (path/'CHECKPOINT.json').exists():
        if not allow_rolling:raise ValueError('Rolling development checkpoint cannot replace final fit')
        result=read(path/'CHECKPOINT.json')
    else:
        result=read(path/'TRAINING.json')
        if not result.get('completed'):raise ValueError('Completed fit required')
    verify(result['code'])
    if sha256_file(path/'trainables.pt')!=result['trainables_sha256']:
        raise ValueError('Master file changed')
    return result


def freeze_test_code():
    paths=list(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'VLA/scene64_conditioning.py',ROOT/'VLA/simstate_scene64.py',
              ROOT/'VLA/simstate_feedback.py',ROOT/'VLA/privileged_visual_r53/scene_map.py']
    binding=read(ROOT/'VLA/audits/cross_vla800_r62_20261018/CODE_AND_PLANT_BINDING.json')
    verify(binding)
    binding.update({str(p):sha256_file(p) for p in paths})
    return binding
