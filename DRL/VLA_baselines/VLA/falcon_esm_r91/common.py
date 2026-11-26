from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VLA/audits/storage_baseline_r91_20261022/falcon_esm'
REPO = ROOT/'VLA/third_party/falcon_vla'
ASSETS = ROOT/'VLA/checkpoints/falcon_esm_r91'
PY = ROOT/'VLA/.venvs/falcon_esm_r91/bin/python'
REV = '902eec199093478b500edd2cde2d8728c556d5de'
WEIGHT = ASSETS/'falcon_esm_fc_calvin_abc_wdepth.pt'
WEIGHT_SHA = '9f026a7bd00a154e7db061398753737003f97e89aa2c111b87070e9afd3d4160'
WEIGHT_SIZE = 15997622110
CFG = REPO/'configs/calvin_finetune/finetune_kosmos_cont-fc-post_esm_wd-0_use-hand_ws-1_act-10.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def bindings():
    files = [*Path(__file__).parent.glob('*.py'), CFG]
    files += list((REPO/'falcon/model').rglob('*.py'))
    files += [REPO/'tools/modeling_kosmos2.py']
    return {str(p.resolve()): sha(p) for p in files}
