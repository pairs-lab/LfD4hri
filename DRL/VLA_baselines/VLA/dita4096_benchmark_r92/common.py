from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.recovery_r85.common import ROOT,read,verify,utc
from VLA.continuous_r86.owner import identity,alive

OUT=ROOT/'VLA/audits/dita4096_800_r92_20261022'
R86=ROOT/'VLA/audits/continuous_r86_20261021'
CP=R86/'dita/fit4096'
SIM=ROOT/'VLA/third_party/lerobot/.venv/bin/python'
PORT=5992
SEEDS=list(range(20284001,20284201))


def bindings():
    files=list(Path(__file__).parent.glob('*.py'))
    for folder in ('dita800_r87','receipt_recovery_r88','benchmark_guardian_r89'):
        files+=list((ROOT/'VLA'/folder).glob('*.py'))
    files += [ROOT/'VLA/recovery_quality_r90/evaluate.py',ROOT/'VLA/recovery_quality_r90/transport.py']
    return {str(p.resolve()):sha256_file(p) for p in files}


def same(expected):
    actual=identity(expected['pid']) if expected else None
    # A surviving child can be reparented after coordinator exit. Start time
    # and complete argv, not PPID alone, guard against PID reuse.
    return bool(actual and all(actual.get(k)==expected.get(k) for k in ('pid','start_ticks','command')))
