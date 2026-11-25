from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.replacement_r84.common import ROOT,OUT as R84,read,utc,verify

OUT=ROOT/'VLA/audits/recovery_r85_20261021'

def bindings():
    return {str(p):sha256_file(p) for p in Path(__file__).parent.glob('*.py')}
