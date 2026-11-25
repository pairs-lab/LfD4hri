"""Same qualified 128-update R71 trainer, only the declared sampler changes."""
import argparse
from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.flower_rotation_r76.sampler import Data,RECIPE
import VLA.flower_r71.fit as native


def main(a):
    saved=(native.GraspData,native.RECIPE,native.freeze_test_code)
    def bindings():
        result=saved[2]()
        result.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
        return result
    native.GraspData,native.RECIPE,native.freeze_test_code=Data,RECIPE,bindings
    try:native.main(a)
    finally:native.GraspData,native.RECIPE,native.freeze_test_code=saved


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=128)
    main(p.parse_args())
