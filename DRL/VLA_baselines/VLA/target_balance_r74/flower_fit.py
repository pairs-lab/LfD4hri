"""Dependency-injected sampler ablation, exact unchanged R71 native trainer.

Changes only the read-only Dataset.draw and declared source binding/recipe.
No model, loss, target conversion, optimizer, camera or controller replacement.
"""
import argparse
from pathlib import Path
from VLA.common.provenance import sha256_file
from VLA.target_balance_r74.sampler import Data,RECIPE
import VLA.flower_r71.fit as native


def main(a):
    original=native.freeze_test_code
    def bindings():
        result=original()
        result.update({str(p.resolve()):sha256_file(p) for p in Path(__file__).parent.glob('*.py')})
        return result
    native.GraspData=Data;native.RECIPE=RECIPE;native.freeze_test_code=bindings
    native.main(a)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=128);main(parser.parse_args())
