"""Retain the qualified S-only continuation under the recovered namespace."""
from .run import OUT,bindings
import VLA.dita_close800_r94.common as common


def main():
    common.OUT=OUT; common.bindings=bindings
    from VLA.dita_close800_r94.other_models import main as continue_other
    continue_other()


if __name__=='__main__': main()
