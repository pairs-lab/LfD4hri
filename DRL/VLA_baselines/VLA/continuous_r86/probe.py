"""Saved TRAIN prediction diagnostic, not another robot/validation trial."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.replacement_r84.data import Data
from .audit import training_audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    atomic_json(a.output,training_audit(a.checkpoint,Data()))
