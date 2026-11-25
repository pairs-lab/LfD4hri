"""Exact native replay of a completed, unchanged quantitative PUT trace."""
import argparse
from pathlib import Path
from VLA.qualitative_r57.replay import main
from VLA.common.provenance import sha256_file
from VLA.common.artifact_io import atomic_json
from types import SimpleNamespace

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--trajectory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    candidate={'seed':a.seed,'trajectory':str(a.trajectory.resolve()),
               'trajectory_sha256':sha256_file(a.trajectory),'cohort':str(a.trajectory.parents[1])}
    shortlist=a.output.parent/(a.output.name+'_candidate.json')
    atomic_json(shortlist,[candidate])
    main(SimpleNamespace(seed=a.seed,output=a.output,shortlist=shortlist,terminal_release=False))
