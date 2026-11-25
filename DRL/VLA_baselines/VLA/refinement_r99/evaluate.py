"""R98 physical loop and durable resume, explicit R99 final authorization."""
import argparse
from pathlib import Path
from .common import callbacks,read


def main(a):
    from VLA.postfit800_r98.evaluate import check_pending_journal
    from VLA.autorefinement_r97.evaluate import main as actual_loop
    if a.resume_progress:check_pending_journal(a.resume_progress,read(a.cohort))
    with callbacks():actual_loop(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('method','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--resume-progress',type=Path)
    main(p.parse_args())
