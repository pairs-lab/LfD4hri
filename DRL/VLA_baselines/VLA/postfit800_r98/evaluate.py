"""Version-appropriate S renderer with the already exercised R97 resume loop."""
import argparse
import json
from pathlib import Path

from .common import read, native_s_callbacks


def check_pending_journal(progress_path,cohort):
    from VLA.autorefinement_r97.evaluate import ledger
    _,rows,done=ledger(progress_path,cohort)
    pending=[s for s in cohort['seeds'] if any((s,t) not in done
        for t in ('touch','pick','move','pick_place'))]
    path=Path(progress_path).parent/'NEURAL_QUERIES.jsonl'
    keys=[]
    if path.exists() and pending:
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'):continue
                q=json.loads(line)
                if q['seed']==pending[0] and (q['seed'],q['task']) not in done:
                    keys.append((q['seed'],q['task'],q['index']))
    if len(set(keys))!=len(keys) or len({(s,t) for s,t,i in keys})>1:
        raise ValueError('Duplicate or multiple incomplete task journal prefixes')
    if sorted(i for s,t,i in keys)!=list(range(len(keys))):
        raise ValueError('Noncontiguous incomplete native query prefix')


def main(a):
    from VLA.autorefinement_r97.evaluate import main as same_physical_loop
    if a.resume_progress:check_pending_journal(a.resume_progress,read(a.cohort))
    with native_s_callbacks():same_physical_loop(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('method','cohort','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--resume-progress',type=Path)
    main(p.parse_args())
