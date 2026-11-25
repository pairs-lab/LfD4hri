"""Expand the tested intervention to every original TRAIN target, once each."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from . import collect
from .audit import BANK, OUT, TASKS, read


def main(a):
    rows = read(BANK/'index.json')
    vocabulary = {task: {r['pick_name'] for r in rows
        if r['task']==task and r['source_kind']=='complete_original'} for task in TASKS}
    if any(v != vocabulary[TASKS[0]] for v in vocabulary.values()):
        raise ValueError('Use common original TRAIN vocabulary, never selected failed test targets')
    collect.TARGETS = tuple(sorted(vocabulary[TASKS[0]]))
    collect.main(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT/'near_miss_sweep')
    main(p.parse_args())
