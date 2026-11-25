"""Unchanged durable native physical loop, with a genuinely new holdout ID."""
import argparse
from contextlib import contextmanager
from pathlib import Path
from .common import approval


@contextmanager
def callbacks():
    import VLA.admission_r100.evaluate as original
    saved = original.approved
    original.approved = approval
    try:
        yield original
    finally:
        original.approved = saved


def audit_result(path, require_complete=True):
    with callbacks() as original:
        return original.audit_result(path, require_complete=require_complete)


def main(args):
    with callbacks() as original:
        original.main(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('method', 'cohort', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--resume-progress', type=Path)
    p.add_argument('--memory-fraction', type=float, default=.86)
    main(p.parse_args())
