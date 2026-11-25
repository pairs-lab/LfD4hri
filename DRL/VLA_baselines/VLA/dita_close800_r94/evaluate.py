"""Exact native physical loop, authorized terminal checkpoint, optional safe resume."""
import argparse
from pathlib import Path

from VLA.dita800_r87.evaluate import authorized_frozen, compile_unchanged_physical_loop
from VLA.recovery_quality_r90.evaluate import renderer, diagnostic_class
from .common import CP, SEEDS, read


def validate(method, cohort):
    if not authorized_frozen(method, cohort):
        raise ValueError('Explicit user final800 authorization required; no invented CONF pass')
    if (method.get('family') != 'dita' or Path(method['checkpoint']).resolve() != CP
            or method.get('actual_checkpoint_optimizer_updates') != 10000 or cohort['seeds'] != SEEDS
            or not method.get('terminal_checkpoint_no_further_Dita_finetune')):
        raise ValueError('Only the exact frozen Dita10,000 and declared800 cohort are permitted')


def main(a):
    validate(read(a.method),read(a.cohort))
    from VLA.autonomy_r83 import evaluate as old
    original_render, factory = old.render, old.environment_class_for_profile
    old.render = renderer('dita')
    old.environment_class_for_profile = lambda profile: diagnostic_class(factory(profile),a.output)
    try:
        if a.resume_progress:
            from VLA.benchmark_guardian_r89.evaluate import main as resume
            resume(a)
        else:
            compile_unchanged_physical_loop(old)(a)
    finally:
        old.render, old.environment_class_for_profile = original_render, factory


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('method','cohort','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--resume-progress',type=Path)
    main(p.parse_args())
