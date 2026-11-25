"""R83 native physical loop, authorized eligibility + correct audit pixels."""
import argparse
from pathlib import Path
from VLA.dita800_r87.evaluate import authorized_frozen,compile_unchanged_physical_loop
from VLA.recovery_quality_r90.evaluate import renderer,diagnostic_class
from .common import CP,SEEDS,read


def validate(method,cohort):
    if not authorized_frozen(method,cohort):
        raise ValueError('No explicit user800 authorization; do not invent CONF pass')
    if (method['family']!='dita' or Path(method['checkpoint']).resolve()!=CP or
            method.get('actual_checkpoint_optimizer_updates')!=4096 or cohort['seeds']!=SEEDS):
        raise ValueError('This benchmark is only for the frozen native Dita4096')


def main(a):
    from VLA.autonomy_r83 import evaluate as old
    validate(read(a.method),read(a.cohort))
    # Compatibility changes the evaluator's receipt reference only. The native
    # server still calls its original NumPy1.26 renderer on the actual payload.
    old.render=renderer('dita')
    factory=old.environment_class_for_profile
    old.environment_class_for_profile=lambda profile:diagnostic_class(factory(profile),a.output)
    try:compile_unchanged_physical_loop(old)(a)
    finally:old.environment_class_for_profile=factory


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('method','cohort','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,required=True);main(p.parse_args())
