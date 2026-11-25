"""Use the existing full native optimizer with an explicitly admitted bank.

Process-local injection, not source edits or a small replacement action model.
The current R86 process never imports this module.
"""
import argparse
from pathlib import Path

from VLA.common.provenance import sha256_file
from .audit import OUT, read
from .data import Data


def bindings():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [OUT/'qualified_recovery/ADMISSION.json', OUT/'qualified_recovery/index.json']
    return {str(p.resolve()): sha256_file(p) for p in files}


def main(a):
    import VLA.replacement_r84.data as shared
    shared.Data = Data
    if a.family == 'dita':
        import VLA.dita_transfer_r84.train as native
    else:
        import VLA.replacement_r84.train as native
        native.Data = Data
    original_bindings = native.bindings
    native.bindings = lambda: {**original_bindings(), **bindings()}
    original_atomic = native.atomic_json
    def record(path, value):
        data = getattr(Data, 'last_instance', None)
        if Path(path).name in ('TRAINING.json', 'status.json') and data is not None:
            value = {**value, 'R90_actual_sampled_source_counts': dict(data.exposure)}
        if Path(path).name == 'TRAINING.json':
            if data is None or data.exposure['near_miss_corrective'] <= 0 or data.exposure['complete_original'] <= 0:
                raise ValueError('No actual exposure to both new recoveries and original TRAIN examples')
            value = {**value, 'R90_data_admission': str(OUT/'qualified_recovery/ADMISSION.json'),
                'R90_data_admission_sha256': sha256_file(OUT/'qualified_recovery/ADMISSION.json'),
                'original_normalizers_preserved': True,
                'new_data_sampling': 'original task/target draw,35% within-target replacement if available;60% first8',
                'future_teacher_phase_in_policy_inputs': False,
                'native_architecture_and_loss_unchanged': True}
        return original_atomic(path, value)
    native.atomic_json = record
    native.fit(a) if a.op == 'fit' else native.reload(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('dita', 'vlanext_s'), required=True)
    sub = p.add_subparsers(dest='op', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--updates', type=int, required=True)
    f.add_argument('--accumulation', type=int, default=32)
    f.add_argument('--resume', type=Path, required=True)
    f.add_argument('--device', default='cuda', choices=('cpu','cuda'))
    f.add_argument('--optimizer', default='adamw8', choices=('adamw8','adamw32'))
    r = sub.add_parser('reload'); r.add_argument('--checkpoint', type=Path, required=True)
    r.add_argument('--device', default='cuda', choices=('cpu','cuda'))
    main(p.parse_args())
