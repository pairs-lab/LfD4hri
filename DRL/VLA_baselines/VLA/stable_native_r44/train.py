"""Same genuine native optimizer, with bounded TRAIN observations and lineage."""
import argparse
import json
from pathlib import Path

from VLA.stable_native_r44.conditioning import install, add_contract


def main(args):
    install()
    import VLA.new_vlas.train as native
    original_save = native.save_checkpoint
    def save(*values, **kwargs):
        checkpoint = original_save(*values, **kwargs)
        add_contract(checkpoint)
        return checkpoint
    native.save_checkpoint = save
    native.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('xvla', 'vla_adapter'), required=True)
    parser.add_argument('--mode', choices=('gate', 'long'), required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, required=True)
    parser.add_argument('--total-updates', type=int, default=7000)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reload-proof', type=Path)
    args = parser.parse_args()
    old_step = 0 if args.resume is None else json.loads((args.resume/'config.json').read_text())['actual_optimizer_updates']
    if args.mode == 'gate' and (args.updates-old_step != 2 or args.accumulation != 4):
        parser.error('Gate must run2 new actual Adam updates and8 micros')
    main(args)
