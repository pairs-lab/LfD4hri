"""Fresh-process native R54 entry points with explicit, narrow repairs.

No policy math, training labels, observation cameras or controller is changed.
All old code stays byte-identical and all new saved contracts bind this file.
"""
import argparse
import importlib
from pathlib import Path
import sys

from VLA.common.provenance import sha256_file
from VLA.family_transfer_r54.common import SCHEMA


def register_smol():
    # The base registry is empty in a clean trainer; importing the concrete
    # configuration is required BEFORE PreTrainedConfig.from_pretrained.
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    return SmolVLAConfig


def bind_contracts(module):
    original = module.atomic_json
    def save(path, value):
        if isinstance(value, dict) and value.get('native_VLA_actions_only'):
            if value.get('state_schema', value.get('schema')) != SCHEMA:
                raise ValueError('Never reinterpret another observation schema')
            value = {**value, 'schema': SCHEMA,
                'r56_integration_repairs_only': True,
                'code': {**value['code'], str(Path(__file__).resolve()): sha256_file(__file__)}}
        original(path, value)
    module.atomic_json = save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('smol-config', 'fit_smol', 'reload_smol',
        'fit_pi0', 'reload_pi0', 'server'))
    args, rest = parser.parse_known_args()
    if args.operation == 'smol-config':
        register_smol()
        from VLA.family_transfer_r54.smol import config
        cfg = config(device='cpu')
        assert cfg.type == 'smolvla' and cfg.max_state_dim == 67 and cfg.chunk_size == 5
        assert len(cfg.image_features) == 3
        print('PASS fresh concrete registration + native67 + horizon5 + three views', flush=True)
        return
    if 'smol' in args.operation or (args.operation == 'server' and 'smolvla' in rest):
        register_smol()
    module = importlib.import_module('VLA.family_transfer_r54.' + args.operation)
    if args.operation.startswith('fit_'):
        bind_contracts(module)
    # Reuse the original CLI parser, but retain the repaired module object.
    # runpy would create a separate namespace and lose atomic_json binding.
    from argparse import Namespace
    def path_options(names, required=()):
        p = argparse.ArgumentParser()
        for name in names:
            p.add_argument('--' + name, type=Path, required=name in required)
        return p
    if args.operation == 'fit_smol':
        p = path_options(('data', 'output', 'resume', 'reload-proof'), ('output',))
        p.add_argument('--mode', choices=('gate', 'long'), required=True)
        p.add_argument('--updates', type=int, required=True)
    elif args.operation == 'fit_pi0':
        p = path_options(('data', 'input-proof', 'output', 'qualification'), ('input-proof', 'output'))
        p.add_argument('--updates', type=int, choices=(2, 1500, 3000), required=True)
    elif args.operation == 'server':
        p = path_options(('checkpoint', 'output'), ('checkpoint', 'output'))
        p.add_argument('--model', choices=('smolvla', 'pi0'), required=True)
        p.add_argument('--port', type=int, required=True)
        p.add_argument('--baseline', action='store_true')
    else:
        p = path_options(('checkpoint', 'output'), ('checkpoint', 'output'))
    parsed = p.parse_args(rest)
    if hasattr(parsed, 'data') and parsed.data is None:
        from VLA.family_transfer_r54.common import DATA
        parsed.data = DATA
    module.main(parsed)


if __name__ == '__main__':
    main()
