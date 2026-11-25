"""Own fresh-process prediction parity with the bounded observation encoder."""
import argparse
from pathlib import Path

from VLA.stable_native_r44.conditioning import install, require_contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require_contract(args.checkpoint)
    install()
    from VLA.new_vlas.reload_gate import main
    main(args)
