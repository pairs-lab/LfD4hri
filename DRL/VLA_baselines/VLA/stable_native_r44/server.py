"""Original neural VLA server with versioned observation saturation only."""
import argparse
from pathlib import Path

from VLA.stable_native_r44.conditioning import bounded_append, require_contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    require_contract(args.checkpoint)
    import VLA.new_vlas.server as native
    native.append_numeric67 = bounded_append
    native.main(args)
