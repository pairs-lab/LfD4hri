"""Explicitly link the external DRL asset tree; never overwrite a destination."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drl-root', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    source = args.drl_root.resolve() / 'asset'
    if not source.is_dir():
        parser.error('The selected checkout has no asset/ directory')
    destination = package / 'asset'
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source:
            print('asset/ already points to the selected checkout')
            return
        parser.error('Refusing to replace existing asset/; inspect it manually')
    destination.symlink_to(os.path.relpath(source, package), target_is_directory=True)
    print('Linked asset/; assets are external and excluded by .gitignore')


if __name__ == '__main__':
    main()
