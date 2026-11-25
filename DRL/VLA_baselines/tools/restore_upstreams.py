"""Restore explicitly selected, pinned upstream source and local text changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def checked_path(base, relative):
    path = base / relative
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Unsafe relative path in upstream manifest')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--fetch', nargs='+', metavar='REPO')
    args = parser.parse_args()
    entries = {r['name']: r for r in json.loads((ROOT / 'UPSTREAMS.json').read_text())['repositories']}
    if not args.fetch:
        for name, row in entries.items():
            print(name, row['commit'], row['url'])
        return
    if any(name not in entries for name in args.fetch):
        parser.error('Unknown repo name; use --list')
    env = {**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1', 'GIT_TERMINAL_PROMPT': '0'}
    for name in args.fetch:
        row = entries[name]
        destination = ROOT / 'VLA' / 'third_party' / name
        if destination.exists():
            parser.error(f'Refusing to change existing {destination}; use a fresh checkout')
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                        '--no-recurse-submodules', row['url'], str(destination)], env=env, check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', row['commit']], env=env, check=True)
        patch = row.get('patch')
        if patch:
            path = checked_path(ROOT, patch['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != patch['sha256']:
                raise ValueError('Upstream patch checksum mismatch')
            subprocess.run(['git', '-C', str(destination), 'apply', '--check', str(path)], check=True)
            subprocess.run(['git', '-C', str(destination), 'apply', str(path)], check=True)
        for extra in row.get('untracked_source_files', []):
            source = checked_path(ROOT, extra['package_path'])
            target = checked_path(destination, extra['upstream_path'])
            if target.exists() or target.is_symlink():
                raise ValueError('Untracked source would overwrite an upstream file')
            if hashlib.sha256(source.read_bytes()).hexdigest() != extra['sha256']:
                raise ValueError('Untracked source checksum mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        print(f'{name}: source restored; install its dependencies and obtain weights separately')


if __name__ == '__main__':
    main()
