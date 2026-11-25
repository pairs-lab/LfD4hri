"""Create a lightweight, non-destructive source snapshot from an existing DRL.

Only explicit source/config suffixes are copied. No training scripts are run.
Existing changed destination files are never overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES = {'.py', '.sh', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.xml', '.urdf'}
SKIP = {'third_party', 'audits', 'datasets', 'checkpoints', 'cache', 'logs',
        'qualitative', 'evaluations', 'frozen_models', 'maintenance', 'assets',
        '.venvs', '.venv', 'venv', '__pycache__', '.git', 'site-packages',
        'node_modules', 'recipe_sources', 'delivery', 'data_indices',
        'demos', 'sources', 'review', 'artifacts'}
MAX_FILE = 2 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_generated(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    path.write_text(text)


def copy_file(source, target, source_root, records):
    if source.is_symlink() or not source.is_file():
        return
    if source.stat().st_size > MAX_FILE:
        raise ValueError(f'Unexpected large source file: {source.relative_to(source_root)}')
    if target.exists() and digest(target) != digest(source):
        raise ValueError(f'Refusing to overwrite changed exported file: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    records.append({'path': str(target.relative_to(ROOT)),
                    'source': str(source.relative_to(source_root)),
                    'bytes': target.stat().st_size, 'sha256': digest(target)})


def source_files(folder):
    for base, directories, files in os.walk(folder, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in SKIP
                                and not d.startswith('reproducibility_')
                                and not Path(base, d).is_symlink())
        for name in sorted(files):
            path = Path(base, name)
            if path.suffix in SOURCE_SUFFIXES and not name.startswith('.'):
                yield path


def git(folder, *args):
    run = subprocess.run(['git', '-C', str(folder), *args],
                         env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'},
                         capture_output=True, timeout=90, check=True)
    return run.stdout


def export_upstreams(source_root, records):
    rows = []
    parent = source_root / 'VLA' / 'third_party'
    for repo in sorted(parent.iterdir()):
        if not (repo / '.git').exists() or repo.is_symlink():
            continue
        url = git(repo, 'remote', 'get-url', 'origin').decode().strip()
        if not url.startswith('https://github.com/') or '@' in url:
            raise ValueError(f'Review non-public upstream URL for {repo.name}')
        row = {'name': repo.name, 'url': url,
               'commit': git(repo, 'rev-parse', 'HEAD').decode().strip()}
        paths = ['*.py', '*.sh', '*.yaml', '*.yml', '*.toml', '*.ini', '*.cfg',
                 '*.json', '*requirements*.txt', '*LICENSE*', '*COPYING*']
        patch = git(repo, 'diff', '--no-ext-diff', '--no-color', '--no-renames',
                    '--no-textconv', 'HEAD', '--', *paths)
        if b'Binary files ' in patch or len(patch) > 8 * 1024 * 1024:
            raise ValueError(f'Large/binary upstream patch needs manual review: {repo.name}')
        if patch:
            path = ROOT / 'upstream_patches' / (repo.name + '.patch')
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != patch:
                raise ValueError('Refusing to replace an existing upstream patch')
            path.write_bytes(patch)
            row['patch'] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
        extras = []
        untracked = git(repo, 'ls-files', '--others', '--exclude-standard', '-z',
                        '--', *paths).decode().split('\0')
        for relative in filter(None, untracked):
            path = repo / relative
            if any(part in SKIP or part.startswith('.') for part in Path(relative).parts):
                continue
            if path.suffix not in SOURCE_SUFFIXES or path.is_symlink():
                continue
            target = ROOT / 'upstream_patches' / repo.name / 'untracked' / relative
            copy_file(path, target, source_root, records)
            extras.append({'upstream_path': relative, 'package_path': str(target.relative_to(ROOT)),
                           'sha256': digest(target)})
        row['untracked_source_files'] = extras
        rows.append(row)
    write_generated(ROOT / 'UPSTREAMS.json', {
        'scope': 'Local source revisions, not a claim of paper-recipe reproduction; no weights included',
        'restore': 'python tools/restore_upstreams.py --fetch REPO', 'repositories': rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drl-root', type=Path, required=True)
    args = parser.parse_args()
    source_root = args.drl_root.resolve()
    if source_root == ROOT or not (source_root / 'VLA/common/environment.py').is_file():
        parser.error('Select the original DRL checkout containing VLA/common/environment.py')
    records = []
    for source in source_files(source_root / 'VLA'):
        copy_file(source, ROOT / source.relative_to(source_root), source_root, records)
    for folder in ('utils', 'env_eval'):
        for source in source_files(source_root / folder):
            copy_file(source, ROOT / source.relative_to(source_root), source_root, records)
    copy_file(source_root / 'config.py', ROOT / 'config.py', source_root, records)
    for source in (source_root / 'VLA/profiles').glob('*.json'):
        copy_file(source, ROOT / source.relative_to(source_root), source_root, records)
    # Lightweight derived gripper geometry remains source, not model weights.
    assets = source_root / 'VLA/assets/body_grasp_v9'
    for source in assets.rglob('*'):
        if source.suffix.lower() in {'.urdf', '.stl', '.mtl', '.jpg', '.png', '.md'}:
            copy_file(source, ROOT / source.relative_to(source_root), source_root, records)
        elif source.suffix == '.obj':
            # The base OBJ exceeds the generic source cap; include only files
            # under 4 MiB in this one explicitly scoped geometry directory.
            if source.is_file() and not source.is_symlink():
                target = ROOT / source.relative_to(source_root)
                if source.stat().st_size > 4 * 1024 * 1024:
                    raise ValueError('Unexpected gripper mesh size')
                if target.exists() and digest(target) != digest(source):
                    raise ValueError('Exported mesh differs')
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                records.append({'path':str(target.relative_to(ROOT)), 'source':str(source.relative_to(source_root)),
                                'bytes':target.stat().st_size,'sha256':digest(target)})
    protocol = source_root / 'VLA/reproducibility_locked21_20261124'
    for name in ('state21_reference.py', 'test_state21_reference.py', 'vla21-lock-v1.json',
                 'camera_profile.json', 'paragraph.tex', 'adaptation_table.tex', 'optimizer_table.tex',
                 'references.bib'):
        copy_file(protocol / name, ROOT / 'protocol21' / name, source_root, records)
    export_upstreams(source_root, records)
    write_generated(ROOT / 'SOURCE_MANIFEST.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_scope': 'DRL/VLA plus source-only DRL support modules; originals unchanged',
        'excluded': sorted(SKIP), 'files': sorted(records, key=lambda x: x['path']),
        'copied_files': len(records), 'copied_bytes': sum(r['bytes'] for r in records),
        'native_21D_migration_completed': False, 'model_jobs_started': False})
    print(f'Exported {len(records)} source/config/geometry files ({sum(r["bytes"] for r in records):,} bytes)')


if __name__ == '__main__':
    main()
