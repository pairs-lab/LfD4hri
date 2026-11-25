"""Download revision-pinned public candidates without touching active models."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from VLA.common.artifact_io import atomic_json

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    'xvla': ('lerobot/xvla-base', 'cdb7964e4fe842935d671bfab5a5ebe00a96648c', 'xvla-base', 'xvla'),
    'vla_adapter': ('VLA-Adapter/LIBERO-Spatial-Pro', 'cb023c6771fc62ea729d218f09ac0583c14f59d4',
                    'vla-adapter-spatial-pro', 'vla_adapter'),
}


def fetch(args):
    from huggingface_hub import HfApi, snapshot_download
    repo, revision, directory, code = MODELS[args.model]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    target = ROOT / 'VLA/checkpoints' / directory
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError('Resolved weight revision differs from the pinned public checkpoint')
    git = subprocess.check_output(['git', '-C', str(ROOT/'VLA/third_party'/code),
                                   'rev-parse', 'HEAD'], text=True).strip()
    files = [{'name': f.rfilename, 'size': f.size,
              'lfs_sha256': f.lfs.sha256 if f.lfs else None} for f in info.siblings]
    atomic_json(out/'source.json', {'repo': repo, 'revision': revision, 'code_commit': git,
        'checkpoint_directory': str(target), 'files': files,
        'intended_variant': 'additional_privileged_domain_adaptation_NOT_published_benchmark_result',
        'created_at_utc': datetime.now(timezone.utc).isoformat()})
    atomic_json(out/'status.json', {'pid': os.getpid(), 'stage': 'downloading',
        'repo': repo, 'revision': revision, 'GPU_used': False, 'success_rate_established': False})
    try:
        snapshot_download(repo, revision=revision, local_dir=target, max_workers=2)
        for f in files:
            path = target/f['name']
            if not path.is_file() or path.stat().st_size != f['size']:
                raise ValueError('Incomplete download: ' + f['name'])
        atomic_json(out/'status.json', {'pid': os.getpid(), 'stage': 'downloaded_all_expected_file_sizes',
            'repo': repo, 'revision': revision, 'GPU_used': False,
            'bytes': sum(f['size'] for f in files), 'file_count': len(files),
            'checkpoint_directory': str(target), 'native_load_or_finetune_NOT_yet_qualified': True,
            'success_rate_established': False})
    except Exception as error:
        atomic_json(out/'status.json', {'pid': os.getpid(), 'stage': 'failed_preserve_download',
            'error': f'{type(error).__name__}: {error}', 'GPU_used': False,
            'success_rate_established': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if args.launch:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        marker = args.output.with_name(args.output.name+'_launch.json')
        if marker.exists() or args.output.exists():
            raise FileExistsError('No duplicate download owner')
        with args.output.with_name(args.output.name+'_launcher.log').open('x') as log:
            child = subprocess.Popen([str(ROOT/'VLA/third_party/lerobot/.venv/bin/python'),
                '-m', 'VLA.new_vlas.fetch', '--model', args.model,
                '--output', str(args.output.resolve())], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '0',
                     'HF_HUB_DISABLE_XET': '1', 'OMP_NUM_THREADS': '1'})
        atomic_json(marker, {'pid': child.pid, 'model': args.model, 'GPU_used': False})
        print(json.dumps({'pid': child.pid, 'model': args.model}))
    else:
        fetch(args)
