"""Hash pinned public checkpoint LFS payloads before starting long fits."""
import argparse
import json
import os
from pathlib import Path

from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        rows = []
        for family in ('xvla', 'vla_adapter'):
            proof_path = args.root/(family+'_download')/'source.json'
            proof = json.loads(proof_path.read_text())
            for entry in proof['files']:
                path = Path(proof['checkpoint_directory'])/entry['name']
                if path.stat().st_size != entry['size']:
                    raise ValueError('Pinned file size differs: '+str(path))
                actual = sha256_file(path)
                if entry['lfs_sha256'] and actual != entry['lfs_sha256']:
                    raise ValueError('Public LFS SHA256 differs: '+str(path))
                rows.append({'path': str(path), 'sha256': actual,
                    'expected_LFS_sha256': entry['lfs_sha256'], 'family': family})
        atomic_json(args.output/'audit.json', {'passed': True, 'pid': os.getpid(),
            'all_public_LFS_payload_SHA256_verified': True, 'files': rows})
        atomic_json(args.output/'status.json', {'status': 'completed', 'pid': os.getpid()})
    except Exception as error:
        atomic_json(args.output/'status.json', {'status': 'failed_preserve_evidence', 'pid': os.getpid(),
            'error': f'{type(error).__name__}: {error}'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
