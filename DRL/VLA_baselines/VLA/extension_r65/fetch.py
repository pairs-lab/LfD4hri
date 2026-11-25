"""Pinned metadata/checkpoint downloads, no GPU/model allocation."""
from pathlib import Path
import hashlib
import json
import subprocess
from huggingface_hub import HfApi, snapshot_download
from VLA.common.artifact_io import atomic_json
from .common import OUT, BASE, QWEN, BASE_REVISION, NATIVE, ORIGINAL


def main():
    out = OUT/'preparation'; out.mkdir(parents=True, exist_ok=True)
    expected = {NATIVE: '146d8474932c8b566717eb2738ba459337ab71c4',
                ORIGINAL: 'ec907778b4af46a45742baf5ebfb2e055560346e'}
    for path, revision in expected.items():
        if subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() != revision:
            raise ValueError('Unexpected native source revision')
    api = HfApi(); info = api.model_info('Qwen/Qwen3-VL-2B-Instruct')
    # All learned Qwen values come from the complete VLA checkpoint. Only
    # architecture/tokenizer/processor metadata is needed from foundation.
    snapshot_download('Qwen/Qwen3-VL-2B-Instruct', revision=info.sha, local_dir=QWEN,
                      allow_patterns=['*.json', '*.txt', '*.jinja', 'tokenizer.model'], max_workers=1)
    snapshot_download('lerobot/VLA-JEPA-Pretrain', revision=BASE_REVISION, local_dir=BASE, max_workers=1)
    from VLA.common.provenance import sha256_file
    files = [p for folder in (BASE, QWEN) for p in folder.iterdir() if p.is_file()]
    atomic_json(out/'DOWNLOADS.json', {
        'passed': True, 'base_repository': 'lerobot/VLA-JEPA-Pretrain', 'base_revision': BASE_REVISION,
        'qwen_repository': 'Qwen/Qwen3-VL-2B-Instruct', 'qwen_metadata_revision': info.sha,
        'native_source_revisions': {str(p): v for p,v in expected.items()},
        'files_sha256': {str(p): sha256_file(p) for p in files},
        'actual_pretrained_robot_policy_not_random_backbone': True,
        'GPU_allocated': False, 'model_success_rate_established': False,
    })


if __name__ == '__main__':
    main()
