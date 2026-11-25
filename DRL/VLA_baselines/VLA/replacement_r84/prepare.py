"""Only the pinned small VLM foundation, never a whole model zoo download."""
import argparse
from pathlib import Path
from VLA.common.artifact_io import atomic_json
from VLA.common.provenance import sha256_file
from .common import BASE, OUT, REPO, REVISION, SOURCE_REVISION, utc

def main():
    from huggingface_hub import HfApi, snapshot_download
    import subprocess
    if subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_REVISION:
        raise ValueError('Wrong VLANeXt source revision')
    info = HfApi().model_info('Qwen/Qwen3.5-0.8B', revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError('Unpinned foundation revision')
    names = [x.rfilename for x in info.siblings if x.rfilename.endswith(('.json', '.jinja', '.txt', '.safetensors'))]
    weight_bytes = sum((x.size or 0) for x in info.siblings if x.rfilename in names)
    if not 1_000_000_000 < weight_bytes < 4_000_000_000:
        raise ValueError('Unexpected small VLM payload size')
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT/'VLANEXT_DOWNLOAD_STATUS.json', {'status':'downloading', 'UTC':utc(),
        'revision':REVISION, 'selected_bytes':weight_bytes, 'local_dir':str(BASE), 'files':names})
    snapshot_download('Qwen/Qwen3.5-0.8B', revision=REVISION, local_dir=BASE,
                      allow_patterns=names, max_workers=1)
    hashes = {str(BASE/n):sha256_file(BASE/n) for n in names}
    atomic_json(OUT/'VLANEXT_FOUNDATION.json', {'status':'downloaded_and_SHA_recorded',
        'UTC':utc(), 'repository':'Qwen/Qwen3.5-0.8B', 'revision':REVISION,
        'source_revision':SOURCE_REVISION, 'files':hashes, 'selected_bytes':weight_bytes,
        'robot_action_pretrained':False, 'native_optimizer_and_robot_SR_verified':False})

if __name__ == '__main__':
    main()
