"""Download a revision-pinned CogACT checkpoint without allocating model memory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, snapshot_download


REPO = "CogACT/CogACT-Small"
REVISION = "ca302a300542393a0b06e291465076a7d9105c23"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("VLA/checkpoints/cogact-small"))
    parser.add_argument("--status", type=Path, default=Path("VLA/logs/cogact_checkpoint_download_status.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    starting = {"status": "checking_remote_metadata", "repo": REPO, "revision": REVISION,
                "pid": os.getpid(), "model_loaded": False, "gpu_used_by_downloader": False}
    args.status.write_text(json.dumps(starting, indent=2) + "\n")
    try:
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, timeout=20)
    except Exception as error:
        starting.update(status="failed_metadata_request", error=f"{type(error).__name__}: {error}")
        args.status.write_text(json.dumps(starting, indent=2) + "\n")
        raise
    names = {"config.json", "dataset_statistics.json", "checkpoints/CogACT-Small.pt"}
    files = {item.rfilename: item.size for item in info.siblings if item.rfilename in names}
    if set(files) != names:
        raise RuntimeError("Pinned CogACT snapshot file set differs from expected")
    remaining = sum(size for name, size in files.items()
                    if not (args.output / name).is_file() or (args.output / name).stat().st_size != size)
    if shutil.disk_usage(args.output).free < remaining + 1024**3:
        raise RuntimeError("Insufficient checkpoint disk space")
    status = {"repo": REPO, "revision": info.sha, "files": files,
              "output": str(args.output.resolve()),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "downloading", "pid": os.getpid(), "model_loaded": False, "gpu_used_by_downloader": False}
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)
    try:
        snapshot_download(REPO, revision=REVISION, allow_patterns=sorted(names),
                          local_dir=args.output, max_workers=1)
        for name, size in files.items():
            if (args.output / name).stat().st_size != size:
                raise RuntimeError(f"Downloaded checkpoint size mismatch: {name}")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        args.status.write_text(json.dumps(status, indent=2) + "\n")
        raise
    status.update(status="downloaded_sizes_verified", finished_at=datetime.now(timezone.utc).isoformat(),
                  caveat="No model load/inference proof or full local cryptographic fingerprint yet")
    args.status.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
