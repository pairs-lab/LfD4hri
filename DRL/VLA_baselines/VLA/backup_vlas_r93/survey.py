"""Capture pinned public metadata without downloading weights or importing models."""

from datetime import datetime, timezone
import codecs
import hashlib
import json
from pathlib import Path
import re
import subprocess

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "VLA/audits/backup_vlas_r93_20261022"
SOURCES = {
    "gr00t_n15": {"repo": "NVIDIA/Isaac-GR00T", "path": ROOT / "VLA/third_party/gr00t_n15",
                  "expected_commit": "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96",
                  "intended_model": "nvidia/GR00T-N1.5-3B"},
    "seer": {"repo": "InternRobotics/Seer", "path": ROOT / "VLA/third_party/seer",
             "expected_commit": "ee228e110e098beefa23f94d558dfa8492259e6e",
             "intended_model": "official DROID-pretrained Seer, exact architecture pending checkpoint inventory"},
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def public_get(url, destination=None):
    response = requests.get(url, timeout=45)
    receipt = {"url": url, "status": response.status_code, "bytes": len(response.content),
               "sha256": hashlib.sha256(response.content).hexdigest()}
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        receipt["saved"] = str(destination)
    return response, receipt


def drive_listing(html):
    """Read public listing JSON; never evaluate the page's JavaScript."""
    match = re.search(r"window\['_DRIVE_ivd'\] = '((?:\\.|[^'])*)';", html)
    if match is None:
        raise ValueError("Public folder no longer exposes the expected listing")
    decoded = codecs.decode(match.group(1).replace(r"\/", "/"), "unicode_escape")
    value = json.loads(decoded)
    entries = []

    def visit(item):
        if not isinstance(item, list):
            return
        if len(item) > 3 and all(isinstance(item[i], str) for i in (0, 2, 3)):
            if item[3].startswith("application/"):
                size = item[13] if len(item) > 13 else None
                entries.append({"id": item[0], "name": item[2], "mime": item[3],
                    "bytes_from_listing": int(size) if str(size).isdigit() else None})
                return
        for child in item:
            visit(child)

    visit(value)
    return entries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = {"UTC": datetime.now(timezone.utc).isoformat(), "sources": {},
                 "weights_downloaded": False, "dependency_installation_performed": False,
                 "GPU_forward_backward_or_training_performed": False}
    for name, item in SOURCES.items():
        path = item["path"]
        command = lambda *args: subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
        commit = command("rev-parse", "HEAD")
        if commit != item["expected_commit"]:
            raise ValueError(f"Unexpected source commit: {name}: {commit}")
        tracked = command("ls-files", "-z").split("\0")
        hashes = {name_: hashlib.sha256((path/name_).read_bytes()).hexdigest()
                  for name_ in tracked if name_ and (path/name_).is_file()}
        record = {"repo": item["repo"], "commit": commit, "path": str(path),
                  "intended_model": item["intended_model"],
                  "git_status": command("status", "--porcelain"),
                  "origin": command("remote", "get-url", "origin"),
                  "tracked_files": len(hashes), "tracked_bytes": sum((path/n).stat().st_size for n in hashes),
                  "source_sha256": hashes,
                  "submodules_initialized": False, "Git_LFS_payloads_requested": False}
        response, receipt = public_get(f'https://api.github.com/repos/{item["repo"]}', OUT/name/"github_repo.json")
        record["github_receipt"] = receipt
        if response.ok:
            metadata = response.json()
            record["adoption_snapshot"] = {k: metadata.get(k) for k in
                ("stargazers_count", "forks_count", "archived", "updated_at", "pushed_at", "license")}
        inventory["sources"][name] = record
    save(OUT/"SOURCE_INVENTORY.json", inventory)

    model_id = "nvidia/GR00T-N1.5-3B"
    response, receipt = public_get(f"https://huggingface.co/api/models/{model_id}?blobs=true",
                                 OUT/"gr00t_n15/hf_model.json")
    if not response.ok:
        raise RuntimeError(f"Official model metadata unavailable: {receipt}")
    metadata = response.json()
    revision = metadata["sha"]
    assets = {"model_id": model_id, "revision": revision, "metadata_receipt": receipt,
              "safetensors_metadata": metadata.get("safetensors"), "gated": metadata.get("gated"),
              "license_metadata": metadata.get("cardData", {}).get("license"),
              "files": [], "small_asset_receipts": []}
    for file in metadata.get("siblings", []):
        if file["rfilename"].endswith((".safetensors", ".bin", ".pt", ".pth")):
            assets["files"].append(file)
    for filename in ["config.json", "model.safetensors.index.json", "README.md"]:
        _, receipt = public_get(f"https://huggingface.co/{model_id}/resolve/{revision}/{filename}",
                                OUT/"gr00t_n15"/filename)
        assets["small_asset_receipts"].append(receipt)
    save(OUT/"gr00t_n15/MODEL_ASSETS.json", assets)

    folder_id = "1rT8JKLhJGIo97jfYUm2JiFUrogOq-dgJ"
    response, receipt = public_get(f"https://drive.google.com/drive/folders/{folder_id}",
                                  OUT/"seer/official_checkpoint_folder.html")
    response.raise_for_status()
    folders = drive_listing(response.text)
    droid = next(item for item in folders if item["name"] == "Droid_Pre-trained")
    droid_response, droid_receipt = public_get(
        f'https://drive.google.com/drive/folders/{droid["id"]}',
        OUT/"seer/official_droid_checkpoint_folder.html")
    droid_response.raise_for_status()
    checkpoints = drive_listing(droid_response.text)
    checkpoint = next(item for item in checkpoints if item["name"] == "seer.pth")
    save(OUT/"seer/CHECKPOINT_ACCESS.json", {
        "UTC": datetime.now(timezone.utc).isoformat(),
        "official_folder_receipt": receipt, "parent_folder_entries": folders,
        "droid_folder_receipt": droid_receipt, "droid_folder_entries": checkpoints,
        "intended_checkpoint": checkpoint,
        "checkpoint_file_page": f'https://drive.google.com/file/d/{checkpoint["id"]}/view',
        "checkpoint_payload_downloaded": False, "strict_tensor_load_performed": False,
        "meaning": "Official public parent-to-DROID-folder provenance and file metadata only. No tensor payload downloaded, authenticated or shape-qualified."})
    print(json.dumps({"inventory": str(OUT/"SOURCE_INVENTORY.json"),
        "commits": {k:v["commit"] for k,v in inventory["sources"].items()},
        "gr00t_revision": revision, "gr00t_safetensors": assets["safetensors_metadata"],
        "seer_folder_http": response.status_code, "seer_intended_checkpoint": checkpoint}, indent=2))


if __name__ == "__main__":
    main()
