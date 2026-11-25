"""Fail-closed train/deploy camera+home binding from native checkpoint lineage."""
import json
from pathlib import Path

from .observation_profile import validate_profile, profile_hash
from .provenance import sha256_file


def checkpoint_profile(checkpoint):
    if checkpoint is None:
        return None
    root = Path(checkpoint)
    if not root.is_dir():
        return None  # pretrained remote IDs have no target-domain profile
    path = root / "vla_checkpoint_manifest.json"
    if not path.exists():
        path = root / "train_config.json"  # OpenVLA native adapter metadata
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    lineage = metadata.get("lineage", metadata)
    if isinstance(lineage.get("source"), dict):
        lineage = lineage["source"]  # OpenPI manifest
    source = lineage.get("source")
    digest = lineage.get("source_manifest_sha256", lineage.get("manifest_sha256"))
    if not isinstance(source, str) or digest is None:
        return None
    source_manifest = Path(source) / "dataset_manifest.json"
    if not source_manifest.is_file() or sha256_file(source_manifest) != digest:
        raise ValueError("Checkpoint source manifest missing or changed; refusing silent sensor-domain mismatch")
    data = json.loads(source_manifest.read_text())
    profile = data.get("observation_profile")
    if profile is None:
        return None  # published legacy checkpoints still use legacy defaults
    profile = validate_profile(profile)
    if profile_hash(profile) != data.get("observation_profile_sha256"):
        raise ValueError("Source sensor-profile hash does not match")
    return profile


def resolve_profile(model, checkpoint=None, requested=None):
    requested_profile = (validate_profile(json.loads(Path(requested).read_text())) if requested else None)
    if model == "oracle":
        return requested_profile
    bound = checkpoint_profile(checkpoint)
    if requested_profile is not None and (bound is None or profile_hash(requested_profile) != profile_hash(bound)):
        raise ValueError("Requested camera/home not bound to this checkpoint's training source; recollect/audit/retrain first")
    return bound
