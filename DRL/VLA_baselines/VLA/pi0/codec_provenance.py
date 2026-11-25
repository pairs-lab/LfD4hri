"""Bind FAST source and decoded targets; no neural model or physics imports."""
import json
from pathlib import Path

from VLA.common.runtime_provenance import verify_source_environment_runtime
from VLA.common.source_payload_provenance import (
    verify_source_payload_for_profile,
    verify_source_payload_fingerprint,
)


def verify_fast_codec_binding(manifest, source, codec_root):
    """V9 requires full byte hashes for source and actual decoded action files.

    Legacy reports without new fields retain their historical compatibility;
    newly generated reports for any profile validate every field they include.
    The selected codec files need not equal the full source for diagnostics.
    """
    source = Path(source)
    profile = json.loads((source / "dataset_manifest.json").read_text()).get("observation_profile")
    is_v9 = (profile or {}).get("gripper_controller") == "body_contact_v9"
    verify_source_payload_for_profile(manifest, source)
    verify_source_environment_runtime(manifest, source)
    fingerprint = manifest.get("codec_payload_fingerprint")
    if is_v9 or fingerprint is not None:
        selected = manifest.get("selected_files")
        if not isinstance(selected, list) or not selected:
            raise RuntimeError("FAST codec selected-file payload identity is missing")
        source_files = {row["file"] for row in json.loads((source / "index.json").read_text())}
        if any(name not in source_files for name in selected):
            raise RuntimeError("FAST codec payload references an unknown source episode")
        return verify_source_payload_fingerprint(
            fingerprint, Path(codec_root), [{"file": name} for name in selected]
        )
