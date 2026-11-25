"""Bind environment/controller source code used by a physics replay.

Camera/home profiles and dataset/asset identities remain separately bound.
This deliberately small, role-specific map is NOT a fingerprint of every
external dependency, PyBullet binary, or robot/object asset.
"""
from pathlib import Path
import json

from VLA.common.provenance import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_VERSION = 1
FINGERPRINT_ROLE = "environment_controller_source"
LEGACY_SOURCES = (
    "VLA/common/environment.py",
    "VLA/common/observation_profile.py",
    "VLA/common/robotiq.py",
)
V9_ADDITIONAL_SOURCES = (
    "VLA/common/body_environment.py",
    "VLA/common/body_gripper.py",
    "VLA/common/physics_safety.py",
    "VLA/common/environment_factory.py",
)


def _controller_contract(profile):
    contract = None if profile is None else profile.get("gripper_controller")
    if contract is None:
        return "legacy"
    if contract == "body_contact_v9":
        return contract
    raise ValueError(f"Unknown environment controller contract: {contract}")


def environment_runtime_fingerprint(profile, *, project_root=PROJECT_ROOT):
    """Return version/role/controller plus relative-path SHA-256 hashes."""
    contract = _controller_contract(profile)
    paths = LEGACY_SOURCES + (V9_ADDITIONAL_SOURCES if contract == "body_contact_v9" else ())
    root = Path(project_root)
    return {
        "version": FINGERPRINT_VERSION,
        "role": FINGERPRINT_ROLE,
        "controller_contract": contract,
        "sources_sha256": {path: sha256_file(root / path) for path in paths},
    }


def verify_environment_runtime_fingerprint(fingerprint, profile, *, project_root=PROJECT_ROOT):
    """Fail closed on absent/malformed/wrong-role/stale controller sources.

    A historical report without this field must be replayed to acquire it;
    it is not silently upgraded by hashing the current implementation.
    Returns the current fingerprint after exact validation.
    """
    expected = environment_runtime_fingerprint(profile, project_root=project_root)
    if not isinstance(fingerprint, dict):
        raise RuntimeError("Environment runtime fingerprint is missing or malformed")
    if (type(fingerprint.get("version")) is not int
            or fingerprint.get("version") != FINGERPRINT_VERSION
            or fingerprint.get("role") != FINGERPRINT_ROLE
            or fingerprint.get("controller_contract") != expected["controller_contract"]):
        raise RuntimeError("Environment runtime fingerprint version/role/controller mismatch")
    sources = fingerprint.get("sources_sha256")
    if not isinstance(sources, dict) or set(sources) != set(expected["sources_sha256"]):
        raise RuntimeError("Environment runtime fingerprint dependency list differs")
    stale = [path for path, digest in expected["sources_sha256"].items() if sources[path] != digest]
    if stale:
        raise RuntimeError("Environment runtime fingerprint is stale: " + ", ".join(stale))
    return expected


def verify_source_environment_runtime(report, source):
    """V9 replay reports must bind its real controller; legacy stays compatible."""
    profile = json.loads((Path(source)/"dataset_manifest.json").read_text()).get("observation_profile")
    fingerprint = report.get("environment_runtime_fingerprint")
    if (profile or {}).get("gripper_controller") == "body_contact_v9" or fingerprint is not None:
        return verify_environment_runtime_fingerprint(fingerprint, profile)
