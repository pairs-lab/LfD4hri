"""Versioned observation contract and source integrity for the new branch."""
import json
from pathlib import Path

from VLA.common.provenance import sha256_file
from VLA.stable_native_r44.conditioning import require_contract
from .scene_map import contract


def source_hashes():
    return {str(p.resolve()):sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def require_visual(checkpoint):
    cfg = require_contract(checkpoint)
    if cfg.get('privileged_visual_observation') != contract():
        raise ValueError('Own explicitly trained third-GT-image checkpoint required')
    if cfg.get('WARM_START_ONLY_DO_NOT_SERVE_OR_REPORT_AS_NEW_FIT'):
        raise ValueError('Warm metadata without new optimization is not servable')
    return cfg
