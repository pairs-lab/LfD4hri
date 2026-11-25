"""Run under original Dita environment to test the actual server renderer."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from VLA.common.artifact_io import atomic_json
from VLA.privileged_visual_r53.scene_map import render

if __name__=='__main__':
    root=Path('VLA/audits/receipt_recovery_r88_20261021')
    p=json.loads((root/'REPRODUCED_REQUEST.json').read_text())
    actual=render(p['scene64'],'pick_place')
    reference=np.array(Image.open(root/'SIM_map_at_failure.png'))
    Image.fromarray(actual).save(root/'DITA_map_at_failure.png')
    atomic_json(root/'CROSS_VERSION_MAP.json',{'Dita_numpy':np.__version__,
        'Dita_map_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
        'SIM_map_sha256':hashlib.sha256(reference.tobytes()).hexdigest(),
        'differing_pixels':int(np.any(actual!=reference,axis=-1).sum()),
        'max_channel_error':int(abs(actual.astype(int)-reference.astype(int)).max()),
        'pixel_mismatch_reproduced':not np.array_equal(actual,reference)})
