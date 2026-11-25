"""Export actual SmolVLA-native RGB for the original and repaired TRAIN case."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from VLA.audit_native_camera_tensors import transformer
from VLA.audit_staged_localization_frames import localization_frames
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repaired", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    transform, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.checkpoint)))
    args.output.mkdir(parents=True)
    rows = []
    for label, source in (("original", args.original), ("repaired", args.repaired)):
        report = json.loads((source/"results.json").read_text())
        if report["training_performed"] or report["final_test_used"] or not report["reset_exact"]:
            raise ValueError("Exact development TRAIN only")
        payload = source/report["episode"]
        digest = sha256_file(payload)
        with np.load(payload, allow_pickle=False) as arrays:
            frames = {**localization_frames(arrays["canonical_actions"]), "last_pre_action": len(arrays["canonical_actions"])-1}
            for stage, frame in frames.items():
                for camera in ("front", "wrist"):
                    results = transform(arrays[camera+"_rgb"][frame], arrays[camera+"_pick_mask"][frame])
                    if len(results) != 1:
                        raise ValueError("One actual SmolVLA image geometry expected")
                    _, rgb, mask = results[0]
                    path = args.output/f"{label}_{stage}_{camera}.png"
                    Image.fromarray(rgb).save(path)
                    ys, xs = np.nonzero(mask)
                    rows.append({"label": label, "stage": stage, "frame": frame, "camera": camera,
                        "path": str(path.resolve()), "sha256": sha256_file(path),
                        "source_payload_sha256": digest, "target_pixels": int(mask.sum()),
                        "target_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(xs) else None})
        if sha256_file(payload) != digest:
            raise ValueError("Actual source payload changed during extraction")
    (args.output/"audit.json").write_text(json.dumps({"scope": "One preserved TRAIN failure/repair native camera diagnostic_NOT_full_qualification",
        "image_method": method, "weights_loaded": False, "training_performed": False,
        "checkpoint_config_sha256": sha256_file(args.checkpoint/"config.json"), "rows": rows}, indent=2)+"\n")
    print(json.dumps({"images": len(rows), "training_performed": False}))


if __name__ == "__main__":
    main()
