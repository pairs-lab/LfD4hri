"""TRAIN-only numeric compatibility scan and honest raw-camera pilot, no model/SR."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from VLA.family_transfer_r54.common import DATA
from VLA.new_vlas.dataset import NativeData

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"VLA/audits/backup_vlas_r93_20261022"


def main():
    os.nice(10)
    data = NativeData(DATA, verify_rgb_hashes=False)
    counts, sources = Counter(), Counter()
    translations, rotations = [], []
    for index, row in enumerate(data.rows):
        n = data.numeric(index)
        action = n["canonical7"]
        state = n["native8"]
        assert action.shape == (row["frames"], 7) and state.shape == (row["frames"], 8)
        assert np.isfinite(action).all() and np.isfinite(state).all()
        current = Rotation.from_rotvec(state[:, 3:6]).as_matrix()
        desired_r = current @ Rotation.from_euler("xyz", action[:, 3:6]).as_matrix()
        desired_p = state[:, :3] + action[:, :3]
        # Native real Seer anchors each relative command to the previous
        # requested pose, not the measured current pose after servo lag.
        anchor_r = np.concatenate((current[:1], desired_r[:-1]))
        anchor_p = np.concatenate((state[:1, :3], desired_p[:-1]))
        translations.append((anchor_r.transpose(0, 2, 1) @ (desired_p-anchor_p)[..., None])[..., 0])
        rotations.append(Rotation.from_matrix(anchor_r.transpose(0, 2, 1) @ desired_r).as_euler("xyz"))
        counts[row["task"]] += row["frames"]
        sources[row["source_kind"]] += 1
    trans, rot = np.concatenate(translations), np.concatenate(rotations)
    result = {"UTC": datetime.now(timezone.utc).isoformat(), "data": str(DATA),
        "trajectories": len(data.rows), "frames": sum(counts.values()),
        "action_frame_counts": dict(counts), "source_trajectory_counts": dict(sources),
        "numeric_cache_hashes_checked": True, "all_source_RGB_hashes_rechecked": False,
        "canonical_is_world_translation_local_xyz_Euler_signed_CLOSE": True,
        "seer_local_from_previous_command_scan": {
            "maximum_absolute_translation_per_axis_m": np.max(abs(trans), axis=0).tolist(),
            "maximum_absolute_Euler_per_axis_rad": np.max(abs(rot), axis=0).tolist(),
            "fraction_any_translation_exceeds_native_0_02m_scale": float(np.any(abs(trans)>.02, axis=1).mean()),
            "fraction_any_rotation_exceeds_native_0_05rad_scale": float(np.any(abs(rot)>.05, axis=1).mean()),
            "meaning": "Preliminary label-derived desired poses before plant bounds; not exact plant-replay certification. Native Tanh/scales must be audited before FT; do not silently clip."},
        "camera_pilot": [], "full_native_processor_parity_qualified": False,
        "no_evaluation_data_or_model_test_outcomes_used": True}
    for task in ("touch", "pick", "move", "pick_place"):
        episode = next(i for i,r in enumerate(data.rows) if r["task"] == task and r["source_kind"] == "complete_original")
        row = data.rows[episode]
        n, images = data.numeric(episode), data.rgb(episode)
        digest = hashlib.file_digest(Path(row["source"]).open("rb"), "sha256").hexdigest()
        assert digest == row["source_sha256"]
        frames = sorted(set([0, len(n["canonical7"])//2, len(n["canonical7"])-1]))
        sheet = Image.new("RGB", (3*320, 2*266), "white")
        draw = ImageDraw.Draw(sheet)
        for column, frame in enumerate(frames):
            for view_index, view in enumerate(("front_rgb", "wrist_rgb")):
                rgb = images[view][frame]
                assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
                assert float(rgb.std()) > 1
                sheet.paste(Image.fromarray(rgb).resize((320,240)), (column*320, view_index*266))
                draw.text((column*320+5, view_index*266+243), f"{task} {view} frame {frame}", fill="black")
                result["camera_pilot"].append({"task":task,"episode_index":episode,"frame":frame,
                    "view":view,"shape":list(rgb.shape),"source":row["source"],"source_sha256":digest,
                    "raw_rgb_sha256":hashlib.sha256(rgb.tobytes()).hexdigest()})
        sheet.save(OUT/f"RAW_CAMERA_PILOT_{task}.jpg", quality=94)
    result["passed"] = True
    (OUT/"DATA_PILOT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="camera_pilot"},indent=2))


if __name__ == "__main__":
    main()
