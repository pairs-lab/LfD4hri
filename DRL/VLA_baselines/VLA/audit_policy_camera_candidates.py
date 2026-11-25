"""Render training-only camera prototypes and measure encoder-token support.

No environment source, active dataset, or current camera configuration is
changed. Token support is geometric coverage, not proof of semantic recognition.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from VLA.audit_finetune_data import resized_mask_metrics, mask_contrast
from VLA.common.environment import CAMERA_CONFIG, ManipulationEnv
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec


CANDIDATES = {
    "current_overview": CAMERA_CONFIG["front"],
    "table_oblique_a": {"eye_world": (-.45, -1.20, .90), "target_world": (0., -.50, .15), "up_world": (0., 0., 1.), "fov_degrees": 42.},
    "table_oblique_b": {"eye_world": (-.42, -1.05, .78), "target_world": (0., -.52, .12), "up_world": (0., 0., 1.), "fov_degrees": 45.},
    "table_oblique_c": {"eye_world": (-.65, -1.30, .75), "target_world": (-.02, -.46, .10), "up_world": (0., 0., 1.), "fov_degrees": 35.},
}


def token_support(mask, size, token_side):
    resized = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.BOX)) >= 128
    cells = size // token_side
    valid = resized[:cells * token_side, :cells * token_side]
    counts = valid.reshape(cells, token_side, cells, token_side).sum(axis=(1, 3))
    return {"cells_touched": int((counts > 0).sum()), "equivalent_full_cell_area": float(counts.sum() / token_side**2),
            "largest_cell_target_fraction": float(counts.max() / token_side**2), "grid": [cells, cells]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worst-preprocess-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    index = json.loads((args.source / "index.json").read_text())
    by_file = {row["file"]: row for row in index}
    preprocess = json.loads(args.worst_preprocess_report.read_text())
    files = list(dict.fromkeys(row["file"] for row in preprocess["samples"]))
    sheets = {phase: Image.new("RGB", (384 * len(CANDIDATES), 408 * len(files)), "black") for phase in ("reset", "middle", "end")}
    rows = []
    with ManipulationEnv(robot="ur5e", image_size=384, required_success_steps=1, seed=0) as env:
        for row_number, filename in enumerate(files):
            item = by_file[filename]
            with np.load(args.source / filename, allow_pickle=False) as episode:
                actions = episode["canonical_actions"]
            env.reset(TaskSpec.from_dict(item["task"]))
            selected = {0: "reset", len(actions) // 2: "middle", len(actions) - 1: "end"}
            for frame in range(len(actions)):
                if frame in selected:
                    phase = selected[frame]
                    for col, (name, camera) in enumerate(CANDIDATES.items()):
                        view = p.computeViewMatrix(camera["eye_world"], camera["target_world"], camera["up_world"])
                        projection = p.computeProjectionMatrixFOV(camera["fov_degrees"], 1., .01, 10.)
                        result = p.getCameraImage(384, 384, viewMatrix=view, projectionMatrix=projection,
                                                  renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
                        rgb = np.asarray(result[2], dtype=np.uint8).reshape(384, 384, 4)[..., :3]
                        ids = np.asarray(result[4]).reshape(384, 384) & ((1 << 24) - 1)
                        pick = ids == env.pick_id
                        place = ids == env.place_id if env.place_id is not None else np.zeros_like(pick)
                        gripper = ids == env.gripper.body
                        raw = args.output / "raw" / filename.replace(".npz", "")
                        raw.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rgb).save(raw / f"{phase}_{name}.png")
                        area, width, height = resized_mask_metrics(pick)
                        row = {"episode": filename, "task": item["task"]["name"], "target": item["task"]["pick_name"],
                               "phase": phase, "frame": frame, "candidate": name, "pick_pixels_224": area,
                               "pick_bbox_224": [width, height], "place_pixels_224": resized_mask_metrics(place)[0],
                               "gripper_pixels_224": resized_mask_metrics(gripper)[0], "target_contrast": mask_contrast(rgb, pick),
                               "geometric_token_support": {
                                   "pi0_224_patch14": token_support(pick, 224, 14),
                                   "rdt_384_patch14": token_support(pick, 384, 14),
                                   "smolvla_512_vit_patch16": token_support(pick, 512, 16),
                                   "smolvla_512_after_pixelshuffle4": token_support(pick, 512, 64),
                               }}
                        rows.append(row)
                        sheet = sheets[phase]
                        sheet.paste(Image.fromarray(rgb), (col * 384, row_number * 408 + 24))
                        ImageDraw.Draw(sheet).text((col * 384 + 3, row_number * 408 + 5),
                                                   f"{name} | {row['target']} | area224={area}", fill="white")
                env.step(actions[frame])
    for phase, sheet in sheets.items():
        sheet.save(args.output / f"{phase}_contact_sheet.png")
    summary = {}
    for name in CANDIDATES:
        selected = [row for row in rows if row["candidate"] == name and row["phase"] == "reset"]
        summary[name] = {"min_reset_pick_pixels_224": min(row["pick_pixels_224"] for row in selected),
                         "min_reset_gripper_pixels_224": min(row["gripper_pixels_224"] for row in selected),
                         "min_reset_smol_decoder_equivalent_token_area": min(row["geometric_token_support"]["smolvla_512_after_pixelshuffle4"]["equivalent_full_cell_area"] for row in selected)}
    report = {"measurement": "training_only_camera_prototype_not_used_by_current_models", "source": str(args.source.resolve()),
              "source_index_sha256": sha256_file(args.source / "index.json"), "candidates": CANDIDATES,
              "summary": summary, "rows": rows, "final_test_used": False,
              "caveat": "Token coverage is geometric, not encoder saliency or semantic recognition. Camera changes require full workspace/trajectory audits and recollected images before training."}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
