"""Inspect committed candidate samples during collection; NEVER qualifies a dataset."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.provenance import sha256_file
from VLA.common.spatial import FIXED_DESTINATION


def native_preview(convert, rgb, mask):
    result = convert(rgb, mask)
    if len(result) != 1:
        raise ValueError("Smol geometry preview expects exactly one backbone")
    _, native, native_mask = result[0]
    if native.shape != (512, 512, 3):
        raise ValueError("Saved model does not use the expected audited 512px geometry")
    return native, native_mask


def audit_arrays(arrays, row):
    n = row["steps"]
    errors = []
    for key in ("front_rgb", "wrist_rgb"):
        value = arrays[key]
        if value.shape != (n, 384, 384, 3) or value.dtype != np.uint8:
            errors.append(key + ": unexpected raw image contract")
    for key, shape in (("joint_positions", (n, 6)), ("canonical_actions", (n, 7)),
                       ("next_observed_joint_positions", (n, 6)),
                       ("ee_position", (n, 3)), ("ee_rotation_6d", (n, 6)),
                       ("gripper_open", (n,)),
                       ("next_joint_targets", (n, 6)), ("gripper_closure", (n,)),
                       ("next_gripper_closure", (n,)), ("next_requested_gripper_closure", (n,))):
        value = arrays[key]
        if value.shape != shape or not np.isfinite(value).all():
            errors.append(key + ": wrong shape/nonfinite")
    if str(arrays["prompt"]) != row["task"]["prompt"]:
        errors.append("text mismatch")
    for camera in ("front", "wrist"):
        for role in ("pick", "place", "robot"):
            mask = arrays[f"{camera}_{role}_mask"]
            if mask.shape != (n, 384, 384) or mask.dtype != np.bool_:
                errors.append(f"{camera}_{role}_mask: unexpected segmentation contract")
    if not np.allclose(arrays["gripper_open"], 1 - arrays["gripper_closure"], atol=1e-9, rtol=0):
        errors.append("open feedback is not the complement of measured closure")
    rotation = arrays["ee_rotation_6d"]
    if rotation.shape == (n, 6):
        first, second = rotation[:, :3], rotation[:, 3:]
        if (not np.allclose(np.linalg.norm(first, axis=1), 1, atol=1e-5, rtol=0)
                or not np.allclose(np.linalg.norm(second, axis=1), 1, atol=1e-5, rtol=0)
                or not np.allclose(np.sum(first * second, axis=1), 0, atol=1e-5, rtol=0)):
            errors.append("TCP rotation6d is not an orthonormal column pair")
    for key in ("gripper_closure", "next_gripper_closure", "next_requested_gripper_closure"):
        if np.any((arrays[key] < 0) | (arrays[key] > 1)):
            errors.append(key + ": closure out of range")
    if n > 1:
        if not np.allclose(arrays["joint_positions"][1:], arrays["next_observed_joint_positions"][:-1], atol=1e-6, rtol=0):
            errors.append("joint before/after time alignment")
        if not np.allclose(arrays["gripper_closure"][1:], arrays["next_gripper_closure"][:-1], atol=1e-9, rtol=0):
            errors.append("measured gripper before/after time alignment")
    expected_intent = (arrays["canonical_actions"][:, 6] > 0).astype(float)
    if not np.array_equal(expected_intent, arrays["next_requested_gripper_closure"]):
        errors.append("requested close intent differs from canonical label")
    if not row["success"] or row.get("safety_violation") is not None:
        errors.append("failed/unsafe collection must be preserved and diagnosed")
    visibility = {}
    verified_before = np.r_[False, np.asarray(arrays["next_verified_grasp"][:-1], dtype=bool)]
    transport = verified_before & (np.linalg.norm(arrays["canonical_actions"][:, :2], axis=1) >= .005)
    for role in ("pick", "place"):
        front = np.count_nonzero(arrays[f"front_{role}_mask"], axis=(1, 2))
        wrist = np.count_nonzero(arrays[f"wrist_{role}_mask"], axis=(1, 2))
        applicable = role == "pick" or row["task"].get("place_name") not in (None, *FIXED_DESTINATION)
        visible = (front > 0) | (wrist > 0)
        visibility[role] = {"mask_applicable": applicable,
                            "reset_front_px": int(front[0]), "reset_wrist_px": int(wrist[0]),
                            "either_view_visible_fraction": float(np.mean(visible)) if applicable else None,
                            "both_views_invisible_frames": np.flatnonzero(~visible).tolist() if applicable else [],
                            "verified_horizontal_transport_frames": int(transport.sum()),
                            "visible_during_verified_horizontal_transport_fraction": float(np.mean(visible[transport])) if applicable and transport.any() else None,
                            "note": "Raw-mask diagnostic only, NOT model-resolution visibility gate" if applicable else "No physical destination body: empty segmentation is NOT a camera failure"}
    return {"errors": errors, "visibility": visibility,
            "max_command_translation_m": float(np.max(np.abs(arrays["canonical_actions"][:, :3]))),
            "close_intent_frames": int(expected_intent.sum()),
            "verified_grasp_frames": int(np.count_nonzero(arrays["next_verified_grasp"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--smol-checkpoint", type=Path, help="Inspect official saved-config 512px image geometry without loading weights")
    args = parser.parse_args()
    if args.samples < 1 or args.output.exists():
        raise ValueError("Use a positive sample count and a NEW audit output directory")
    source, output = args.source.resolve(), args.output.resolve()
    # Snapshot only the committed index; the writer may append later. Selected
    # immutable NPZ payloads are independently hash-bound and rechecked below.
    index_text = (source / "index.json").read_text()
    rows = json.loads(index_text)
    if not rows:
        raise ValueError("No committed episodes")
    selected = sorted(set(np.linspace(0, len(rows)-1, min(args.samples, len(rows)), dtype=int).tolist()))
    output.mkdir(parents=True)
    (output / "index_snapshot.json").write_text(index_text)
    sheet = Image.new("RGB", (384 * 6, (384 + 38) * len(selected)), "white")
    draw = ImageDraw.Draw(sheet)
    convert = native_sheet = native_draw = None
    if args.smol_checkpoint:
        from VLA.audit_native_camera_tensors import transformer
        convert, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.smol_checkpoint)))
        native_sheet = Image.new("RGB", (512*6, (512+38)*len(selected)), "white")
        native_draw = ImageDraw.Draw(native_sheet)
    reports = []
    for sheet_row, index in enumerate(selected):
        row = rows[index]
        path = source / row["file"]
        digest = sha256_file(path)
        with np.load(path, allow_pickle=False) as arrays:
            result = audit_arrays(arrays, row)
            for column, t in enumerate((0, row["steps"]//2, row["steps"]-1)):
                for camera_number, camera in enumerate(("front", "wrist")):
                    x, y = (2*column + camera_number)*384, sheet_row*422
                    image = Image.fromarray(arrays[camera + "_rgb"][t])
                    sheet.paste(image, (x, y+38))
                    draw.text((x+4, y+2), f"{row['file']} t={t} {camera}", fill="black")
                    draw.text((x+4, y+17), row["task"]["prompt"][:58], fill="black")
                    if convert is not None:
                        native, mask = native_preview(convert, arrays[camera + "_rgb"][t], arrays[camera + "_pick_mask"][t])
                        nx, ny = (2*column + camera_number)*512, sheet_row*550
                        native_sheet.paste(Image.fromarray(native), (nx, ny+38))
                        native_draw.text((nx+4, ny+2), f"{row['file']} t={t} {camera}", fill="black")
                        native_draw.text((nx+4, ny+17), row["task"]["prompt"][:70], fill="black")
                        result.setdefault("native_image_samples", []).append({"frame": t, "camera": camera,
                            "size": list(native.shape), "target_pixels": int(np.count_nonzero(mask))})
        if sha256_file(path) != digest:
            raise RuntimeError("Selected committed payload changed during audit")
        reports.append({"file": row["file"], "sha256": digest, "task": row["task"], **result})
    sheet.save(output / "raw_observation_contact_sheet.png")
    if native_sheet is not None:
        native_sheet.save(output / "smol_native512_contact_sheet.png")
        if len(selected) > 4:
            for first_row in range(0, len(selected), 4):
                native_sheet.crop((0, first_row*550, native_sheet.width, min(first_row+4, len(selected))*550)).save(
                    output / f"smol_native512_page_{first_row//4:02d}.png")
    report = {"scope": "partial committed candidate diagnostic, NOT full camera audit or qualification",
              "source": str(source), "committed_snapshot_episodes": len(rows),
              "task_counts_in_snapshot": dict(Counter(row["task"]["name"] for row in rows)),
              "sampled_episodes": len(reports), "snapshot_sha256": sha256_file(output / "index_snapshot.json"),
              "sample_health_checks_passed": all(not row["errors"] for row in reports),
              "training_certified": False, "model_preprocessing_audited": False,
              "segmentation_used_for_model_inputs": False, "episodes": reports}
    if convert is not None:
        report["native_image_geometry_preview"] = {"method": method, "weights_loaded": False,
            "checkpoint": str(args.smol_checkpoint.resolve()),
            "config_sha256": sha256_file(args.smol_checkpoint / "config.json"),
            "full_training_deployment_input_parity_claimed": False}
    (output / "partial_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "episodes"}, indent=2))
    if not report["sample_health_checks_passed"]:
        raise RuntimeError("Candidate sample health checks failed; inspect preserved report")


if __name__ == "__main__":
    main()
