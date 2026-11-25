"""Audit temporal goal gaps through actual SmolVLA prepare_images, CPU only.

Uses source masks only for geometry measurements. No learned weights, physics
steps, source writes, or privileged model inputs are involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.common.spatial import FIXED_DESTINATION


def missing_runs(rows: list[dict], field: str, threshold: int) -> list[dict]:
    runs = []
    start = None
    for frame, row in enumerate([*rows, {field: threshold}]):
        if row[field] < threshold and start is None:
            start = frame
        elif row[field] >= threshold and start is not None:
            runs.append({"start_frame": start, "end_frame": frame - 1,
                         "steps": frame - start, "seconds_at_10hz": (frame - start) / 10})
            start = None
    return runs


def main():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.common import vla_utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-files", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    source = args.source.resolve()
    index = json.loads((source / "index.json").read_text())
    initial_index = sha256_file(source / "index.json")
    initial_manifest = sha256_file(source / "dataset_manifest.json")
    payload = source_payload_fingerprint(source, index)
    profile = json.loads((source / "dataset_manifest.json").read_text())["observation_profile"]
    runtime = environment_runtime_fingerprint(profile)
    if "highcenter" not in profile["name"]:
        raise ValueError("Expected new highcenter source profile")
    names = ("observation.images.camera1", "observation.images.camera2")
    dummy = SimpleNamespace(config=SimpleNamespace(image_features=dict.fromkeys(names),
                            resize_imgs_with_padding=(512, 512), empty_cameras=0))

    def transform(front, wrist):
        batch = {name: torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255
                 for name, image in zip(names, (front, wrist), strict=True)}
        # Call the EXACT policy method without instantiating/loading its model.
        images, masks = SmolVLAPolicy.prepare_images(dummy, batch)
        assert len(images) == 2 and all(bool(mask.all()) for mask in masks)
        assert all(image.shape == (1, 3, 512, 512) and torch.isfinite(image).all() for image in images)
        return images

    def counts(front, wrist):
        masks = [np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, -1) for mask in (front, wrist)]
        # prepare_images returns normalized [-1,1]; 0 is original coverage .5.
        return [int((image[0, 0] >= 0).sum()) for image in transform(*masks)]

    def panel(data, row, title):
        images = transform(data["front_rgb"][row["frame"]], data["wrist_rgb"][row["frame"]])
        result = Image.new("RGB", (1024, 548), "black")
        draw = ImageDraw.Draw(result)
        draw.text((4, 4), title, fill="white")
        draw.text((4, 17), f"t{row['frame']} held={row['verified_grasp_before_action']} xy={row['delta_xy_m']:.4f} dz={row['delta_z_m']:.4f}", fill="white")
        draw.text((4, 29), f"FRONT / WRIST native512 goalpx {row['front_goal_pixels_512']} / {row['wrist_goal_pixels_512']}; pickpx {row['front_pick_pixels_512']} / {row['wrist_pick_pixels_512']}", fill="white")
        for k, image in enumerate(images):
            rgb = np.rint((image[0].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            result.paste(Image.fromarray(rgb), (512 * k, 36))
        return result

    rows_all, summaries = [], []
    wanted = set(args.episode_files)
    selected = [item for item in index if item["file"] in wanted]
    if len(selected) != len(wanted):
        raise ValueError("Missing or duplicate requested episodes")
    first_transport_panels = []
    # 49px at224 has exactly 256px at512 for the same normalized area.
    equivalent_area_threshold = 256
    for item in selected:
        place = item["task"].get("place_name")
        if place is None or place in FIXED_DESTINATION:
            raise ValueError("Requested episode must have an actual physical goal object")
        rows = []
        with np.load(source / item["file"], allow_pickle=False) as data:
            verified = np.r_[False, data["next_verified_grasp"][:-1]].astype(bool)
            actions = data["canonical_actions"]
            for frame, action in enumerate(actions):
                pick = counts(data["front_pick_mask"][frame], data["wrist_pick_mask"][frame])
                goal = counts(data["front_place_mask"][frame], data["wrist_place_mask"][frame])
                delta_xy = float(np.linalg.norm(action[:2]))
                row = {"file": item["file"], "frame": frame,
                       "front_pick_pixels_512": pick[0], "wrist_pick_pixels_512": pick[1],
                       "combined_pick_pixels_512": max(pick),
                       "front_goal_pixels_512": goal[0], "wrist_goal_pixels_512": goal[1],
                       "combined_goal_pixels_512": max(goal),
                       "verified_grasp_before_action": bool(verified[frame]),
                       "horizontal_transport_after_verified": bool(verified[frame] and delta_xy >= .005),
                       "delta_xy_m": delta_xy, "delta_z_m": float(action[2]),
                       "canonical_action": action.tolist()}
                rows.append(row)
            transport = [r for r in rows if r["horizontal_transport_after_verified"]]
            first_transport = transport[0] if transport else None
            if first_transport is None:
                raise ValueError("No actual verified horizontal transport in episode")
            first_held = next(r["frame"] for r in rows if r["verified_grasp_before_action"])
            missing = [r["frame"] for r in rows if r["combined_goal_pixels_512"] < equivalent_area_threshold]
            selected_frames = sorted(set([0, first_held, first_transport["frame"], len(rows)-1,
                                          *([missing[0], missing[-1]] if missing else [])]))
            timeline = Image.new("RGB", (2048, 548 * ((len(selected_frames) + 1) // 2)), "black")
            for i, frame in enumerate(selected_frames):
                cell = panel(data, rows[frame], f"{item['file']} | {item['task']['prompt']}")
                timeline.paste(cell, ((i % 2) * 1024, (i // 2) * 548))
                cell.save(args.output / f"{item['file'][:-4]}_t{frame:03d}_native512.png")
            timeline.save(args.output / (item["file"][:-4] + "_native512_timeline.png"))
            first_transport_panels.append(panel(data, first_transport, f"FIRST verified horizontal transport | {item['file']}"))
        rows_all.extend(rows)
        summaries.append({
            "file": item["file"], "prompt": item["task"]["prompt"], "frames": len(rows),
            "first_verified_observation_frame": first_held,
            "first_horizontal_transport_frame": first_transport["frame"],
            "first_horizontal_transport_observation": first_transport,
            "combined_goal_zero_runs": missing_runs(rows, "combined_goal_pixels_512", 1),
            "combined_goal_below_native49_runs": missing_runs(rows, "combined_goal_pixels_512", 49),
            "combined_goal_below224equivalent49_runs": missing_runs(rows, "combined_goal_pixels_512", equivalent_area_threshold),
            "horizontal_transport_frames": len(transport),
            "horizontal_transport_both_goal_zero_frames": [r["frame"] for r in transport if r["combined_goal_pixels_512"] == 0],
            "horizontal_transport_both_goal_below224equivalent49_frames": [r["frame"] for r in transport if r["combined_goal_pixels_512"] < equivalent_area_threshold],
            "verified_grasp_both_goal_zero_frames": [r["frame"] for r in rows if r["verified_grasp_before_action"] and r["combined_goal_pixels_512"] == 0],
        })
        print(json.dumps(summaries[-1]), flush=True)
    sheet = Image.new("RGB", (1024, 548 * len(first_transport_panels)), "black")
    for i, cell in enumerate(first_transport_panels):
        sheet.paste(cell, (0, 548 * i))
    sheet.save(args.output / "first_transport_native512_contact_sheet.png")
    verify_environment_runtime_fingerprint(runtime, profile)
    if payload != source_payload_fingerprint(source, index):
        raise RuntimeError("Source payload changed")
    if initial_index != sha256_file(source / "index.json") or initial_manifest != sha256_file(source / "dataset_manifest.json"):
        raise RuntimeError("Source metadata changed")
    report = {"scope": "Actual SmolVLAPolicy.prepare_images on five hardest TRAIN physical-goal episodes, CPU only; no neural policy or qualification claim",
              "source": str(source), "source_index_sha256": initial_index,
              "source_manifest_sha256": initial_manifest, "source_payload_fingerprint": payload,
              "environment_runtime_fingerprint": runtime,
              "native_image_size": [512, 512], "control_hz": 10,
              "policy_preprocessing": "actual unbound prepare_images, two valid native camera slots; rawRGB/255 -> officialresize_with_pad -> [-1,1]",
              "preprocessing_source_sha256": sha256_file(Path(SmolVLAPolicy.prepare_images.__code__.co_filename)),
              "resize_helper_source_sha256": sha256_file(Path(vla_utils.__file__)),
              "thresholds": {"native49": 49, "native256_equivalent_area_to224x49": equivalent_area_threshold},
              "threshold_caveat": "These are image geometry diagnostics, not neural recognition guarantees",
              "temporal_scope": "Consecutive full episode observations; PRE-action verified hold is previous transition flag; transport xy>=.005m. dz separately reported; post-hold vertical lift is not mislabeled as horizontal transport",
              "summaries": summaries, "frames": rows_all}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
