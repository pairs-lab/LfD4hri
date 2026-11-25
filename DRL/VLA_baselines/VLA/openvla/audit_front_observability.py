"""Read-only FRONT-only geometry through the actual OpenVLA CPU processor.

Ground-truth masks and grasp flags are audit metadata, NEVER policy inputs.
Neither nonzero visibility nor the diagnostic 49px threshold certifies neural
recognition or task performance. This tool does not qualify training sources.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import (
    environment_runtime_fingerprint,
    verify_environment_runtime_fingerprint,
)
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.common.spatial import FIXED_DESTINATION


def missing_runs(values: list[int], threshold: int) -> list[dict]:
    runs = []
    start = None
    for frame, value in enumerate([*values, threshold]):
        if value < threshold and start is None:
            start = frame
        elif value >= threshold and start is not None:
            runs.append({"start_frame": start, "end_frame": frame - 1,
                         "steps": frame - start, "seconds_at_10hz": (frame - start) / 10})
            start = None
    return runs


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"frames": 0}
    pixels = np.asarray([r["front_pixels_224"] for r in rows])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["file"]].append(row)
    longest = {"below49": None, "zero": None}
    episodes_below49 = episodes_zero = 0
    for file, episode_rows in grouped.items():
        counts = [r["front_pixels_224"] for r in episode_rows]
        for name, threshold in (("below49", 49), ("zero", 1)):
            runs = missing_runs(counts, threshold)
            if name == "below49":
                episodes_below49 += bool(runs)
            else:
                episodes_zero += bool(runs)
            for run in runs:
                # Input selections can skip phases; use actual observation
                # frames and split nonconsecutive portions before calling this
                # summary if exact phase-specific durations are needed.
                mapped = dict(run, file=file,
                              start_frame=episode_rows[run["start_frame"]]["frame"],
                              end_frame=episode_rows[run["end_frame"]]["frame"])
                if longest[name] is None or mapped["steps"] > longest[name]["steps"]:
                    longest[name] = mapped
    return {
        "frames": len(rows), "episodes": len(grouped),
        "minimum_pixels": int(pixels.min()), "median_pixels": float(np.median(pixels)),
        "below49_frames": int((pixels < 49).sum()),
        "visible_at_least49_fraction": float((pixels >= 49).mean()),
        "below16_frames": int((pixels < 16).sum()), "zero_frames": int((pixels == 0).sum()),
        "episodes_with_below49": episodes_below49, "episodes_with_zero": episodes_zero,
        "longest_below49_run": longest["below49"], "longest_zero_run": longest["zero"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--launch", action="store_true", help="Detached CPU-only audit with immutable log/launch receipt")
    args = parser.parse_args()
    if args.launch:
        launch_path = args.output.with_name(args.output.name + "_launch.json")
        log_path = args.output.with_name(args.output.name + "_driver.log")
        if args.output.exists() or launch_path.exists() or log_path.exists():
            raise FileExistsError("Preserve prior audit or launch; do not duplicate")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "VLA.openvla.audit_front_observability", "--source", str(args.source.resolve()),
            "--output", str(args.output.resolve()), "--model", args.model]
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        launch_path.write_text(json.dumps({"pid": process.pid, "CPU_only": True,
            "neural_weights_loaded": False, "command": command}, indent=2) + "\n")
        print(process.pid)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    source = args.source.resolve()
    index = json.loads((source / "index.json").read_text())
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    initial_index_sha, initial_manifest_sha = sha256_file(source / "index.json"), sha256_file(source / "dataset_manifest.json")
    payload = source_payload_fingerprint(source, index)
    profile = manifest.get("observation_profile")
    if profile is None or "highcenter" not in profile.get("name", ""):
        raise ValueError("This diagnostic is scoped to the new highcenter profile")
    runtime = environment_runtime_fingerprint(profile)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,
                                              local_files_only=True).image_processor
    if len(processor.tvf_normalize_params) != 2:
        raise ValueError("Expected OpenVLA's two native image backbones")
    means = [np.asarray(r["mean"])[:, None, None] for r in processor.tvf_normalize_params]
    stds = [np.asarray(r["std"])[:, None, None] for r in processor.tvf_normalize_params]

    def restored(tensor, backbone):
        return tensor[backbone * 3:backbone * 3 + 3].numpy() * stds[backbone] + means[backbone]

    def native_counts(mask):
        rgb = np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
        tensor = processor.apply_transform(Image.fromarray(rgb))
        return [int((restored(tensor, backbone)[0] >= .5).sum()) for backbone in range(2)]

    frames, episodes, startup = [], [], []
    worst = {}
    for number, item in enumerate(index, 1):
        task = item["task"]["name"]
        place = item["task"].get("place_name")
        physical_goal = place is not None and place not in FIXED_DESTINATION
        recovery = bool(item.get("recovery_start_actions_not_training_labels"))
        roles = ("pick", "physical_goal") if physical_goal else ("pick",)
        with np.load(source / item["file"], allow_pickle=False) as episode:
            n = len(episode["canonical_actions"])
            verified = np.r_[False, episode["next_verified_grasp"][:-1]].astype(bool)
            requested = episode["canonical_actions"][:, 6]
            measured = episode["gripper_closure"]
            # Each indexed NpzFile access decompresses a WHOLE member. Cache
            # each role mask once per episode without changing a single pixel.
            role_masks = {role: episode["front_pick_mask" if role == "pick" else "front_place_mask"]
                          for role in roles}
            episode_rows = []
            for frame in range(n):
                phase = ("verified_carry" if verified[frame] else
                         "close_unverified" if requested[frame] > .5 else "open_approach_or_retry")
                for role in roles:
                    counts = native_counts(role_masks[role][frame])
                    row = {"file": item["file"], "task": task, "frame": frame,
                           "role": role, "phase": phase, "is_recovery_episode": recovery,
                           "front_pixels_224": min(counts), "pixels_by_backbone": counts,
                           "target_name": item["task"]["pick_name"] if role == "pick" else place}
                    frames.append(row)
                    episode_rows.append(row)
                    categories = ["all"] + (["verified_carry"] if verified[frame] else [])
                    if frame == n - 1:
                        categories.append("end")
                    for category in categories:
                        identity = (task, role, category)
                        if identity not in worst or row["front_pixels_224"] < worst[identity]["front_pixels_224"]:
                            worst[identity] = dict(row, category=category)
                if recovery and frame == 0:
                    startup.append({"file": item["file"], "task": task,
                                    "prompt": item["task"]["prompt"],
                                    "measured_encoder_fk_closure": float(measured[0]),
                                    "requested_canonical_gripper": float(requested[0]),
                                    "target_pixels_224": episode_rows[-len(roles)]["front_pixels_224"],
                                    "physical_goal_pixels_224": episode_rows[-1]["front_pixels_224"] if physical_goal else None})
        episodes.append({"file": item["file"], "task": task, "steps": n,
                         "is_recovery_episode": recovery,
                         "roles": {role: summarize([r for r in episode_rows if r["role"] == role]) for role in roles}})
        if number % 20 == 0:
            progress = {"pid": os.getpid(), "status": "CPU_native_front_audit_in_progress",
                "completed_episodes": number, "planned_episodes": len(index),
                "completed_observation_frames": sum(r["steps"] for r in episodes),
                "neural_weights_loaded": False, "model": "openvla", "source": str(source)}
            temporary = args.output / "progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2) + "\n")
            temporary.replace(args.output / "progress.json")
            print(f"{number}/{len(index)} highcenter TRAIN episodes", flush=True)

    def panel_for(row, title):
        with np.load(source / row["file"], allow_pickle=False) as data:
            tensor = processor.apply_transform(Image.fromarray(data["front_rgb"][row["frame"]]))
        panel = Image.new("RGB", (448, 264), "black")
        draw = ImageDraw.Draw(panel)
        draw.text((3, 3), title, fill="white")
        draw.text((3, 16), f"{row['file']} t={row['frame']} {row['role']} px={row['front_pixels_224']}", fill="white")
        for backbone in range(2):
            rgb = np.rint(np.clip(restored(tensor, backbone), 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
            panel.paste(Image.fromarray(rgb), (backbone * 224, 40))
        return panel

    panels = []
    exported = []
    for (task, role, category), row in worst.items():
        name = f"{task}_{role}_{category}_worst.png"
        panel = panel_for(row, f"DINO / SigLIP | {task} {category}: {row['target_name']}")
        panel.save(args.output / name)
        exported.append(dict(row, png=name))
        if category == "all":
            panels.append(panel)
    sheet = Image.new("RGB", (896, 264 * ((len(panels) + 1) // 2)), "black")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2) * 448, (i // 2) * 264))
    sheet.save(args.output / "native_front_worst_contact_sheet.png")
    recovery_panels = []
    for row in startup:
        visibility = next(r for r in frames if r["file"] == row["file"] and r["frame"] == 0 and r["role"] == "pick")
        panel = panel_for(visibility, f"Recovery t0 | FKclosure={row['measured_encoder_fk_closure']:.3f} | {row['prompt']}")
        name = row["file"][:-4] + "_recovery_t0.png"
        panel.save(args.output / name)
        row["png"] = name
        recovery_panels.append(panel)
    if recovery_panels:
        sheet = Image.new("RGB", (896, 264 * ((len(recovery_panels) + 1) // 2)), "black")
        for i, panel in enumerate(recovery_panels):
            sheet.paste(panel, ((i % 2) * 448, (i // 2) * 264))
        sheet.save(args.output / "native_front_recovery_t0_contact_sheet.png")

    verify_environment_runtime_fingerprint(runtime, profile)
    if payload != source_payload_fingerprint(source, index):
        raise RuntimeError("Source payload changed during read-only audit")
    if initial_index_sha != sha256_file(source / "index.json") or initial_manifest_sha != sha256_file(source / "dataset_manifest.json"):
        raise RuntimeError("Source metadata changed during read-only audit")
    report = {
        "scope": "Actual OpenVLA native FRONT-only geometry on immutable highcenter TRAIN; no neural predictions, training qualification or performance claim",
        "policy_interface": ["front_rgb", "text"], "mask_and_contact_usage": "Audit-only; never model inputs",
        "source": str(source), "source_index_sha256": initial_index_sha,
        "source_manifest_sha256": initial_manifest_sha,
        "source_payload_fingerprint": payload, "environment_runtime_fingerprint": runtime,
        "episodes": len(index), "unique_observation_frames": sum(r["steps"] for r in episodes),
        "task_counts": dict(Counter(r["task"]["name"] for r in index)),
        "image_resize_strategy": processor.image_resize_strategy,
        "count_method": "real two-backbone apply_transform; inverse instantiated normalization; >=0.5; conservative min DINO/SigLIP",
        "threshold_caveat": "49 native224 pixels is an observability diagnostic, not a validated neural recognition/success threshold",
        "temporal_scope": "All-frame per-episode roles; 10Hz. Phase stats intentionally omit temporal runs because filtered observations need not be consecutive",
        "roles": {role: {"all": summarize([r for r in frames if r["role"] == role]),
                           "by_task": {task: summarize([r for r in frames if r["role"] == role and r["task"] == task]) for task in ("touch", "pick", "move", "pick_place")}}
                  for role in ("pick", "physical_goal")},
        "phase_frame_counts": {phase: {role: {k: v for k, v in summarize([r for r in frames if r["phase"] == phase and r["role"] == role]).items() if not k.startswith("longest_")}
                                      for role in ("pick", "physical_goal")}
                               for phase in ("open_approach_or_retry", "close_unverified", "verified_carry")},
        "episodes_summary": episodes, "recovery_startup": startup, "worst_frames": exported,
    }
    (args.output / "frame_visibility.json").write_text(json.dumps(frames, separators=(",", ":")) + "\n")
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "progress.json").write_text(json.dumps({"pid": os.getpid(), "status": "completed_CPU_geometry_audit",
        "completed_episodes": len(index), "planned_episodes": len(index),
        "completed_observation_frames": report["unique_observation_frames"], "neural_weights_loaded": False,
        "training_qualification": False}, indent=2) + "\n")
    print(json.dumps({"episodes": len(index), "roles": report["roles"], "recovery_startup": startup}, indent=2))


if __name__ == "__main__":
    main()
