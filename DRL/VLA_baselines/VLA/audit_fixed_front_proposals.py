"""TRAIN-only fixed-camera stress proposals on exactly replayed source states.

No active profile/data/robot/control changes. Raw-mask/PIL224 proposals are
NOT native encoder qualification, a promoted camera, or model success.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from VLA.audit_finetune_data import resized_mask_metrics
from VLA.common.observation_profile import camera_geometry, render_geometry
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "visibility-report", "train-episodes", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    manifest = json.loads((source/"dataset_manifest.json").read_text())
    report = json.loads(args.visibility_report.read_text())
    if (report["source_index_sha256"] != sha256_file(source/"index.json")
            or report["camera_views_used_for_visibility"] != ["front"]):
        raise ValueError("Requires this exact front-only audit")
    train = json.loads(args.train_episodes.read_text())
    if any(Path(row["source"]).resolve() != source for row in train):
        raise ValueError("TRAIN declaration refers to another source")
    allowed = {row["file"]: row["sha256"] for row in train}
    bad = {row["file"] for row in report["rows"] if row["failures"]}
    index = json.loads((source/"index.json").read_text())
    selected = [row for row in index if row["file"] in bad & allowed.keys()]
    if not selected:
        raise ValueError("No TRAIN-only camera stress cases")
    current = manifest["observation_profile"]["cameras"]["front"]
    cameras = {"current": current,
        "left_fixed": {**copy.deepcopy(current), "eye_world": [-.45, -1.30, 1.0]},
        "right_fixed": {**copy.deepcopy(current), "eye_world": [.45, -1.30, 1.0]},
        "higher_fixed": {**copy.deepcopy(current), "eye_world": [0, -1.30, 1.25], "fov_degrees": 38.}}
    args.output.mkdir(parents=True)
    rows, summaries, cells = [], [], []
    with audit_environment(source, image_size=384, render_observations=False,
                           required_success_steps=1, use_semantic_success=False) as env:
        for item in selected:
            path = source/item["file"]
            if sha256_file(path) != allowed[item["file"]]:
                raise ValueError("TRAIN payload changed")
            with np.load(path, allow_pickle=False) as arrays:
                reset_source_episode(env, source, item)
                worst, visibility = {}, {key: [] for key in cameras}
                maximum = 0.
                for t, action in enumerate(arrays["canonical_actions"]):
                    observation = env.observe()
                    maximum = max(maximum, float(np.max(np.abs(observation.joint_positions-arrays["joint_positions"][t]))),
                        float(np.max(np.abs(observation.ee_position-arrays["ee_position"][t]))))
                    if maximum > 1e-6:
                        raise RuntimeError("Source physics state does not reproduce")
                    travel = bool(t and arrays["next_grasped_object"][t-1]
                                  and np.linalg.norm(action[:2]) >= .005)
                    outcome = env.semantic_success_metrics()
                    semantic_before = env.is_semantic_success(outcome)
                    for name, camera in cameras.items():
                        rgb, seg, _, _ = render_geometry(env, camera_geometry(env, camera, "front"), camera["fov_degrees"], 384)
                        if name == "current" and not np.array_equal(rgb, arrays["front_rgb"][t]):
                            raise RuntimeError("Source front pixels do not reproduce exactly")
                        ids = np.where(seg >= 0, seg & ((1 << 24)-1), -1)
                        goal = ids == env.place_id
                        record = {"file": item["file"], "candidate": name, "frame": t,
                            "transport": travel, "already_semantic_success": semantic_before,
                            **outcome, "goal_raw384_pixels": int(goal.sum()),
                            "goal_pil224_pixels": resized_mask_metrics(goal)[0],
                            "pick_pil224_pixels": resized_mask_metrics(ids == env.pick_id)[0],
                            "reset_all_objects_min_pil224_pixels": min(resized_mask_metrics(ids == body)[0]
                                for body in env.object_ids.values()) if t == 0 else None}
                        rows.append(record)
                        if travel:
                            visibility[name].append(record["goal_raw384_pixels"] >= 25)
                            if name not in worst or record["goal_raw384_pixels"] < worst[name][0]["goal_raw384_pixels"]:
                                worst[name] = (record, rgb.copy())
                    _, _, done, info = env.step(action)
                    if info.get("safety_violation") or (done and t != len(arrays["canonical_actions"])-1):
                        raise RuntimeError("Exact source replay unexpectedly stopped")
                for name, values in visibility.items():
                    summaries.append({"file": item["file"], "candidate": name, "transport_frames": len(values),
                        "goal_visible_raw384_fraction": float(np.mean(values)), "source_state_max_error": maximum})
                    record, rgb = worst[name]
                    stem = Path(item["file"]).stem+"_"+name
                    Image.fromarray(rgb).save(args.output/(stem+"_raw384.png"))
                    cell = Image.new("RGB", (224, 258), "white")
                    cell.paste(Image.fromarray(rgb).resize((224, 224)), (0, 34))
                    ImageDraw.Draw(cell).text((2, 2), f"{name} t={record['frame']} px224={record['goal_pil224_pixels']}", fill="black")
                    ImageDraw.Draw(cell).text((2, 17), f"goalXY={record['goal_xy_distance_m']:.3f}m", fill="black")
                    cells.append(cell)
            print(item["file"]+": exact state/pixel replay; fixed-camera proposals rendered", flush=True)
    sheet = Image.new("RGB", (224*len(cameras), 258*len(selected)), "white")
    for i, cell in enumerate(cells):
        sheet.paste(cell, ((i%len(cameras))*224, (i//len(cameras))*258))
    sheet.save(args.output/"proposal_contact_sheet.png")
    output = {"scope": "TRAIN-only source-state camera proposals; NOT native encoder/full camera qualification",
        "source": str(source), "source_index_sha256": sha256_file(source/"index.json"),
        "selection": "ALL failed front-only cases within the frozen TRAIN split; no neural outcomes",
        "selected_files": [item["file"] for item in selected], "excluded_nontrain_failed_files": sorted(bad-allowed.keys()),
        "train_declaration_sha256": sha256_file(args.train_episodes), "candidates": cameras,
        "active_camera_or_dataset_changed": False, "final_test_used": False,
        "requires_all_tasks_workspace_and_native_preprocessing_before_promotion": True,
        "summaries": summaries, "rows": rows}
    (args.output/"audit.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
