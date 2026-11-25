"""Replay completed TRAIN diagnostics to measure early covariate drift.

All saved TRAIN trials are included, not selected by success. No GPU/model is
loaded and no teacher action is executed. Expert poses/masks/labels are audit
references only. Actual camera tensors use the saved official image geometry.
This is not another learned rollout or a generalization/success certificate.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import pybullet as p

from VLA.audit_native_camera_tensors import transformer
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit, reset_observation_errors, validate_reset_observation
from VLA.fingerprint_checkpoint import verify_fingerprint


def trajectory_distance(observation, arrays):
    """Nearest SAME-episode supervised pose, without putting it in control."""
    tcp = np.asarray(observation.ee_position, dtype=np.float64)
    distance = np.linalg.norm(np.asarray(arrays["ee_position"]) - tcp, axis=1)
    nearest = int(np.argmin(distance))
    joint_delta = np.asarray(arrays["joint_positions"][nearest]) - observation.joint_positions
    joint_delta = np.arctan2(np.sin(joint_delta), np.cos(joint_delta))
    return {"nearest_expert_frame": nearest,
            "nearest_expert_tcp_distance_m": float(distance[nearest]),
            "nearest_expert_joint_wrapped_rms_rad": float(np.sqrt(np.mean(joint_delta**2))),
            "nearest_expert_action_audit_only": arrays["canonical_actions"][nearest].tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trace-report", "source-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.prefix_steps <= 12:
        raise ValueError("Fresh output and bounded1..12 early replay steps required")
    traces = json.loads(args.trace_report.read_text())
    if traces.get("split_mode") != "train_diagnostic" or traces.get("final_test_used") is not False:
        raise ValueError("Only completed TRAIN diagnostics may guide training repair")
    source = Path(traces["source"])
    audit, manifest = verify_source_audit(source, args.source_audit)
    if audit["index_sha256"] != traces["source_index_sha256"]:
        raise ValueError("Trace source index differs")
    identity = verify_fingerprint(Path(traces["checkpoint"]))
    if identity["content_tree_sha256"] != traces["checkpoint_content_sha256"]:
        raise ValueError("Completed trace checkpoint differs")
    selected = traces["results"]
    if len({row["file"] for row in selected}) != len(selected):
        raise ValueError("Duplicate trace episodes")
    index = {row["file"]: row for row in json.loads((source/"index.json").read_text())}
    render, method = transformer(SimpleNamespace(model="smolvla", checkpoint=traces["checkpoint"]))
    args.output.mkdir(parents=True)
    plan = {"scope": "saved_TRAIN_early_divergence_replay_NOT_new_neural_trials",
        "trace_report": str(args.trace_report.resolve()), "trace_report_sha256": sha256_file(args.trace_report),
        "source": str(source.resolve()), "source_index_sha256": audit["index_sha256"],
        "checkpoint_content_sha256": identity["content_tree_sha256"],
        "selected_files": [row["file"] for row in selected], "selection_rule": "ALL completed TRAIN diagnostic trials",
        "prefix_steps": args.prefix_steps, "teacher_actions_executed": False,
        "weights_loaded": False, "training_performed": False, "validation_used": False, "final_test_used": False,
        "model_image_method": method, "script_sha256": sha256_file(Path(__file__))}
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    rows, sheets = [], []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=True) as env:
        for trial in selected:
            item = index[trial["file"]]
            folder = args.output/Path(trial["file"]).stem
            folder.mkdir()
            observation = reset_source_episode(env, source, item)
            records = []
            with np.load(source/item["file"], allow_pickle=False) as arrays:
                reset = validate_reset_observation(reset_observation_errors(
                    observation, env.normalized_gripper_closure(), arrays))
                for step in range(min(args.prefix_steps, len(trial["trace"]))+1):
                    row = {"step": step, **trajectory_distance(observation, arrays),
                        "tcp_world_audit_only": np.asarray(observation.ee_position).tolist(),
                        "measured_aperture_closure": env.normalized_gripper_closure(),
                        "requested_closure_audit_only": env.commanded_gripper_closure(),
                        "correct_target_held_audit_only": env.gripper.grasped_object == env.pick_id,
                        "object_contact_count": len(p.getContactPoints(bodyA=env.gripper.body)),
                        "actual_target_xy_residual_m_audit_only": float(np.linalg.norm(
                            env.gripper.grasp_center()[:2]-np.asarray(p.getBasePositionAndOrientation(env.pick_id)[0])[:2]))}
                    if step < len(trial["trace"]):
                        action = trial["trace"][step]["action"]
                        row["next_saved_learned_action"] = action
                        reference = np.asarray(row["nearest_expert_action_audit_only"])
                        row["nearest_expert_translation_error_m"] = float(np.linalg.norm(np.asarray(action)[:3]-reference[:3]))
                    if step in (0, 5, args.prefix_steps):
                        tiles = []
                        for view in ("front", "wrist"):
                            rgb = np.asarray(getattr(observation, view+"_rgb"))
                            mask = env.target_mask(view, "pick")
                            native_name, native, native_mask = render(rgb, mask)[0]
                            Image.fromarray(rgb).save(folder/f"step{step:03d}_{view}_raw384.png")
                            Image.fromarray(native).save(folder/f"step{step:03d}_{view}_native512.png")
                            row[view+"_native_target_pixels_audit_only"] = int(native_mask.sum())
                            tile = Image.new("RGB", (256, 280), "white")
                            tile.paste(Image.fromarray(native).resize((256, 256)), (0, 24))
                            ImageDraw.Draw(tile).text((4, 5), f"{item['task']['name']} t={step} {view}", fill="black")
                            tiles.append(tile)
                        panel = Image.new("RGB", (512, 280), "white")
                        for number, tile in enumerate(tiles):
                            panel.paste(tile, (number*256, 0))
                        sheets.append(panel)
                    records.append(row)
                    if step == args.prefix_steps or step == len(trial["trace"]):
                        break
                    observation, _, done, info = env.step(trial["trace"][step]["action"])
                    if bool(info.get("safety_violation")) != bool(trial["trace"][step]["safety_abort"]):
                        raise ValueError("Saved learned action safety outcome is not reproducible")
                    if done:
                        break
            rows.append({"file": item["file"], "task": item["task"]["name"], "prompt": item["task"]["prompt"],
                         "reset_check": reset, "observations": records})
            (args.output/"progress.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(item["file"], "max_nearest_expert_tcp_mm", round(1000*max(r["nearest_expert_tcp_distance_m"] for r in records), 2), flush=True)
    sheet = Image.new("RGB", (512*3, 280*len(selected)), "white")
    for number, panel in enumerate(sheets):
        sheet.paste(panel, ((number%3)*512, (number//3)*280))
    sheet.save(args.output/"actual_model_inputs_contact_sheet.png")
    verify_source_audit(source, args.source_audit)
    verify_fingerprint(Path(traces["checkpoint"]))
    (args.output/"results.json").write_text(json.dumps({**plan, "rows": rows,
        "source_and_checkpoint_reverified": True, "neural_success_claimed": False}, indent=2)+"\n")


if __name__ == "__main__":
    main()
