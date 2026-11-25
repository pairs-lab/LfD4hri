"""All TRAIN-derived near-miss starts: measured closure versus OPEN intervention.

Offline deployed predictions, NOT new rollouts/SR or a held-out generalization
test. The intervention changes only encoder aperture; it is not a real physical
state and cannot prove one unique causal explanation for robot failures.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.observation import VLAObservation
from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.audit_smolvla_conditioning import controlled_prediction
from VLA.smolvla.audit_conditioning import verify_explicit_source_audit
from VLA.smolvla.audit_checkpoint_predictions import validate_saved_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "source", "source-audit", "normalization-source", "normalization-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from VLA.evaluate_smolvla_v9_source import verify_source_audit
    from VLA.smolvla.adapter import SmolVLAPolicyAdapter
    from VLA.audit_native_camera_tensors import transformer

    identity = verify_fingerprint(args.checkpoint.resolve())
    verify_explicit_source_audit(args.source, args.source_audit)
    normal_audit, normal_manifest = verify_source_audit(args.normalization_source, args.normalization_audit)
    validate_saved_normalization(args.checkpoint.resolve(), args.normalization_source.resolve(), normal_audit, identity)
    manifest = json.loads((args.source / "dataset_manifest.json").read_text())
    if profile_hash(checkpoint_profile(args.checkpoint)) != manifest["observation_profile_sha256"]:
        raise ValueError("Source differs from checkpoint camera/controller")
    index = json.loads((args.source / "index.json").read_text())
    if {row["task"]["name"] for row in index} != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("Need four tasks")
    for row in index:
        if not row.get("near_object_recovery_start_evidence") or not row.get("recovery_start_actions_not_training_labels"):
            raise ValueError("Only actual near-object closed-empty TRAIN-derived starts")
    args.output.mkdir(parents=True)
    plan = {"scope": "ALL actual TRAIN-derived near-miss reset predictions; NOT rollouts or SR",
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "source": str(args.source.resolve()), "source_index_sha256": sha256_file(args.source / "index.json"),
        "source_audit_sha256": sha256_file(args.source_audit), "selected_files": [row["file"] for row in index],
        "normalization_source": str(args.normalization_source.resolve()), "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)), "training_performed": False,
        "final_test_used": False, "privileged_policy_inputs": False,
        "selection_uses_neural_outcomes": False, "same_noise_across_encoder_interventions": True,
        "intervened_OPEN_feedback_is_NOT_a_real_physical_state": True}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sensor = SimpleNamespace(closure=0.)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint.resolve()), seed=args.seed)
    render, method = transformer(SimpleNamespace(model="smolvla", checkpoint=str(args.checkpoint.resolve())))
    panels, reviewed_tasks, results = [], set(), []
    try:
        for item in index:
            task = TaskSpec.from_dict(item["task"])
            with np.load(args.source / item["file"], allow_pickle=False) as arrays:
                closure = float(arrays["gripper_closure"][0])
                teacher = arrays["canonical_actions"][0]
                if closure < .95 or teacher[6] != -1 or arrays["next_grasped_object"][0]:
                    raise ValueError("Invalid corrective reset/label, no neural conclusions")
                observation = VLAObservation(arrays["front_rgb"][0], arrays["wrist_rgb"][0],
                    arrays["joint_positions"][0], .085 * (1-closure), float(arrays["gripper_open"][0]),
                    arrays["ee_position"][0], arrays["ee_rotation_6d"][0], str(arrays["prompt"].item()), 0)
                measured = controlled_prediction(policy, task, observation, sensor, closure)
                repeated = controlled_prediction(policy, task, observation, sensor, closure)
                artificial_open = controlled_prediction(policy, task, observation, sensor, 0.)
                repeat_error = float(np.max(np.abs(measured - repeated)))
                if repeat_error > 1e-6:
                    raise ValueError("Controlled prediction did not repeat with same noise")
                results.append({"file": item["file"], "task": task.name, "prompt": task.prompt,
                    "actual_measured_closure": closure, "actual_requested_closure_before": 1.,
                    "teacher_action_audit_only": teacher.tolist(), "measured_feedback_prediction": measured.tolist(),
                    "artificial_open_feedback_prediction": artificial_open.tolist(),
                    "repeat_max_error": repeat_error,
                    "correct_actual_reopen_command": bool(measured[6] < -.2),
                    "feedback_intervention_gripper_abs_effect": float(abs(measured[6]-artificial_open[6])),
                    "feedback_intervention_translation_effect_m": float(np.linalg.norm(measured[:3]-artificial_open[:3]))})
                if task.name not in reviewed_tasks:
                    reviewed_tasks.add(task.name)
                    panel = Image.new("RGB", (1024, 550), "white")
                    ImageDraw.Draw(panel).text((8, 8), task.prompt + " | actual closed-empty | teacher OPEN", fill="black")
                    for number, view in enumerate(("front", "wrist")):
                        rgb = getattr(observation, view + "_rgb")
                        native = render(rgb, np.zeros(rgb.shape[:2], bool))[0][1]
                        name = task.name + "_" + view + "_native.png"
                        Image.fromarray(native).save(args.output / name)
                        panel.paste(Image.fromarray(native), (number*512, 32))
                    panels.append(panel)
            if len(results) % 19 == 0:
                (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print("actual near-miss predictions", len(results), "/", len(index), flush=True)
    finally:
        policy.close()
    verify_explicit_source_audit(args.source, args.source_audit)
    verify_source_audit(args.normalization_source, args.normalization_audit)
    if verify_fingerprint(args.checkpoint)["content_tree_sha256"] != identity["content_tree_sha256"]:
        raise ValueError("Checkpoint changed")
    sheet = Image.new("RGB", (1024, 550*len(panels)), "white")
    for number, panel in enumerate(panels):
        sheet.paste(panel, (0, 550*number))
    sheet.save(args.output / "actual_model_inputs_contact_sheet.png")
    summary = {}
    for task in ("touch", "pick", "move", "pick_place"):
        rows = [row for row in results if row["task"] == task]
        summary[task] = {"examples": len(rows), "correct_reopen_commands": sum(row["correct_actual_reopen_command"] for row in rows),
            "mean_gripper_feedback_intervention_effect": float(np.mean([row["feedback_intervention_gripper_abs_effect"] for row in rows])),
            "median_predicted_z_m": float(np.median([row["measured_feedback_prediction"][2] for row in rows]))}
    report = {**plan, "model_image_method": method, "summary": summary, "rows": results,
        "success_rate_claimed": False, "unique_causal_explanation_claimed": False}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
