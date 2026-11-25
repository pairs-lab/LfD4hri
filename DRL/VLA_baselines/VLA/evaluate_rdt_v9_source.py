"""RDT closed-loop source-scene diagnostics in the actual deployment simulator.

Frozen reset metadata is evaluator-only. No teacher action, target coordinate,
mask or grasp-success signal enters the learned policy's HTTP request.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import (reset_observation_errors, safety_abort_info, select_episodes,
                                           validate_reset_observation, verify_source_audit)
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.rdt.adapter import RDTPolicy
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit-report", "training-source", "training-audit", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split-mode", choices=("heldout", "train_diagnostic"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--integration-step-cap", type=int,
                        help="Explicit1..10-step integration-only diagnostic; never a task success-rate experiment")
    args = parser.parse_args()
    if args.integration_step_cap is not None and not 1 <= args.integration_step_cap <= 10:
        raise ValueError("Integration cap must be1..10; omit for unchanged full task horizons")
    if args.output.exists():
        raise FileExistsError(args.output)
    source, training = args.source.resolve(), args.training_source.resolve()
    audit, manifest = verify_source_audit(source, args.audit_report)
    train_audit, train_manifest = verify_source_audit(training, args.training_audit)
    if train_manifest.get("split") != "train":
        raise ValueError("Requires the actual RDT TRAIN source")
    if args.split_mode == "heldout":
        if manifest.get("split") not in {"val", "validation"} or manifest.get("final_test_used"):
            raise ValueError("Only frozen validation, never final test, is permitted")
        validate_holdout(training, source)
    elif source != training:
        raise ValueError("Train diagnostic must use the actual optimizer source")
    profile = checkpoint_profile(args.checkpoint)
    if profile is None or profile_hash(profile) != manifest["observation_profile_sha256"]:
        raise ValueError("Checkpoint/source camera and home mismatch")
    contract = json.loads((args.checkpoint / "pybullet_rdt_config.json").read_text())
    if (contract["lineage"]["source_index_sha256"] != train_audit["index_sha256"]
            or contract["lineage"]["source_manifest_sha256"] != train_audit["manifest_sha256"]
            or contract.get("scaling") != {"state": "identity_SI", "action": "identity_SI"}):
        raise ValueError("RDT deployed state/action scaling or training lineage differs")
    identity = verify_fingerprint(args.checkpoint)
    selected = select_episodes(json.loads((source / "index.json").read_text()), args.episodes_per_task)
    args.output.mkdir(parents=True)
    (args.output / "selected_episodes.json").write_text(json.dumps(selected, indent=2) + "\n")
    results = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=True) as env:
        policy = RDTPolicy(env, checkpoint=str(args.checkpoint.resolve()), port=args.port, seed=args.seed, replan_steps=1)
        recorder = VideoRecorder(args.output / "rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
        try:
            for item in selected:
                task = TaskSpec.from_dict(item["task"])
                observation = reset_source_episode(env, source, item)
                policy.reset(task)
                folder = args.output / Path(item["file"]).stem
                folder.mkdir()
                initial = bool(env.is_semantic_success())
                if initial:
                    raise ValueError("Invalid episode: already successful at reset")
                with np.load(source / item["file"], allow_pickle=False) as recorded:
                    errors = validate_reset_observation(reset_observation_errors(observation, env.normalized_gripper_closure(), recorded))
                (folder / "reset_observation.json").write_text(json.dumps(errors, indent=2) + "\n")
                trace, correct, wrong, info = [], False, False, None
                for step in range(min(task.max_steps, args.integration_step_cap or task.max_steps)):
                    if step % 10 == 0:
                        for camera in ("front", "wrist"):
                            Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"step_{step:04d}_{camera}.png")
                    recorder.add(observation, f"RDT {args.split_mode} | {task.prompt}")
                    action = policy.predict(observation)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid learned RDT action")
                    observation, _, done, info = env.step(action)
                    grasped = env.gripper.grasped_object
                    correct |= grasped == env.pick_id
                    wrong |= grasped is not None and grasped != env.pick_id
                    aborted, reason = safety_abort_info(info)
                    trace.append({"step": int(info["step"]), "action": action.tolist(),
                        "executed_action": info.get("executed_canonical_action"),
                        "safety_clamped_action": bool(info.get("safety_clamped_action", False)),
                        "semantic_success": bool(info["semantic_raw_success"]), "strict_success": bool(info["strict_raw_success"]),
                        "safety_abort": aborted, "safety_reason": reason})
                    if done:
                        break
                if info is None:
                    raise ValueError("Empty learned episode")
                aborted, reason = safety_abort_info(info)
                success = bool(info["success"] and not aborted)
                for camera in ("front", "wrist"):
                    Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"terminal_{info['step']:04d}_{camera}.png")
                recorder.add(observation, f"RDT {args.split_mode} | success={success}")
                results.append({"file": item["file"], "task": task.name, "prompt": task.prompt, "steps": int(info["step"]),
                    "success": success, "initial_semantic_success": initial, "correct_target_grasp_seen": correct,
                    "wrong_target_grasp_seen": wrong, "final_ee_object_distance_m": info["ee_object_distance"],
                    "final_goal_xy_distance_m": info.get("goal_xy_distance_m"), "safety_abort": aborted,
                    "safety_diagnostics_before_rollback": info.get("safety_diagnostics_before_rollback"),
                    "safety_abort_reason": reason, "reset_observation_check": errors,
                    "safety_clamped_steps": sum(row["safety_clamped_action"] for row in trace), "trace": trace})
                (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{item['file']}: success={success}, steps={info['step']}, abort={aborted}", flush=True)
        finally:
            recorder.close()
            policy.close()
    verify_source_audit(source, args.audit_report)
    verify_source_audit(training, args.training_audit)
    verify_fingerprint(args.checkpoint)
    report = {"scope": "RDT_frozen_source_scene_diagnostic_not_final_benchmark", "split_mode": args.split_mode,
        "source": str(source), "source_index_sha256": audit["index_sha256"],
        "source_manifest_sha256": audit["manifest_sha256"], "training_source": str(training),
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "seed": args.seed, "replan_steps": 1, "required_success_steps": 1,
        "semantic_success_thresholds": env.semantic_success_thresholds, "trials": len(results),
        "successes": sum(row["success"] for row in results), "safety_aborts": sum(row["safety_abort"] for row in results),
        "initial_false_positives": 0, "privileged_information_in_controls": False, "oracle_fallback": False,
        "final_test_used": False, "results": results}
    if args.integration_step_cap is not None:
        report.update(scope="RDT_bounded_pipeline_integration_ONLY_NOT_success_rate",
                      integration_step_cap=args.integration_step_cap,
                      full_task_horizons_executed=False, neural_success_claimed=False)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
