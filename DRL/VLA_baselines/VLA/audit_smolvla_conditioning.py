"""Controlled input interventions on a frozen policy; NOT success or causality.

Every intervention resets the deployed queue and uses the SAME baseline prompt
to seed diffusion noise. Changing prompt therefore cannot change the noise.
Black images are intentionally out-of-distribution diagnostics, not training
data. No coordinates, masks, teacher actions or success flags enter prediction.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.evaluate_smolvla_v9_source import select_episodes, verify_source_audit


def interventions(observation, alternate_prompt):
    if not alternate_prompt.strip() or alternate_prompt == observation.prompt:
        raise ValueError("Need a different, nonempty scene instruction")
    front = np.zeros_like(observation.front_rgb)
    wrist = np.zeros_like(observation.wrist_rgb)
    return {
        "baseline": observation,
        "repeat": observation,
        "alternate_target_text": replace(observation, prompt=alternate_prompt),
        "front_blackout": replace(observation, front_rgb=front),
        "wrist_blackout": replace(observation, wrist_rgb=wrist),
        "both_blackout": replace(observation, front_rgb=front, wrist_rgb=wrist),
    }


def controlled_prediction(policy, baseline_task, observation, sensor, closure):
    # Baseline task seeds ALL conditions, including changed observation.prompt.
    policy.reset(baseline_task)
    sensor.closure = float(closure)
    action = np.asarray(policy.predict(observation), dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("Invalid deployed action")
    return action


def effect(action, baseline):
    delta = np.asarray(action) - np.asarray(baseline)
    return {"translation_l2_m": float(np.linalg.norm(delta[:3])),
            "rotation_l2_rad": float(np.linalg.norm(delta[3:6])),
            "gripper_abs_difference": float(abs(delta[6]))}


def main():
    from VLA.audit_counterfactual_collection import alternate_target
    from VLA.fingerprint_checkpoint import verify_fingerprint
    from VLA.smolvla.adapter import SmolVLAPolicyAdapter
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout, validate_saved_normalization

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "training-source", "training-audit", "validation", "validation-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources = {"train": (args.training_source.resolve(), args.training_audit.resolve()),
               "heldout": (args.validation.resolve(), args.validation_audit.resolve())}
    proofs = {split: verify_source_audit(source, audit) for split, (source, audit) in sources.items()}
    validate_holdout(sources["train"][0], sources["heldout"][0])
    identity = verify_fingerprint(args.checkpoint.resolve())
    normalization = validate_saved_normalization(args.checkpoint.resolve(), sources["train"][0], proofs["train"][0], identity)
    selected = {split: select_episodes(json.loads((source / "index.json").read_text()), 2)
                for split, (source, _) in sources.items()}
    args.output.mkdir(parents=True)
    plan = {"scope": "controlled_conditioning_diagnostic_not_success_rate",
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
            "seed": args.seed, "same_noise_in_all_conditions": True, "neural_outcome_selection": False,
            "selection": {split: [row["file"] for row in rows] for split, rows in selected.items()},
            "sources": {split: {"path": str(source), "index_sha256": sha256_file(source / "index.json"),
                                   "audit_sha256": sha256_file(audit)} for split, (source, audit) in sources.items()},
            "script_sha256": sha256_file(Path(__file__)), "normalization": normalization,
            "training_performed": False, "final_test_used": False,
            "caveat": "Blackout sensitivity alone is neither correct grounding nor evidence of a causal failure."}
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sensor = SimpleNamespace(closure=0.)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint.resolve()), seed=args.seed)
    results = []
    try:
        for split, rows in selected.items():
            for item in rows:
                task = TaskSpec.from_dict(item["task"])
                alternate = alternate_target(task).prompt
                with np.load(sources[split][0] / item["file"], allow_pickle=False) as episode:
                    # One reset and one close-range expert observation. No
                    # future or intervening expert actions are executed.
                    frames = sorted({0, len(episode["canonical_actions"]) // 2})
                    for frame in frames:
                        closure = float(episode["gripper_closure"][frame])
                        observation = VLAObservation(
                            front_rgb=episode["front_rgb"][frame], wrist_rgb=episode["wrist_rgb"][frame],
                            joint_positions=episode["joint_positions"][frame], gripper_width=.085 * (1 - closure),
                            gripper_open=float(episode["gripper_open"][frame]),
                            ee_position=episode["ee_position"][frame], ee_rotation_6d=episode["ee_rotation_6d"][frame],
                            prompt=str(episode["prompt"].item()), step=frame)
                        predictions = {name: controlled_prediction(policy, task, obs, sensor, closure)
                                       for name, obs in interventions(observation, alternate).items()}
                        baseline = predictions["baseline"]
                        repeat_error = float(np.max(np.abs(predictions["repeat"] - baseline)))
                        if repeat_error > 1e-6:
                            raise RuntimeError("Common-noise repeatability gate failed")
                        expected = episode["canonical_actions"][frame].astype(np.float64)
                        results.append({"split": split, "episode": item["file"], "task": task.name,
                            "frame": frame, "prompt": observation.prompt, "alternate_prompt": alternate,
                            "repeat_max_abs_error": repeat_error, "teacher_action_audit_only": expected.tolist(),
                            "baseline_translation_error_m": float(np.linalg.norm(baseline[:3] - expected[:3])),
                            "predictions": {key: value.tolist() for key, value in predictions.items()},
                            "effects": {key: effect(value, baseline) for key, value in predictions.items()}})
                        (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{split}: {item['file']} | {len(results)} observations", flush=True)
    finally:
        policy.close()
    for source, audit in sources.values():
        verify_source_audit(source, audit)
    verify_fingerprint(args.checkpoint.resolve())
    summaries = {}
    for split in sources:
        subset = [row for row in results if row["split"] == split]
        summaries[split] = {"observations": len(subset),
            "mean_teacher_translation_error_m": float(np.mean([row["baseline_translation_error_m"] for row in subset])),
            "mean_translation_effect_m": {condition: float(np.mean([row["effects"][condition]["translation_l2_m"] for row in subset]))
                for condition in ("alternate_target_text", "front_blackout", "wrist_blackout", "both_blackout")}}
    report = {**plan, "repeatability_passed": True, "summaries": summaries, "rows": results,
              "neural_success_claimed": False}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
