"""Same-sensor/common-noise text-to-target discrimination, not rollout success.

Compares deployed decoded chunks against BOTH correct teacher branches. Target
coordinates, masks and teacher actions are audit references, never model input.
Predeclares ALL target-paired scenes; no low-loss/easy-case subset selection.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.observation_profile import profile_hash
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.audit_conditioning import verify_explicit_source_audit
from VLA.paired_decision_audit import paired_decision_frame


def grounding_metrics(predictions, teachers, separation_threshold_m=.002):
    """XY chunk error/assignment, accounting for shared early vertical motion."""
    n = min(*(len(x) for x in predictions), *(len(x) for x in teachers))
    if not n:
        raise ValueError("No common supervised horizon")
    predicted = np.asarray([x[:n, :2] for x in predictions], dtype=np.float64)
    expected = np.asarray([x[:n, :2] for x in teachers], dtype=np.float64)
    difference = expected[0] - expected[1]
    actual = predicted[0] - predicted[1]
    separation = float(np.sqrt(np.mean(np.sum(difference**2, axis=-1))))
    errors = np.sqrt(np.mean(np.sum((predicted[:, None] - expected[None])**2, axis=-1), axis=-1))
    informative = separation >= separation_threshold_m
    return {"common_horizon": n, "informative_teacher_xy_separation": informative,
        "teacher_xy_separation_rms_m": separation,
        "predicted_xy_separation_rms_m": float(np.sqrt(np.mean(np.sum(actual**2, axis=-1)))),
        "difference_projection_onto_teacher": float(np.sum(actual*difference)/np.sum(difference**2)) if informative else None,
        "branch_error_to_each_teacher_m": errors.tolist(),
        "requested_teacher_preferred": [bool(errors[i, i] < errors[i, 1-i]) for i in range(2)] if informative else None,
        "own_vs_swapped_assignment_margin_m": float(errors[0, 1]+errors[1, 0]-errors[0, 0]-errors[1, 1]) if informative else None,
        "identical_action_mean_baseline_error_m": float(separation/2),
        "caveat": "Offline XY action discrimination, not world-coordinate localization or closed-loop success"}


def paired_groups(index):
    groups = {}
    for row in index:
        if row.get("recovery_start_actions_not_training_labels"):
            raise ValueError("This target grounding audit requires ordinary paired resets")
        key = (row["task"]["name"], row.get("canonical_source"), row.get("paired_scene_source"))
        if not all(key):
            raise ValueError("Missing explicit canonical pair provenance")
        groups.setdefault(key, []).append(row)
    for pair in groups.values():
        if (len(pair) != 2 or {r.get("variant") for r in pair} != {0, 1}
                or len({r["task"]["pick_name"] for r in pair}) != 2):
            raise ValueError("Need exactly two different target branches per scene")
    return [(key, sorted(pair, key=lambda r: r["variant"])) for key, pair in groups.items()]


def ordinary_target_pairs_from_mixed_train(index):
    """Select declared ordinary target pairs, never outcomes or easy cases."""
    selected = [row for row in index if row.get("grounding_text_intervention") == "target"
                and not row.get("recovery_start_actions_not_training_labels")]
    if not selected:
        raise ValueError("Mixed TRAIN source contains no declared ordinary target pairs")
    groups = paired_groups(selected)
    if {key[0] for key, _ in groups} != {"touch", "pick", "move", "pick_place"}:
        raise ValueError("All four primitive target pairs are required")
    return groups


def main():
    import torch
    from transformers import AutoTokenizer
    from VLA.smolvla.adapter import SmolVLAPolicyAdapter

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "paired-data", "source-audit", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--noise-seeds", nargs="+", type=int, default=[20261010])
    parser.add_argument("--observation-frame", type=int, default=0,
                        help="Predeclare a same-sensor decision frame after a common physical prefix")
    parser.add_argument("--decision-stage", choices=("fixed", "first_text_dependent_action"), default="fixed",
                        help="Causal first branch decision from identical history; no later/easier frame selection")
    parser.add_argument("--mixed-train-source", action="store_true",
                        help="Audit ALL declared ordinary target pairs inside an already audited mixed TRAIN source")
    args = parser.parse_args()
    if args.output.exists() or len(set(args.noise_seeds)) != len(args.noise_seeds) or args.observation_frame < 0:
        raise ValueError("Fresh output and unique common-noise seeds required")
    if args.decision_stage != "fixed" and args.observation_frame != 0:
        raise ValueError("Use either a fixed frame or the declared causal decision rule")
    args.checkpoint, args.paired_data = args.checkpoint.resolve(), args.paired_data.resolve()
    def verify_data():
        if not args.mixed_train_source:
            return verify_explicit_source_audit(args.paired_data, args.source_audit)
        from VLA.evaluate_smolvla_v9_source import verify_source_audit
        audit, manifest = verify_source_audit(args.paired_data, args.source_audit)
        if manifest.get("split") != "train" or manifest.get("final_test_used") is not False:
            raise ValueError("Mixed grounding audit requires actual TRAIN, not validation/test")
        return audit
    verify_data()
    identity = verify_fingerprint(args.checkpoint)
    manifest = json.loads((args.paired_data/"dataset_manifest.json").read_text())
    if (not args.mixed_train_source and manifest.get("text_intervention") != "target") or manifest.get("final_test_used") is not False:
        raise ValueError("Requires target-paired TRAIN-only data")
    if profile_hash(checkpoint_profile(args.checkpoint)) != manifest["observation_profile_sha256"]:
        raise ValueError("Camera/controller profile differs")
    index = json.loads((args.paired_data/"index.json").read_text())
    groups = ordinary_target_pairs_from_mixed_train(index) if args.mixed_train_source else paired_groups(index)
    args.output.mkdir(parents=True)
    plan = {"scope": "counterfactual_target_grounding_diagnostic_NOT_success_rate",
        "checkpoint": str(args.checkpoint), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "paired_data": str(args.paired_data), "source_index_sha256": sha256_file(args.paired_data/"index.json"),
        "observation_frame": args.observation_frame,
        "decision_stage": args.decision_stage, "mixed_train_source": args.mixed_train_source,
        "causal_frame_helper_sha256": sha256_file(Path(__file__).parents[1]/"paired_decision_audit.py"),
        "source_audit_sha256": sha256_file(args.source_audit), "noise_seeds": args.noise_seeds,
        "selected_pairs": [[r["file"] for r in pair] for _, pair in groups],
        "selection_uses_neural_outcomes": False, "same_noise_across_text_branches": True,
        "privileged_policy_inputs": False, "training_performed": False, "final_test_used": False,
        "script_sha256": sha256_file(Path(__file__))}
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint/"tokenizer", local_files_only=True)
    sensor = SimpleNamespace(closure=0.)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint), replan_steps=1)
    rows = []
    try:
        policy._load()
        device = next(policy.model.parameters()).device
        for key, pair in groups:
            examples = []
            decision = None
            if args.decision_stage == "first_text_dependent_action":
                with np.load(args.paired_data/pair[0]["file"], allow_pickle=False) as first, \
                        np.load(args.paired_data/pair[1]["file"], allow_pickle=False) as second:
                    decision = paired_decision_frame(first, second, horizon=5)
            frame = args.observation_frame if decision is None else decision["frame"]
            for row in pair:
                with np.load(args.paired_data/row["file"], allow_pickle=False) as episode:
                    if frame >= len(episode["canonical_actions"]):
                        raise ValueError("Predeclared decision frame is outside a paired trajectory")
                    task = TaskSpec.from_dict(row["task"])
                    closure = float(episode["gripper_closure"][frame])
                    examples.append((task, VLAObservation(
                        front_rgb=episode["front_rgb"][frame], wrist_rgb=episode["wrist_rgb"][frame],
                        joint_positions=episode["joint_positions"][frame], gripper_width=.085*(1-closure),
                        gripper_open=float(episode["gripper_open"][frame]), ee_position=episode["ee_position"][frame],
                        ee_rotation_6d=episode["ee_rotation_6d"][frame], prompt=str(episode["prompt"].item()), step=frame),
                        closure, episode["canonical_actions"][frame:frame+policy.model.config.chunk_size]))
            for field in ("front_rgb", "wrist_rgb", "joint_positions", "ee_position", "ee_rotation_6d"):
                if not np.array_equal(getattr(examples[0][1], field), getattr(examples[1][1], field)):
                    raise ValueError("Text-only intervention changed sensor "+field)
            if examples[0][2] != examples[1][2]:
                raise ValueError("Text-only intervention changed actual aperture")
            for seed in args.noise_seeds:
                predictions, token_rows = [], []
                for task, observation, closure, _teacher in examples:
                    sensor.closure = closure
                    policy.reset(examples[0][0])
                    precision = torch.autocast("cuda", dtype=torch.bfloat16) if policy._fp32_masters else nullcontext()
                    with torch.inference_mode(), precision:
                        processed = policy.preprocess(policy.build_sensor_batch(observation))
                        tokens = processed["observation.language.tokens"][0]
                        mask = processed["observation.language.attention_mask"][0].bool()
                        used = tokens[mask].cpu().tolist()
                        full = tokenizer(observation.prompt+"\n", truncation=False)["input_ids"]
                        if used != full:
                            raise ValueError("Deployed text tokens differ from full instruction (possible truncation)")
                        token_rows.append({"input_ids": used, "decoded": tokenizer.decode(used), "token_count": len(used)})
                        state = processed["observation.state"]
                        generator = torch.Generator(device=device).manual_seed(seed)
                        noise = torch.randn((1, policy.model.config.chunk_size, policy.model.config.max_action_dim),
                            device=state.device, dtype=torch.float32, generator=generator)
                        chunk = policy.postprocess(policy.model.predict_action_chunk(processed, noise=noise))
                    values = chunk.squeeze(0).float().cpu().numpy()
                    if values.shape != (policy.model.config.chunk_size, 7) or not np.isfinite(values).all():
                        raise ValueError("Invalid deployed canonical prediction")
                    predictions.append(values)
                if token_rows[0]["input_ids"] == token_rows[1]["input_ids"]:
                    raise ValueError("Different target texts collapse to identical tokens")
                metrics = grounding_metrics(predictions, [x[3] for x in examples])
                rows.append({"task": key[0], "scene_source": key[2], "files": [r["file"] for r in pair],
                    "observation_frame": frame, "causal_frame_audit": decision,
                    "noise_seed": seed, "prompts": [x[0].prompt for x in examples], "token_audit": token_rows,
                    "predicted_chunks": [x.tolist() for x in predictions],
                    "teacher_chunks_audit_only": [x[3].tolist() for x in examples], **metrics})
                (args.output/"progress.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(f"{len(rows)}/{len(groups)*len(args.noise_seeds)} {key[0]} {key[2]}", flush=True)
    finally:
        policy.close()
    verify_data()
    verify_fingerprint(args.checkpoint)
    summaries = {}
    for task in ("touch", "pick", "move", "pick_place"):
        all_rows = [r for r in rows if r["task"] == task]
        useful = [r for r in all_rows if r["informative_teacher_xy_separation"]]
        summaries[task] = {"pairs_times_seeds": len(all_rows), "informative_cases": len(useful),
            "uninformative_shared_early_moves": len(all_rows)-len(useful),
            "both_requested_branches_preferred": sum(all(r["requested_teacher_preferred"]) for r in useful),
            "requested_branch_preference_fraction": float(np.mean([v for r in useful for v in r["requested_teacher_preferred"]])) if useful else None,
            "median_difference_projection": float(np.median([r["difference_projection_onto_teacher"] for r in useful])) if useful else None}
    report = {**plan, "sensor_identity_passed": True, "full_text_not_truncated": True,
        "summaries": summaries, "rows": rows, "neural_success_claimed": False}
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
