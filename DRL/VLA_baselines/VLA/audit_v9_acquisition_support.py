"""TRAIN-only acquisition/recovery support and actual native phase previews.

This is a read-only data diagnostic, never a qualification or learned-success
certificate. Simulator grasp flags/masks select and annotate AUDIT frames only.
No policy, source payload, runtime controller or training recipe is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from VLA.common.provenance import sha256_file


TASKS = ("touch", "pick", "move", "pick_place")


def longest_run(values):
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def transition_support(arrays):
    """Align each pre-action observation with its actual post-action feedback.

    A closed hand without the requested target is NOT necessarily empty. It
    can be acquiring or holding a distractor. Do not turn this into a runtime
    grasp detector, or assume intent=measured closure.
    """
    action = np.asarray(arrays["canonical_actions"])
    closure = np.asarray(arrays["gripper_closure"])
    next_closure = np.asarray(arrays["next_gripper_closure"])
    requested = np.asarray(arrays["next_requested_gripper_closure"])
    correct_after = np.asarray(arrays["next_grasped_object"], dtype=bool)
    verified_after = np.asarray(arrays["next_verified_grasp"], dtype=bool)
    length = len(action)
    if (not length or action.shape != (length, 7)
            or any(value.shape != (length,) for value in
                   (closure, next_closure, requested, correct_after, verified_after))):
        raise ValueError("Malformed acquisition transitions")
    if not all(np.isfinite(value).all() for value in (action, closure, next_closure, requested)):
        raise ValueError("Non-finite acquisition values")
    if any(np.any((value < 0) | (value > 1)) for value in (closure, next_closure, requested)):
        raise ValueError("Gripper feedback/intent outside [0,1]")
    if not np.allclose(requested, (action[:, 6] + 1) / 2, atol=1e-7, rtol=0):
        raise ValueError("Applied canonical close intent differs from native label")
    if length > 1 and not np.allclose(next_closure[:-1], closure[1:], atol=1e-6, rtol=0):
        raise ValueError("Measured closure is not transition-synchronized")
    # Supported V9 sources begin unheld: ordinary home or explicitly unheld
    # startup. Held-at-reset data would need a recorded initial hold label.
    correct_before = np.r_[False, correct_after[:-1]]
    verified_before = np.r_[False, verified_after[:-1]]
    closing = action[:, 6] > .2
    close_without_target = closing & ~correct_before
    measured_closed_without_target = (closure > .5) & ~correct_before
    reopening = measured_closed_without_target & (action[:, 6] < -.2)
    acquired = correct_after & ~correct_before
    lost = correct_before & ~correct_after
    carry = correct_before & closing
    lift = carry & (action[:, 2] > .005)
    up_unconfirmed = close_without_target & ~correct_after & (action[:, 2] > .005)
    return {"closing_without_target": close_without_target,
            "measured_closed_without_target": measured_closed_without_target,
            "corrective_reopening": reopening, "target_acquired": acquired,
            "target_lost": lost, "verified_before": verified_before,
            "carry": carry, "lift_after_target_held": lift,
            "close_up_without_target_before_or_after": up_unconfirmed}


def phase_steps(arrays, support):
    length = len(arrays["canonical_actions"])
    selected = {"reset": 0}
    for label, key in (("corrective_reopen", "corrective_reopening"),
                       ("close_before_target", "closing_without_target"),
                       ("pre_acquisition", "target_acquired"),
                       ("lift_after_target_held", "lift_after_target_held"),
                       ("target_lost", "target_lost")):
        ids = np.flatnonzero(support[key])
        if len(ids):
            selected[label] = int(ids[0])
            if key == "target_acquired" and ids[0] + 1 < length:
                selected["post_acquisition_observation"] = int(ids[0] + 1)
    selected["last_pre_action_NOT_terminal"] = length - 1
    return selected


def weighted_support(index, per_episode, weights):
    if len(weights) != sum(row["steps"] for row in index) or not np.isclose(np.sum(weights), 1):
        raise ValueError("Sampling distribution does not match exact source transitions")
    masses, start = Counter(), 0
    for row in index:
        support = per_episode[row["file"]]["support"]
        length = row["steps"]
        for key in support:
            masses[key] += float(np.sum(weights[start:start + length][support[key]]))
        start += length
    return dict(masses)


def percentile_summary(values):
    return None if not len(values) else dict(zip(
        ("minimum", "p10", "median", "p90", "maximum"),
        map(float, np.percentile(values, (0, 10, 50, 90, 100)))))


def audit(args):
    from VLA.evaluate_smolvla_v9_source import verify_source_audit
    from VLA.audit_training_readiness import physical_scene_key
    from VLA.audit_native_camera_tensors import transformer

    source = args.source.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    proof, manifest = verify_source_audit(source, args.source_audit)
    if manifest.get("split") not in (None, "train"):
        raise ValueError("Only TRAIN or a TRAIN-only candidate, never outer validation")
    if manifest.get("gripper_state_contract") != "encoder_fk_aperture_closure_v1":
        raise ValueError("Requires actual V9 measured-aperture feedback")
    index = json.loads((source / "index.json").read_text())
    task_counts, per_episode, summaries = Counter(), {}, defaultdict(Counter)
    held_closure, closed_unconfirmed_closure = [], []
    target_pixels, critical_pixels = [], []
    candidates, recovery_names = defaultdict(list), []
    for row in index:
        name, filename = row["task"]["name"], row["file"]
        if name not in TASKS:
            raise ValueError("Unknown task")
        with np.load(source / filename, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in (
                "canonical_actions", "gripper_closure", "next_gripper_closure",
                "next_requested_gripper_closure", "next_grasped_object", "next_verified_grasp",
                "front_pick_mask", "wrist_pick_mask")}
        if len(arrays["canonical_actions"]) != row["steps"]:
            raise ValueError("Episode length differs from index")
        support = transition_support(arrays)
        steps = phase_steps(arrays, support)
        pixel_counts = np.stack([arrays[c + "_pick_mask"].reshape(row["steps"], -1).sum(1)
                                 for c in ("front", "wrist")])
        both_absent = np.max(pixel_counts, axis=0) == 0
        critical = (support["closing_without_target"] | support["target_acquired"]
                    | support["corrective_reopening"] | support["target_lost"])
        critical_ids = np.flatnonzero(critical)
        worst = int(critical_ids[np.argmin(np.max(pixel_counts[:, critical_ids], axis=0))]) if len(critical_ids) else 0
        steps["worst_acquisition_target_visibility"] = worst
        counts = {key: int(value.sum()) for key, value in support.items()}
        counts.update(frames=row["steps"], episodes=1,
                      critical_target_absent_in_both_views_frames=int((critical & both_absent).sum()),
                      target_absent_in_both_views_frames=int(both_absent.sum()),
                      episodes_with_acquisition=int(support["target_acquired"].any()),
                      episodes_with_corrective_reopening=int(support["corrective_reopening"].any()))
        summaries[name].update(counts)
        task_counts[name] += 1
        record = {"file": filename, "task": name, "target": row["task"]["pick_name"],
                  "documented_recovery_prefix": bool(row.get("recovery_start_actions_not_training_labels")),
                  "counts": counts, "phase_steps": steps,
                  "worst_critical_best_view_target_pixels_raw384": int(np.max(pixel_counts[:, worst])),
                  "longest_both_view_target_absence_control_ticks": longest_run(both_absent),
                  "max_measured_vs_requested_closure_gap": float(np.max(np.abs(
                      arrays["next_gripper_closure"] - arrays["next_requested_gripper_closure"])))}
        per_episode[filename] = {"record": record, "support": support}
        held_closure.extend(arrays["gripper_closure"][support["verified_before"]].tolist())
        closed_unconfirmed_closure.extend(arrays["gripper_closure"][support["measured_closed_without_target"]].tolist())
        target_pixels.extend(np.max(pixel_counts, axis=0).tolist())
        critical_pixels.extend(np.max(pixel_counts[:, critical], axis=0).tolist())
        if record["documented_recovery_prefix"]:
            recovery_names.append(filename)
        if name != "touch":
            candidates[name].append((record["worst_critical_best_view_target_pixels_raw384"], filename))
    if set(task_counts) != set(TASKS):
        raise ValueError("Four-task source required")
    # Every recovery prefix plus one deterministic worst acquisition per carry
    # primitive. Selection is data-quality evidence, never neural outcomes.
    preview_names = list(dict.fromkeys(recovery_names + [min(candidates[t])[1]
                         for t in ("pick", "move", "pick_place")]))
    args.output.mkdir(parents=True)
    conversion, method = transformer(SimpleNamespace(model=args.model,
        checkpoint=str(args.image_checkpoint) if args.image_checkpoint else None,
        processor=str(args.image_processor) if args.image_processor else None))
    cells, native_rows = [], []
    for filename in preview_names:
        record = per_episode[filename]["record"]
        with np.load(source / filename, allow_pickle=False) as archive:
            for phase, step in record["phase_steps"].items():
                native_images = []
                for camera in ("front", "wrist"):
                    backbone, rgb, mask = conversion(archive[camera + "_rgb"][step],
                                                     archive[camera + "_pick_mask"][step])[0]
                    stem = f"{Path(filename).stem}_{phase}_{camera}_{backbone}.png"
                    Image.fromarray(rgb).save(args.output / stem)
                    yy, xx = np.where(mask)
                    native_rows.append({"file": filename, "phase": phase, "step": step,
                        "camera": camera, "target_pixels": int(mask.sum()),
                        "target_bbox": [int(np.ptp(xx) + 1), int(np.ptp(yy) + 1)] if len(xx) else [0, 0],
                        "measured_closure": float(archive["gripper_closure"][step]),
                        "close_intent": float(archive["canonical_actions"][step, 6]),
                        "target_verified_before": bool(per_episode[filename]["support"]["verified_before"][step]),
                        "png": stem, "sha256": sha256_file(args.output / stem)})
                    native_images.append(Image.fromarray(rgb).resize((224, 224)))
                cell = Image.new("RGB", (448, 272), "black")
                draw = ImageDraw.Draw(cell)
                draw.text((3, 3), record["task"] + " / " + record["target"], fill="white")
                draw.text((3, 17), phase + f" t={step}", fill="white")
                draw.text((3, 31), f"closure={float(archive['gripper_closure'][step]):.3f}, cmd={float(archive['canonical_actions'][step,6]):+.0f}", fill="white")
                for k, picture in enumerate(native_images):
                    cell.paste(picture, (224 * k, 48))
                cells.append(cell)
    # Small per-episode sheets remain readable; do not make one giant image.
    for filename in preview_names:
        prefix = Path(filename).stem
        ids = [k for k, row in enumerate(native_rows[::2]) if row["file"] == filename]
        sheet = Image.new("RGB", (448 * 3, 272 * ((len(ids) + 2) // 3)), "black")
        for k, cell_id in enumerate(ids):
            sheet.paste(cells[cell_id], ((k % 3) * 448, (k // 3) * 272))
        sheet.save(args.output / (prefix + "_phase_sheet.png"))
    sampling = {}
    if args.smol_sampling_contract:
        from VLA.smolvla.balanced_sampling import frame_weights
        contract = json.loads(args.smol_sampling_contract.read_text())
        if contract["source_index_sha256"] != proof["index_sha256"] or contract["frames"] != sum(r["steps"] for r in index):
            raise ValueError("Actual Smol sampling contract differs from audited source")
        weights = frame_weights(index, contract["initial_frame_probability"], contract["target_balanced"])
        sampling["actual_smolvla_first_action_probability_mass"] = weighted_support(index, per_episode, weights)
    if args.rdt_train_config:
        config = json.loads(args.rdt_train_config.read_text())
        if (config["lineage"]["source_index_sha256"] != proof["index_sha256"]
                or config["lineage"]["episodes"] != len(index) or config.get("max_episodes") is not None):
            raise ValueError("Actual RDT training distribution differs from audited source")
        weights = np.concatenate([np.full(r["steps"], 1 / (len(index) * r["steps"])) for r in index])
        sampling["actual_rdt_uniform_episode_first_action_probability_mass"] = weighted_support(index, per_episode, weights)
    verify_source_audit(source, args.source_audit)
    report = {"scope": "TRAIN acquisition/recovery support diagnostic ONLY; no qualification/learned-success claim",
        "source": str(source), "source_index_sha256": proof["index_sha256"],
        "source_manifest_sha256": proof["manifest_sha256"], "source_audit_sha256": sha256_file(args.source_audit),
        "source_payload_tree_sha256": proof["source_payload_fingerprint"]["payload_tree_sha256"],
        "episodes": len(index), "independent_layouts": len({physical_scene_key(r["task"]) for r in index}),
        "tasks": dict(summaries), "measured_closure_verified_target_held": percentile_summary(held_closure),
        "measured_closure_over_half_without_requested_target": percentile_summary(closed_unconfirmed_closure),
        "critical_best_view_target_pixels_RAW384": percentile_summary(critical_pixels),
        "sampling_mass": sampling, "native_model": args.model, "native_method": method,
        "sampling_mass_interpretation": "Analytic expected first-action probabilities from the recorded recipes, NOT reconstructed actual microbatch draws",
        "preview_selection": "ALL documented recovery prefixes + worst critical target visibility per carry primitive",
        "native_previews": native_rows, "episode_records": [v["record"] for v in per_episode.values()],
        "manual_native_image_review_required": True, "neural_weights_loaded": False,
        "final_test_used": False, "validation_used": False, "training_performed": False,
        "data_sufficiency_established": False,
        "caveats": ["Stored images are PRE-action observations; the last frame is not the terminal post-action image.",
            "Grasp flags/masks are audit-only privileged labels, never new policy feedback.",
            "Without-requested-target is not proof of an empty hand; acquisition and distractor holding are possible.",
            "Probability masses concern first supervised actions, not future chunk targets or independent examples.",
            "Raw384 visibility counters are not native tensor visibility; exported native previews are a selected subset.",
            "Feedback separability and manual previews do not prove learned recovery or recognition."]}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return {k: report[k] for k in ("scope", "episodes", "independent_layouts", "tasks", "sampling_mass")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "source-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--model", choices=("smolvla", "rdt"), required=True)
    parser.add_argument("--image-checkpoint", type=Path)
    parser.add_argument("--image-processor", type=Path)
    parser.add_argument("--smol-sampling-contract", type=Path)
    parser.add_argument("--rdt-train-config", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2))


if __name__ == "__main__":
    main()
