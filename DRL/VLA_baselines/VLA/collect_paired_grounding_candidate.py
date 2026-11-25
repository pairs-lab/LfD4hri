"""Bounded TRAIN-only, same-image/different-text teacher data candidate.

No source, benchmark or learned control changes. These paired copies do not
increase independent scene count and remain unqualified until camera/replay
and model conversion audits pass. No neural outcomes select the source scenes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.audit_counterfactual_collection import alternate_target, alternate_destination, physical_scene_spec, trajectory_action_difference
from VLA.audit_training_readiness import physical_scene_key
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import TASKS, reset_observation_errors, validate_reset_observation, verify_source_audit


def select_grounding_rows(rows, count=2, tasks=TASKS):
    if type(count) is not int or not 1 <= count <= 4:
        raise ValueError("This candidate supports 1..4 independent layouts per task")
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError("Unknown grounding primitive selection")
    selected = []
    for task in tasks:
        ordinary = [row for row in rows if row["task"]["name"] == task
                    and not row.get("paired_scene_source")
                    and not row.get("recovery_start_actions_not_training_labels")]
        chosen, seen = [], set()
        # Source order only. Do not prefer successful neural outcomes or easy
        # target shapes. Missing an alternate target fails, rather than skips.
        for row in ordinary:
            group = physical_scene_key(row["task"])
            if group not in seen:
                chosen.append(row); seen.add(group)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError("Insufficient ordinary, independent TRAIN layouts: " + task)
        selected.extend(chosen)
    return selected


def select_target_coverage_rows(rows, tasks=TASKS):
    """First ordinary TRAIN layout per requested object/task; no outcomes.

    These are instruction interventions in existing layouts, NOT additional
    independent scenes. Recovery/previously paired resets are excluded.
    """
    if not tasks or set(tasks) - set(TASKS):
        raise ValueError("Unknown grounding primitive selection")
    selected = []
    for task in tasks:
        target_names = sorted({row["task"]["pick_name"] for row in rows if row["task"]["name"] == task})
        if not target_names:
            raise ValueError("Missing task vocabulary")
        seen = set()
        for target in target_names:
            candidates = [row for row in rows if row["task"]["name"] == task and row["task"]["pick_name"] == target
                          and not row.get("paired_scene_source") and not row.get("recovery_start_actions_not_training_labels")
                          and physical_scene_key(row["task"]) not in seen]
            if not candidates:
                raise ValueError("Missing ordinary independent TRAIN layout: " + task + "/" + target)
            selected.append(candidates[0])
            seen.add(physical_scene_key(candidates[0]["task"]))
    return selected


class FrozenRetargetReset:
    """Teacher-only proxy: actual frozen reset first, then task label change."""
    def __init__(self, env, source, row):
        self.env, self.source, self.row = env, source, row

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        original = TaskSpec.from_dict(self.row["task"])
        if (task.name != original.name or task.task_id != original.task_id
                or physical_scene_spec(task) != physical_scene_spec(original)):
            raise ValueError("Counterfactual changed the primitive or physical scene")
        if self.env.observation_profile.get("grasp_candidate_scope") != "all_dynamic_scene_objects":
            raise ValueError("Teacher retargeting must not change controller candidate scope")
        reset_source_episode(self.env, self.source, self.row)
        self.env.task = task
        self.env.pick_id = self.env.object_ids[task.pick_name]
        self.env.place_id = self.env.object_ids.get(task.place_name)
        position = p.getBasePositionAndOrientation(self.env.pick_id)[0]
        self.env.initial_pick_z = position[2]
        self.env.initial_pick_position = np.asarray(position, dtype=np.float64)
        self.env.initial_pick_top, self.env.initial_pick_bottom = self.env._pick_top(), self.env._pick_bottom()
        self.env.initial_place_top = self.env._body_top(self.env.place_id) if self.env.place_id is not None else None
        return self.env.observe()


def pair_errors(first, second):
    if str(first["prompt"]) == str(second["prompt"]):
        raise ValueError("Paired prompts must differ")
    errors = {"initial_joint_max_abs_error": float(np.max(np.abs(first["joint_positions"][0] - second["joint_positions"][0]))),
              "initial_measured_closure_abs_error": abs(float(first["gripper_closure"][0]) - float(second["gripper_closure"][0]))}
    for camera in ("front", "wrist"):
        errors[camera + "_initial_max_pixel_error"] = int(np.max(np.abs(
            first[camera + "_rgb"][0].astype(np.int16) - second[camera + "_rgb"][0].astype(np.int16))))
    different = trajectory_action_difference(first["canonical_actions"], second["canonical_actions"])
    if any(value != 0 for value in errors.values()) or not (
            different["common_prefix_max_action_difference"] > .001 or different["different_trajectory_lengths"]):
        raise ValueError("Same-image/different-text grounding pair failed")
    return {**errors, **different}


def pair_decision_frame_errors(first, second, frame):
    """Same actual sensors after a common hover prefix, not matching actions."""
    if type(frame) is not int or frame < 0:
        raise ValueError("Predeclare a nonnegative common decision frame")
    keys = ("front_rgb", "wrist_rgb", "joint_positions", "gripper_closure", "ee_position", "ee_rotation_6d")
    if any(frame >= len(branch[key]) for branch in (first, second) for key in (*keys, "canonical_actions")):
        raise ValueError("Decision frame is outside a paired trajectory")
    if any(not np.array_equal(first[key][frame], second[key][frame]) for key in keys):
        raise ValueError("Post-hover text-only intervention changed actual sensors")
    if not np.array_equal(first["canonical_actions"][:frame], second["canonical_actions"][:frame]):
        raise ValueError("Post-hover frame lacks a common physical action prefix")
    n = min(5, len(first["canonical_actions"])-frame, len(second["canonical_actions"])-frame)
    xy_separation = float(np.sqrt(np.mean(np.sum((first["canonical_actions"][frame:frame+n, :2] -
                                                 second["canonical_actions"][frame:frame+n, :2])**2, axis=1))))
    if xy_separation < .002:
        raise ValueError("Predeclared post-hover teacher chunks do not discriminate target XY")
    return {"observation_frame": frame, "all_actual_sensors_identical": True,
            "common_physical_prefix_identical": True, "teacher_xy_separation_rms_m": xy_separation}


def main():
    from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts-per-task", type=int, default=2)
    parser.add_argument("--one-layout-per-target", action="store_true",
                        help="Predeclare first ordinary TRAIN layout per picked object/task; paired copies are NOT new scenes")
    parser.add_argument("--gentle-near-touch", action="store_true",
                        help="Expert-only near-surface/5N-boundary protocol for ALL touch branches; no scoring/controller change")
    parser.add_argument("--staged-acquisition", action="store_true",
                        help="Offline teacher only: common vertical hover, horizontal alignment, slower descent")
    parser.add_argument("--decision-frame", type=int, default=0,
                        help="Predeclared same-sensor grounding frame; positive values require staged teacher")
    parser.add_argument("--intervention", choices=("target", "destination"), default="target",
                        help="Destination pairs keep requested object fixed and change only an existing configured goal")
    parser.add_argument("--clearance-valid-destination-pairs", action="store_true",
                        help="Prevalidate all pairs; allow explicit geometry-only replacement of obstructed uncommanded virtual regions")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if args.decision_frame < 0 or (args.decision_frame and not args.staged_acquisition) or (args.staged_acquisition and args.gentle_near_touch):
        raise ValueError("Positive decision frame requires staged acquisition, not an extra gentle-touch proxy")
    if output.exists():
        raise FileExistsError(output)
    audit, parent = verify_source_audit(source, args.source_audit)
    _val_audit, validation = verify_source_audit(args.validation, args.validation_audit)
    if parent.get("split") != "train" or validation.get("split") not in {"val", "validation"}:
        raise ValueError("Only frozen TRAIN collection and VALIDATION exclusion are permitted")
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
    validate_holdout(source, args.validation.resolve())
    tasks = TASKS if args.intervention == "target" else ("move", "pick_place")
    index = json.loads((source / "index.json").read_text())
    selected = (select_target_coverage_rows(index, tasks) if args.one_layout_per_target
                else select_grounding_rows(index, args.layouts_per_task, tasks))
    if args.clearance_valid_destination_pairs and args.intervention != "destination":
        raise ValueError("Clearance-valid destination rule only applies to destination pairs")
    from VLA.valid_destination_pairing import valid_destination_pair, scene_errors
    interventions = []
    for number, original in enumerate(selected):
        task = TaskSpec.from_dict(original["task"])
        if args.clearance_valid_destination_pairs:
            alternate, rule = valid_destination_pair(task)
        else:
            alternate = alternate_target(task, choice=number) if args.intervention == "target" else alternate_destination(task)
            rule = {"rule": "legacy_existing_configured_destination_or_target"}
        for instruction in (task, alternate):
            errors = scene_errors(instruction)
            if errors:
                raise ValueError("Invalid paired instruction BEFORE collection: " + "; ".join(errors))
        interventions.append((alternate, rule))
    output.mkdir(parents=True)
    plan = {"source": str(source), "source_index_sha256": audit["index_sha256"],
            "source_manifest_sha256": audit["manifest_sha256"], "source_payload_fingerprint": audit["source_payload_fingerprint"],
            "selected_source_files": [row["file"] for row in selected],
            "selection_rule": "first_ordinary_layout_per_requested_object_and_task" if args.one_layout_per_target else "first_ordinary_layouts_in_source_order",
            "gentle_near_touch": args.gentle_near_touch,
            "staged_acquisition": args.staged_acquisition, "decision_frame": args.decision_frame,
            "intervention": args.intervention,
            "destination_pair_rules": [{"source_file": row["file"], **rule}
                                       for row, (_, rule) in zip(selected, interventions)],
            "destination_rule_source_sha256": sha256_file(Path(__file__).with_name("valid_destination_pairing.py")),
            "selection_uses_neural_outcomes": False, "final_test_used": False,
            "validation_used_for_collection": False, "controller_changed": False,
            "collection_pybullet_binary_sha256": sha256_file(Path(p.__file__)),
            "collector_source_sha256": sha256_file(Path(__file__))}
    if args.staged_acquisition:
        plan["staged_teacher_sources"] = {str(Path(__file__).with_name(name+".py")): sha256_file(Path(__file__).with_name(name+".py"))
            for name in ("staged_acquisition_teacher", "surface_limited_touch_teacher", "robust_touch_teacher",
                         "geometry_clearance_teacher", "ray_refined_clearance_teacher", "adaptive_clearance_teacher",
                         "neighbour_clearance_teacher", "clearance_correction_teacher")}
    (output / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows, pairs = [], []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        for number, original in enumerate(selected):
            task = TaskSpec.from_dict(original["task"])
            proxy = FrozenRetargetReset(env, source, original)
            teacher = CollisionPlannedTeacher
            if args.staged_acquisition:
                from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher
                teacher = StagedAcquisitionTeacher
            if task.name == "touch" and args.gentle_near_touch:
                from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
                from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher
                proxy = NearSurfaceTouchCollection(env, source, original)
                teacher = SurfaceLimitedTouchTeacher
            pair = []
            alternate, destination_rule = interventions[number]
            for variant, instruction in enumerate((task, alternate)):
                arrays, meta = collect_episode(proxy, instruction, policy_factory=teacher)
                filename = f"{task.name}_layout{number:03d}_variant{variant}.npz"
                meta.update(file=filename, variant=variant, canonical_source=str(source),
                    canonical_source_index_sha256=audit["index_sha256"], paired_source_dataset=str(source),
                    paired_scene_source=original["file"], paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                    recovery_start_actions_not_training_labels=[], candidate_only=True)
                meta["destination_pair_rule"] = destination_rule
                meta["gentle_near_touch"] = bool(task.name == "touch" and args.gentle_near_touch)
                if meta["gentle_near_touch"]:
                    gap, force = proxy.terminal_geometry()
                    meta.update(expert_terminal_surface_gap_m=gap, expert_terminal_max_normal_force_n=force,
                                force_verified_physical_contact_claimed=False)
                np.savez_compressed(output / filename, **arrays)
                rows.append(meta); pair.append(arrays)
                (output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(f"{filename}: success={meta['success']} steps={meta['steps']} safety={meta['safety_violation']}", flush=True)
                if not meta["success"] or meta["safety_violation"] is not None:
                    raise RuntimeError("Preserved failed candidate; do not consume for training")
                if variant == 0:
                    # The source observation must still be reproduced by this
                    # collector, not just by its two retargeted variants.
                    with np.load(source / original["file"], allow_pickle=False) as recorded:
                        from types import SimpleNamespace
                        observed = SimpleNamespace(joint_positions=arrays["joint_positions"][0],
                            front_rgb=arrays["front_rgb"][0], wrist_rgb=arrays["wrist_rgb"][0])
                        source_reset = validate_reset_observation(reset_observation_errors(
                            observed, arrays["gripper_closure"][0], recorded))
            decision = pair_decision_frame_errors(*pair, args.decision_frame) if args.decision_frame else None
            pairs.append({"source_file": original["file"], "source_reset": source_reset, **pair_errors(*pair),
                          "decision_frame_audit": decision})
    verify_source_audit(source, args.source_audit)
    verify_source_audit(args.validation, args.validation_audit)
    verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
    if (sha256_file(Path(__file__)) != plan["collector_source_sha256"] or
            any(sha256_file(path) != digest for path, digest in plan.get("staged_teacher_sources", {}).items())):
        raise RuntimeError("Declared collection implementation changed; candidate remains unqualified")
    manifest = copy.deepcopy(parent)
    for key in ("split", "readiness_report", "readiness_report_sha256", "training_qualification",
                "qualification_parent", "qualification_parent_index_sha256", "qualification_parent_manifest_sha256"):
        manifest.pop(key, None)
    manifest.update(candidate_only=True, training_certified=False, stratified_target_vocabulary=False,
        declared_task_counts=dict(Counter(row["task"]["name"] for row in rows)),
        episodes=len(rows), frames=sum(row["steps"] for row in rows),
        purpose="Paired text-grounding teacher candidate; NOT final data sufficiency or model performance",
        text_intervention=args.intervention,
        grounding_observation_frame=args.decision_frame,
        staged_acquisition=args.staged_acquisition,
        final_test_used=False, index_sha256=sha256_file(output / "index.json"))
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"passed": True, "scope": "Teacher collection/reset pair checks ONLY; NOT camera/native replay qualification",
        "episodes": len(rows), "independent_physical_layouts": len({physical_scene_key(row["task"]) for row in rows}),
        "pairs": pairs, "source": str(source), "source_index_sha256": audit["index_sha256"],
        "index_sha256": manifest["index_sha256"], "source_payload_fingerprint": source_payload_fingerprint(output, rows),
        "environment_runtime_fingerprint": parent["environment_runtime_fingerprint"],
        "paired_variants_are_new_independent_layouts": False, "finetune_performed": False,
        "next_required_gates": ["full camera audit", "native and canonical physics replay", "model conversion/stats/parity"],
        "final_test_used": False}
    (output / "collection_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    from VLA.audit_finetune_data import save_contact_sheets
    grouped = {task: [output / row["file"] for row in rows if row["task"]["name"] == task] for task in tasks}
    save_contact_sheets(output, grouped, output, task_names=tasks)
    print(json.dumps({"passed": True, "episodes": len(rows), "layouts": report["independent_physical_layouts"]}), flush=True)


if __name__ == "__main__":
    main()
