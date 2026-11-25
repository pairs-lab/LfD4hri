"""Predeclared TRAIN-only near-object missed-close recovery candidate.

Uses unchanged guarded robot controls to reach physically closed/unheld starts.
Bad-start controls are reset metadata, NEVER expert action labels. No source,
held-out layout, running fit, controller or success predicate is changed. This
is deliberate contact-free missed-close support, not natural slip/drop support.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pybullet as p

from VLA.audit_counterfactual_collection import alternate_target
from VLA.audit_training_readiness import physical_scene_key
from VLA.collect_demos import collect_episode
from VLA.collect_paired_grounding_candidate import pair_errors, select_target_coverage_rows
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import source_payload_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import TASKS, verify_source_audit
from VLA.near_object_recovery_start import closed_empty_near_object_start, validated_near_object_reset_environment
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
from VLA.surface_limited_touch_teacher import SurfaceLimitedTouchTeacher


def opening_retreat_action(tcp, height, closure):
    """OPEN + vertical retreat only; no diagonal sweep through nearby objects."""
    if tcp[2] >= height - .008 and closure <= .01:
        return None
    return np.asarray([0, 0, np.clip(.65 * (height - tcp[2]), 0, .045),
                       0, 0, 0, -1], dtype=np.float32)


class SafeRecoveryOpeningTeacher:
    """Offline-only corrective prefix before unchanged waypoint expertise.

    Opening while moving diagonally from a near-object miss can fling the
    object. First retreat vertically in the original guarded environment.
    This is a demonstration label, NEVER a learned-policy fallback.
    """
    def __init__(self, env, teacher_type):
        self.env, self.teacher_type = env, teacher_type

    def reset(self, task):
        self.task, self.delegate = task, self.teacher_type(self.env)
        self.opening = True
        self.retreat_height = max(float(self.env.gripper.grasp_center()[2]) + .12,
                                  max(p.getAABB(body)[1][2] for body in self.env.object_ids.values()) + .12)
        self.original_positions = {name: np.asarray(p.getBasePositionAndOrientation(body)[0])
                                   for name, body in self.env.object_ids.items()}

    @property
    def phase(self):
        return "vertical_open_retreat" if self.opening else self.delegate.phase

    def predict(self, observation):
        if self.opening:
            for name, body in self.env.object_ids.items():
                drift = np.linalg.norm(np.asarray(p.getBasePositionAndOrientation(body)[0]) - self.original_positions[name])
                if drift > .001:
                    raise ValueError("Recovery opening moved a source object >1mm; preserve failure, no pushing")
            if p.getContactPoints(bodyA=self.env.gripper.body):
                raise ValueError("Recovery opening made gripper contact; preserve failure")
            action = opening_retreat_action(self.env.gripper.grasp_center(), self.retreat_height,
                                            self.env.measured_gripper_closure())
            if action is not None:
                return action
            self.opening = False
            self.delegate.reset(self.task)
        return self.delegate.predict(observation)


class GentleRecoveryCollection(NearSurfaceTouchCollection):
    def __init__(self, recovery_proxy):
        super().__init__(recovery_proxy.env, None, None)
        self.recovery_proxy = recovery_proxy

    def reset(self, task):
        return self.recovery_proxy.reset(task)


def recovery_checks(arrays, task_name, metadata):
    actions = arrays["canonical_actions"]
    initial_closed = float(arrays["gripper_closure"][0]) >= .95
    first_open = float(actions[0, 6]) == -1 and float(arrays["next_requested_gripper_closure"][0]) == 0
    first_unheld = not bool(arrays["next_grasped_object"][0]) and not bool(arrays["next_verified_grasp"][0])
    ended_held = task_name == "touch" or bool(arrays["next_verified_grasp"][-1] and arrays["next_bilateral_pad_contact"][-1])
    return {"initial_actual_closure_at_least_95pct": initial_closed,
            "first_expert_label_reopens": first_open,
            "no_false_grasp_on_first_action": first_unheld,
            "verified_carry_at_task_end": ended_held,
            "no_safety_abort": metadata.get("safety_violation") is None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "source-audit", "validation", "validation-audit", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--previous-plan", type=Path,
                        help="Bind an expert-only repair to the SAME original metadata-selected layouts")
    parser.add_argument("--settling-aware-startup", action="store_true",
                        help="Explicit matched-time separate-simulator settling audit, old absolute gate remains default")
    parser.add_argument("--staged-yaw-clear-teacher", action="store_true",
                        help="Offline opening retreat followed by the guarded staged/yaw-clear teacher")
    parser.add_argument("--near-surface-terminal-quality", action="store_true",
                        help="Strict raw touch success AND existing <=0.2mm/<=5N label-quality rule; evaluator unchanged")
    parser.add_argument("--bounded-transit", action="store_true",
                        help="Offline travel at max(24cm, all object tops+12cm), not unnecessarily at home height")
    parser.add_argument("--expanded-candidates", action="store_true",
                        help="Original cardinal prefix then <=32 all-object diagonal/tangential proposals")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source_audit, manifest = verify_source_audit(source, args.source_audit)
    _val_audit, validation = verify_source_audit(args.validation, args.validation_audit)
    if manifest.get("split") != "train" or validation.get("split") not in ("val", "validation"):
        raise ValueError("Only original frozen TRAIN selection and VALIDATION exclusion")
    from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
    validate_holdout(source, args.validation.resolve())
    index = json.loads((source / "index.json").read_text())
    selected = select_target_coverage_rows(index)
    from VLA.valid_destination_pairing import scene_errors
    interventions = [alternate_target(TaskSpec.from_dict(row["task"]), choice=number)
                     for number, row in enumerate(selected)]
    for row, alternate in zip(selected, interventions):
        if scene_errors(TaskSpec.from_dict(row["task"])) or scene_errors(alternate):
            raise ValueError("Invalid physical task BEFORE collection")
    output.mkdir(parents=True)
    plan = {"source": str(source), "source_index_sha256": source_audit["index_sha256"],
        "source_manifest_sha256": source_audit["manifest_sha256"],
        "selected_source_files": [r["file"] for r in selected],
        "selection_rule": "first_ordinary_layout_per_requested_object_and_task",
        "selection_uses_neural_outcomes": False, "planned_episodes": 2 * len(selected),
        "intervention": "target", "gentle_near_touch": True,
        "recovery_start_kind": "closed_empty_near_object_missed_close",
        "startup_actions_are_training_labels": False,
        "recovery_teacher_opening": "vertical_OPEN_retreat_before_any_horizontal_approach",
        "source_objects_modified": False, "controller_changed": False,
        "validation_used_for_collection": False, "final_test_used": False,
        "collector_source_sha256": sha256_file(Path(__file__)),
        "startup_protocol_source_sha256": sha256_file(Path(__file__).with_name("near_object_recovery_start.py")),
        "collection_pybullet_binary_sha256": sha256_file(Path(p.__file__))}
    plan.update(bounded_object_clear_transit=args.bounded_transit,
                expanded_offline_proposals=args.expanded_candidates)
    if args.settling_aware_startup:
        plan.update(startup_object_motion_audit="matched_time_no_mount_motion_baseline_on_original_absolute_drift_failure",
            settling_baseline_source_sha256=sha256_file(Path(__file__).with_name("settling_baseline.py")),
            camera_visibility_gates_changed=False, physics_safety_and_success_changed=False)
    if args.staged_yaw_clear_teacher:
        plan.update(gentle_near_touch=False, corrective_teacher="vertical_open_retreat_then_staged_yaw_clear",
            corrective_teacher_sha256=sha256_file(Path(__file__).with_name("yaw_clear_touch_teacher.py")))
    if args.near_surface_terminal_quality:
        plan.update(gentle_near_touch=True, offline_touch_label_quality="strict_raw_AND_gap_le_0.2mm_AND_force_le_5N",
            touch_quality_source_sha256=sha256_file(Path(__file__).with_name("semantic_touch_teacher.py")))
    # Freeze the full offline label-generation closure, not only its outer
    # class. Otherwise changing an ancestor could silently change labels.
    label_sources = ("collect_v9_recovery_candidate", "collect_demos", "near_object_recovery_start",
        "dataset_replay_context", "audit_counterfactual_collection", "collision_grasp_planner",
        "collision_planned_teacher", "grasp_contact_prototype", "semantic_touch_teacher",
        "surface_limited_touch_teacher", "robust_touch_teacher", "geometry_clearance_teacher",
        "ray_refined_clearance_teacher", "adaptive_clearance_teacher", "neighbour_clearance_teacher",
        "clearance_correction_teacher", "staged_acquisition_teacher", "obstacle_aware_touch_teacher",
        "yaw_clear_touch_teacher", "settling_baseline", "near_object_candidate_expansion")
    plan["offline_label_sources_sha256"] = {str(Path(__file__).with_name(name+".py").resolve()):
        sha256_file(Path(__file__).with_name(name+".py")) for name in label_sources}
    if args.previous_plan:
        previous = json.loads(args.previous_plan.read_text())
        for key in ("source", "source_index_sha256", "source_manifest_sha256", "selected_source_files",
                    "planned_episodes", "recovery_start_kind", "validation_used_for_collection", "final_test_used"):
            if previous.get(key) != plan[key]:
                raise ValueError("Repair changed original source/selection: " + key)
        plan["previous_declaration"] = {"path": str(args.previous_plan.resolve()),
            "sha256": sha256_file(args.previous_plan), "same_layouts_and_task_interventions": True}
    (output / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows, pairs = [], []
    state = {"status": "in_progress", "planned_episodes": plan["planned_episodes"],
             "completed_episodes": 0, "training_performed": False, "error": None}
    def save_state():
        (output / "collection_state.json").write_text(json.dumps(state, indent=2) + "\n")
    save_state()
    baseline_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) if args.settling_aware_startup else None
    def settling_reference(task, actions):
        from VLA.settling_baseline import baseline_object_poses
        return baseline_pool.submit(baseline_object_poses,
            (str(source), task.to_dict(), [float(a[6]) for a in actions])).result(timeout=120)
    try:
        with audit_environment(source, image_size=384, required_success_steps=1,
                               success_threshold_scale=1., use_semantic_success=False) as env:
            for number, (row, alternate) in enumerate(zip(selected, interventions)):
                if any(sha256_file(path) != digest for path, digest in plan["offline_label_sources_sha256"].items()):
                    raise RuntimeError("Offline label implementation changed during collection")
                task = TaskSpec.from_dict(row["task"])
                # Pin original historical source thresholds BEFORE startup;
                # subsequent physical resets do not replace this snapshot.
                reset_source_episode(env, source, row)
                startup, evidence = closed_empty_near_object_start(env, task,
                    settling_baseline=settling_reference if baseline_pool else None,
                    bounded_transit=args.bounded_transit, expanded_candidates=args.expanded_candidates)
                (output / f"layout{number:03d}_startup_evidence.json").write_text(
                    json.dumps({"source_file": row["file"], "actions_NOT_labels": [a.tolist() for a in startup],
                                "evidence": evidence}, indent=2) + "\n")
                proxy = validated_near_object_reset_environment(env, task, startup, evidence)
                teacher = CollisionPlannedTeacher
                collection = proxy
                if task.name == "touch":
                    collection = GentleRecoveryCollection(proxy)
                    teacher = SurfaceLimitedTouchTeacher
                if args.staged_yaw_clear_teacher:
                    from VLA.yaw_clear_touch_teacher import YawClearTouchTeacher
                    # Same strict first-hit teacher/data predicate as the main
                    # staged cohort; no NearSurface wrapper predicate change.
                    collection, teacher = proxy, YawClearTouchTeacher
                if task.name == "touch" and args.near_surface_terminal_quality:
                    collection = GentleRecoveryCollection(proxy)
                arrays_pair = []
                for variant, instruction in enumerate((task, alternate)):
                    arrays, meta = collect_episode(collection, instruction,
                        policy_factory=lambda proxy_env: SafeRecoveryOpeningTeacher(proxy_env, teacher))
                    filename = f"{task.name}_layout{number:03d}_variant{variant}.npz"
                    meta.update(file=filename, variant=variant, candidate_only=True,
                        canonical_source=str(source), canonical_source_index_sha256=source_audit["index_sha256"],
                        paired_source_dataset=str(source), paired_scene_source=row["file"],
                        paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                        recovery_start_kind=plan["recovery_start_kind"],
                        recovery_start_actions_not_training_labels=[a.tolist() for a in startup],
                        near_object_recovery_start_evidence=evidence,
                        near_object_recovery_replay_evidence=proxy.near_object_recovery_replay_evidence,
                        gentle_near_touch=task.name == "touch" and (not args.staged_yaw_clear_teacher or args.near_surface_terminal_quality))
                    meta["recovery_expert_after_safe_opening"] = teacher.__module__ + "." + teacher.__name__
                    meta["recovery_teacher_opening"] = plan["recovery_teacher_opening"]
                    checks = recovery_checks(arrays, task.name, meta)
                    meta["near_object_recovery_checks"] = checks
                    if task.name == "touch" and (not args.staged_yaw_clear_teacher or args.near_surface_terminal_quality):
                        gap, force = collection.terminal_geometry()
                        meta.update(expert_terminal_surface_gap_m=gap,
                            expert_terminal_max_normal_force_n=force,
                            force_verified_physical_contact_claimed=False)
                    np.savez_compressed(output / filename, **arrays)
                    rows.append(meta); arrays_pair.append(arrays)
                    (output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
                    state["completed_episodes"] = len(rows); save_state()
                    print(f"{filename}: success={meta['success']} steps={meta['steps']} safety={meta['safety_violation']} checks={all(checks.values())}", flush=True)
                    if not meta["success"] or not all(checks.values()):
                        raise RuntimeError("Failed recovery episode retained; no skipping or partial training")
                pairs.append({"source_file": row["file"], **pair_errors(*arrays_pair)})
        verify_source_audit(source, args.source_audit)
        verify_source_audit(args.validation, args.validation_audit)
        protected_code = {"collect_v9_recovery_candidate.py": plan["collector_source_sha256"],
                          "near_object_recovery_start.py": plan["startup_protocol_source_sha256"]}
        if args.settling_aware_startup:
            protected_code["settling_baseline.py"] = plan["settling_baseline_source_sha256"]
        if args.staged_yaw_clear_teacher:
            protected_code["yaw_clear_touch_teacher.py"] = plan["corrective_teacher_sha256"]
        if any(sha256_file(Path(__file__).with_name(name)) != digest for name, digest in protected_code.items()):
            raise RuntimeError("Declared recovery collector/startup/teacher changed; candidate stays unqualified")
        if any(sha256_file(path) != digest for path, digest in plan["offline_label_sources_sha256"].items()):
            raise RuntimeError("Offline label implementation changed; no qualification")
        candidate = copy.deepcopy(manifest)
        for key in ("split", "readiness_report", "readiness_report_sha256", "training_qualification",
                    "qualification_parent", "qualification_parent_index_sha256", "qualification_parent_manifest_sha256"):
            candidate.pop(key, None)
        candidate.update(candidate_only=True, training_certified=False, stratified_target_vocabulary=False,
            declared_task_counts=dict(Counter(r["task"]["name"] for r in rows)),
            episodes=len(rows), frames=sum(r["steps"] for r in rows),
            purpose="Near-body missed-close recovery candidate; NOT broad natural-failure recovery or learned success",
            recovery_start_kind=plan["recovery_start_kind"], recovery_bad_start_actions_in_training_labels=False,
            near_object_missed_grasp_recovery_certified=False, dropped_or_wrong_object_recovery_certified=False,
            learned_vla_performance_certified=False, final_test_used=False,
            index_sha256=sha256_file(output / "index.json"))
        (output / "dataset_manifest.json").write_text(json.dumps(candidate, indent=2) + "\n")
        report = {"passed": True, "scope": "Offline guarded recovery teacher/reset-pair checks ONLY",
            "episodes": len(rows), "independent_physical_layouts": len({physical_scene_key(r["task"]) for r in rows}),
            "pairs": pairs, "source": str(source), "index_sha256": candidate["index_sha256"],
            "source_payload_fingerprint": source_payload_fingerprint(output, rows),
            "environment_runtime_fingerprint": manifest["environment_runtime_fingerprint"],
            "new_independent_scenes": False, "near_object_recovery_checks_passed": True,
            "recovery_start_kind": plan["recovery_start_kind"], "finetune_performed": False,
            "next_required_gates": ["full source/camera audit", "canonical/native strict replay",
                                    "actual native acquisition image review", "conversion/stats/input parity"]}
        (output / "collection_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        state.update(status="completed_candidate_audit_passed"); save_state()
        print(json.dumps({"episodes": len(rows), "passed": True, "training_performed": False}), flush=True)
    except Exception as exc:
        state.update(status="stopped_error_NOT_training", error=str(exc)); save_state()
        raise
    finally:
        if baseline_pool:
            baseline_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
