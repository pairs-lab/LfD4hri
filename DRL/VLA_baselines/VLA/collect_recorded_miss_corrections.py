"""Offline TRAIN-only expert corrections from actual recorded safe empty closes.

Recorded neural prefixes recreate states ONLY, never become expert labels.
No validation failures, model fallback, relaxed safeguards or benchmark controls.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.audit_recorded_neural_action_replay import action_binding, sensor
from VLA.audit_training_readiness import source_reset_provenance
from VLA.audit_counterfactual_collection import alternate_target, physical_scene_spec
from VLA.collect_demos import collect_episode
from VLA.collision_planned_teacher import CollisionPlannedTeacher
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import verify_environment_runtime_fingerprint
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import safety_abort_info, verify_source_audit


def eligible(frame):
    actual = frame["after"]
    return (actual["closed_empty_proxy"] and actual["surface_clearance_m"] >= .008
            and not actual["gripper_contacts"] and not frame["safety_abort"])


def verify_recreated(actual, expected):
    for key in ("joint_positions_rad", "mount_position_m", "tcp_position_m", "target_position_m",
                "measured_encoder_fk_closure", "jaw_gap_m"):
        if not np.allclose(actual[key], expected[key], atol=1e-6, rtol=0):
            raise ValueError("Recorded off-policy state mismatch: " + key)
    if not (actual["closed_empty_proxy"] and actual["surface_clearance_m"] >= .008
            and not actual["gripper_contacts"] and actual["held_body_id"] is None):
        raise ValueError("Recreated state is not physically safe closed-empty")


class RecordedPrefixReset:
    def __init__(self, env, source, item, prefix, expected):
        self.env, self.source, self.item = env, source, item
        self.prefix, self.expected = prefix, expected
        self.reset_evidence = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, task):
        original = TaskSpec.from_dict(self.item["task"])
        if ((task.name, task.task_id) != (original.name, original.task_id)
                or physical_scene_spec(task) != physical_scene_spec(original)):
            raise ValueError("Correction must retain original primitive and every physical body")
        reset_source_episode(self.env, self.source, self.item)
        for frame in self.prefix:
            _, _, done, info = self.env.step(np.asarray(frame["action"], dtype=np.float32))
            aborted, _ = safety_abort_info(info)
            if done or aborted or info.get("grasped_object") is not None:
                raise RuntimeError("Recorded prefix terminated, grasped or became unsafe")
            if not np.array_equal(np.asarray(info["executed_canonical_action"], dtype=np.float32),
                                  np.asarray(frame["executed_action"], dtype=np.float32)):
                raise ValueError("Recorded safety clamp/action differs")
        actual = sensor(self.env)
        verify_recreated(actual, self.expected)
        # Text intervention changes only the teacher's requested target AFTER
        # exact original-state verification; controller still sees ALL bodies.
        if self.env.observation_profile.get("grasp_candidate_scope") != "all_dynamic_scene_objects":
            raise ValueError("Retargeting cannot change controller grasp scope")
        self.env.task = task
        self.env.pick_id = self.env.object_ids[task.pick_name]
        self.env.place_id = self.env.object_ids.get(task.place_name)
        self.env.initial_place_top = self.env._body_top(self.env.place_id) if self.env.place_id is not None else None
        self.reset_evidence = sensor(self.env)
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        position = p.getBasePositionAndOrientation(self.env.pick_id)[0]
        self.env.initial_pick_z = position[2]
        self.env.initial_pick_position = np.asarray(position, dtype=np.float64)
        self.env.initial_pick_top, self.env.initial_pick_bottom = self.env._pick_top(), self.env._pick_bottom()
        return self.env.observe()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--feedback-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-targets", action="store_true", help="Two same-state/different-text corrections per recorded miss")
    parser.add_argument("--clearance-aware-teacher", action="store_true", help="Expert-waypoint ablation for preserved table-blocked empty close; never changes V9 runtime")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    audit, parent = verify_source_audit(source, args.source_audit)
    replay = json.loads(args.feedback_replay.read_text())
    progress_path = Path(replay["progress"])
    if parent.get("split") != "train" or replay.get("passed") is not True or Path(replay["source"]).resolve() != source:
        raise ValueError("Only verified recorded-action replays from TRAIN are allowed")
    if sha256_file(progress_path) != replay["progress_sha256_at_snapshot"]:
        raise ValueError("Recorded rollout changed")
    if sha256_file(Path(p.__file__)) != replay["replay_pybullet_binary"]["sha256"]:
        raise ValueError("Use actual deployment PyBullet binary for dynamic state replay")
    rows = json.loads((source / "index.json").read_text())
    source_by_file = {row["file"]: row for row in rows}
    original_by_file = {row["file"]: row for row in json.loads(progress_path.read_text())}
    selections = []
    for row in replay["rows"]:
        original = original_by_file[row["file"]]
        if action_binding(original) != row["action_trace_sha256"]:
            raise ValueError("Trace binding changed")
        matches = [frame for frame in row["feedback"] if eligible(frame)]
        if matches:
            selections.append({"file": row["file"], "prefix_steps": matches[0]["step"],
                               "expected_state": matches[0]["after"]})
    if not selections or output.exists():
        raise ValueError("Need eligible recorded states and a NEW output directory")
    output.mkdir(parents=True)
    policy_factory = CollisionPlannedTeacher
    if args.clearance_aware_teacher:
        from VLA.clearance_correction_teacher import ClearanceCorrectionTeacher
        policy_factory = ClearanceCorrectionTeacher
    teacher_path = Path(__file__).with_name("clearance_correction_teacher.py" if args.clearance_aware_teacher else "collision_planned_teacher.py")
    plan = {"scope": "Adaptive TRAIN expert corrections, NOT new independent scenes or neural results",
            "source": str(source), "source_index_sha256": audit["index_sha256"],
            "source_manifest_sha256": audit["manifest_sha256"],
            "feedback_replay_sha256": sha256_file(args.feedback_replay),
            "progress_sha256": sha256_file(progress_path), "selections": selections,
            "selection": "first physically closed-empty, contact-free state with >=8mm clearance per TRAIN rollout",
            "script_sha256": sha256_file(Path(__file__)),
            "pybullet_binary_sha256": sha256_file(Path(p.__file__)),
            "paired_targets": args.paired_targets,
            "clearance_aware_expert_waypoints": args.clearance_aware_teacher,
            "teacher_source_sha256": sha256_file(teacher_path),
            "validation_or_test_failures_used": False, "prefix_controls_are_training_labels": False}
    (output / "collection_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    manifest = copy.deepcopy(parent)
    for key in ("split", "readiness_report", "readiness_report_sha256", "training_qualification", "qualification_parent",
                "qualification_parent_index_sha256", "qualification_parent_manifest_sha256"):
        manifest.pop(key, None)
    manifest.update(candidate_only=True, training_certified=False, stratified_target_vocabulary=False,
                    final_test_used=False, purpose=plan["scope"], collection_plan_sha256=sha256_file(output / "collection_plan.json"))
    collected = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=False) as env:
        jobs = [(selection, variant) for selection in selections for variant in range(2 if args.paired_targets else 1)]
        first_variants = {}
        for selection, variant in jobs:
            item = source_by_file[selection["file"]]
            prefix = original_by_file[item["file"]]["trace"][:selection["prefix_steps"]]
            proxy = RecordedPrefixReset(env, source, item, prefix, selection["expected_state"])
            task = TaskSpec.from_dict(item["task"])
            if variant:
                task = alternate_target(task)
            arrays, meta = collect_episode(proxy, task, policy_factory=policy_factory)
            filename = "corrected_" + Path(item["file"]).stem + (f"_variant{variant}" if args.paired_targets else "") + ".npz"
            np.savez_compressed(output / filename, **arrays)
            provenance = source_reset_provenance(source, item)
            startup = list(item.get("recovery_start_actions_not_training_labels", ())) + [f["action"] for f in prefix]
            meta.update(file=filename, candidate_only=True, variant=variant,
                paired_reset_strategy="canonical_spawn_order_then_teacher_retarget",
                paired_scene_source=provenance.get("paired_scene_source", item["file"]),
                canonical_source=provenance.get("canonical_source", str(source)),
                canonical_source_index_sha256=provenance.get("canonical_source_index_sha256", audit["index_sha256"]),
                paired_collection_source=str(output), recovery_start_actions_not_training_labels=startup,
                recovery_start_kind="recorded_neural_train_closed_empty",
                recorded_train_rollout=item["file"], recorded_prefix_steps=len(prefix),
                correction_initial_state=proxy.reset_evidence)
            collected.append(meta)
            (output / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
            manifest.update(episodes=len(collected), frames=sum(row["steps"] for row in collected))
            (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            safe = meta["success"] and meta.get("safety_violation") is None
            (output / "collection_state.json").write_text(json.dumps({"status": "in_progress" if safe else "stopped_on_failure",
                "completed_episodes": len(collected), "planned_episodes": len(jobs)}, indent=2) + "\n")
            print(f"{filename}: success={meta['success']} steps={meta['steps']} safety={meta.get('safety_violation')}", flush=True)
            if not safe:
                raise RuntimeError("Preserved correction failed; diagnose without deleting/filtering")
            if arrays["gripper_closure"][0] < .95 or arrays["canonical_actions"][0, 6] >= 0:
                raise RuntimeError("Correction must start physically closed-empty and reopen")
            if args.paired_targets:
                if variant == 0:
                    first_variants[item["file"]] = arrays
                else:
                    from VLA.collect_paired_grounding_candidate import pair_errors
                    meta["same_state_text_pair_checks"] = pair_errors(first_variants.pop(item["file"]), arrays)
                    (output / "index.json").write_text(json.dumps(collected, indent=2) + "\n")
    verify_source_audit(source, args.source_audit)
    verify_environment_runtime_fingerprint(parent["environment_runtime_fingerprint"], parent["observation_profile"])
    if (sha256_file(Path(__file__)) != plan["script_sha256"]
            or sha256_file(teacher_path) != plan["teacher_source_sha256"]
            or sha256_file(args.feedback_replay) != plan["feedback_replay_sha256"]
            or sha256_file(progress_path) != plan["progress_sha256"]):
        raise RuntimeError("Correction implementation or bound recorded evidence changed")
    (output / "collection_state.json").write_text(json.dumps({"status": "completed_candidate_not_qualified",
        "completed_episodes": len(collected), "planned_episodes": len(jobs)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
