"""TRAIN-only physical wrong-held recovery candidate, NOT qualified data.

Offline expert deliberately acquires a distractor, then records correction
labels for the original instruction. No object teleport, neural failure seed,
deployed fallback, sensor/controller/scoring change or automatic promotion.
Existing closed-empty reset/readiness helpers do NOT support this new role;
it requires separate deterministic replay/qualification before any fitting.
"""
import argparse
from collections import Counter
import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.audit_counterfactual_collection import alternate_target
from VLA.audit_training_readiness import physical_scene_key
from VLA.benchmark import _interaction_audit
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.success_contract_snapshot import recorded_distance_thresholds
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment
from VLA.evaluate_smolvla_v9_source import verify_source_audit
from VLA.probe_wrong_object_recovery import WrongHeldOpeningProbe
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher

TASKS = ("touch", "pick", "move", "pick_place")
ROLE = "physically_bilateral_wrong_held_correction_v1"


class PhysicalWrongHeldReset:
    """Collection-only wrapper; deliberate startup is NEVER an expert label."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def retarget(self, task):
        self.env.task = task
        self.env.pick_id = self.env.object_ids[task.pick_name]
        self.env.place_id = self.env.object_ids.get(task.place_name)
        self.env.initial_place_top = self.env._body_top(self.env.place_id) if self.env.place_id is not None else None

    def reset(self, task):
        self.env.reset(task)
        saved = {name: copy.deepcopy(getattr(self.env, name)) for name in
                 ("initial_pick_z", "initial_pick_position", "initial_pick_top", "initial_pick_bottom")}
        alternate = alternate_target(task)
        self.startup_task = replace(alternate, name="pick", task_id=1, max_steps=160,
                                    prompt=f"Pick the {alternate.pick_name} up")
        self.retarget(self.startup_task)
        teacher = StagedAcquisitionTeacher(self.env)
        teacher.reset(self.startup_task)
        self.startup_actions = []
        for _ in range(self.startup_task.max_steps):
            action = teacher.predict(self.env.observe())
            _, _, done, info = self.env.step(action)
            self.startup_actions.append(np.asarray(action, np.float32).tolist())
            if info.get("safety_violation"):
                raise ValueError("Deliberate wrong-held startup hit an unchanged safety guard")
            held = self.env.gripper.grasped_object
            if held == self.env.pick_id and self.env.gripper._bilateral(held):
                break
            if done:
                raise ValueError("Startup ended before bilateral distractor acquisition")
        else:
            raise ValueError("Startup did not physically acquire a distractor")
        self.retarget(task)
        for name, value in saved.items(): setattr(self.env, name, value)
        initial = _interaction_audit(self.env)
        if (initial["held_object"] != alternate.pick_name or initial["correct_object_held"]
                or np.linalg.norm(np.asarray(initial["object_base_xyz_m"]) - saved["initial_pick_position"]) > .001):
            raise ValueError("Held identity or unchanged requested-target start evidence failed")
        self.initial_evidence = {**initial, "bilateral_wrong_body": True,
            "actual_measured_closure": self.env.measured_gripper_closure(),
            "object_poses_teleported": False, "startup_controls_are_training_labels": False,
            "natural_neural_failure_claimed": False}
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        self.held_before, self.held_after = [], []
        return self.env.observe()

    def step(self, action):
        self.held_before.append(_interaction_audit(self.env)["held_object"] or "")
        output = self.env.step(action)
        self.held_after.append(_interaction_audit(self.env)["held_object"] or "")
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "validation", "validation-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or not 1 <= args.episodes_per_task <= 10:
        raise ValueError("Fresh bounded candidate required")
    _, manifest = verify_source_audit(source, args.source_audit)
    _, val = verify_source_audit(args.validation, args.validation_audit)
    if manifest.get("split") != "train" or val.get("split") not in {"val", "validation"}:
        raise ValueError("Explicit TRAIN and excluded validation required")
    validate_holdout(source, args.validation)
    index = json.loads((source / "index.json").read_text())
    selected = []
    for name in TASKS:
        candidates = [r for r in index if r["task"]["name"] == name and not r.get("paired_scene_source")]
        selected += candidates[:args.episodes_per_task]
    if Counter(r["task"]["name"] for r in selected) != {t: args.episodes_per_task for t in TASKS}:
        raise ValueError("Not enough ordinary TRAIN layouts for the declared pilot")
    excluded = {physical_scene_key(r["task"]) for r in json.loads((args.validation / "index.json").read_text())}
    if any(physical_scene_key(r["task"]) in excluded for r in selected):
        raise ValueError("Validation layout leaked into correction collection")
    fixed = {str(p.resolve()): sha256_file(p) for p in (source / "index.json", source / "dataset_manifest.json",
        args.source_audit, args.validation / "index.json", args.validation_audit, Path(__file__))}
    output.mkdir(parents=True)
    plan = {"scope": "TRAIN_only_wrong_held_recovery_candidate_NOT_finetune_or_learned_success",
        "candidate_only": True, "training_qualified": False, "planned_episodes": len(selected),
        "selected_source_files": [r["file"] for r in selected], "source": str(source),
        "selection_uses_neural_outcomes": False, "final_test_used": False, "development_seeds_used": False,
        "reset_role": ROLE, "source_hashes": fixed, "oracle_deployed_fallback": False,
        "known_required_gates": ["dedicated wrong-held reset replay", "initial held-body feedback contract",
            "first-OPEN and terminal-contact label checks", "held-start-aware readiness/sampling",
            "source/camera/native-input review and full canonical/native physics qualification"]}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    rows = []
    env = audit_environment(source, robot="ur5e", image_size=384, control_hz=10, seed=0,
                            required_success_steps=1, use_semantic_success=False)
    wrapper = PhysicalWrongHeldReset(env)
    try:
        for number, item in enumerate(selected):
            row = {"source_file": item["file"], "task": item["task"], "success": False, "error": None}
            try:
                env.dist_threshold, env.obj_threshold = recorded_distance_thresholds(item)
                arrays, metadata = collect_episode(wrapper, TaskSpec.from_dict(item["task"]),
                                                    policy_factory=WrongHeldOpeningProbe)
                arrays["audit_any_held_object_before"] = np.asarray(wrapper.held_before)
                arrays["audit_any_held_object_after"] = np.asarray(wrapper.held_after)
                if len(arrays["canonical_actions"]) != len(wrapper.held_before):
                    raise ValueError("Held-identity audit arrays are not aligned with labels")
                checks = {"starts_physically_wrong_held": wrapper.initial_evidence["bilateral_wrong_body"],
                    "first_label_is_stationary_OPEN": bool(np.array_equal(arrays["canonical_actions"][0], [0,0,0,0,0,0,-1])),
                    "first_action_releases_wrong_body": arrays["audit_any_held_object_after"][0] == "",
                    "terminal_strict_success": bool(metadata["success"]),
                    "no_safety_abort": metadata.get("safety_violation") is None,
                    "verified_terminal_grasp": item["task"]["name"] == "touch" or bool(arrays["next_verified_grasp"][-1])}
                filename = f"{item['task']['name']}_wrongheld_{number:03d}.npz"
                np.savez_compressed(output / filename, **arrays)
                for camera in ("front", "wrist"):
                    Image.fromarray(arrays[camera+"_rgb"][0]).save(output / f"{Path(filename).stem}_reset_{camera}.png")
                    Image.fromarray(arrays[camera+"_rgb"][-1]).save(output / f"{Path(filename).stem}_last_observation_{camera}.png")
                row.update(metadata, file=filename, recovery_start_kind=ROLE,
                    recovery_start_evidence=wrapper.initial_evidence, correction_checks=checks,
                    startup_task_not_training_instruction=wrapper.startup_task.to_dict(),
                    recovery_start_actions_not_training_labels=wrapper.startup_actions,
                    teacher_success_and_feedback_checks=all(checks.values()), candidate_only=True)
            except Exception as exc:
                row["error"] = str(exc)
                row["failed_startup_actions_not_training_labels"] = getattr(wrapper, "startup_actions", [])
            rows.append(row)
            (output / "index.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({"task": row["task"]["name"], "strict_expert_success": row["success"],
                "error": row["error"], "data_qualified": False}), flush=True)
        if any(sha256_file(Path(p)) != h for p, h in fixed.items()):
            raise ValueError("Bound source evidence changed")
        report = {**plan, "completed_attempts": len(rows), "strict_expert_successes": sum(r["success"] for r in rows),
            "all_correction_feedback_checks_passed": all(r.get("teacher_success_and_feedback_checks") for r in rows),
            "rows": rows, "qualified_for_training": False, "new_learned_trials": 0}
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
