"""40 predeclared unique TRAIN student prefixes -> OFFLINE correction labels.

Student: actual learned SmolVLA+state18. Scene64 is only measured/recorded,
NOT silently fed to an unfitted model. Prefixes20/40/60 cover later missed
grasps. Every failed/early-success/safety/held-start case is retained. The
expert is used only to produce training labels, NEVER to replace a VLA in
success-rate evaluation. Candidates cannot be trained until replay gates.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.audit_training_readiness import physical_scene_key
from VLA.benchmark import _interaction_audit
from VLA.collect_demos import collect_episode
from VLA.collect_onpolicy_handoff_pilot import QualityPrefixTouch
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit, reset_observation_errors, validate_reset_observation
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.simstate_scene64 import SCHEMA, scene_state
from VLA.smolvla.simstate_adapter import SimStateSmolVLA
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher

TASKS = ("touch", "pick", "move", "pick_place")


def select_train_cases(rows, excluded, per_task=10, seed=20261415):
    rng = np.random.default_rng(seed); selected = []; seen = set()
    for task in TASKS:
        candidates = [r for r in rows if r["task"]["name"] == task and not r.get("paired_scene_source")
            and not r.get("recovery_start_actions_not_training_labels") and r.get("success")]
        names = sorted({r["task"]["pick_name"] for r in candidates}); rng.shuffle(names)
        choices = []
        for name in names:
            options = [r for r in candidates if r["task"]["pick_name"] == name]
            rng.shuffle(options)
            for row in options:
                key = physical_scene_key(row["task"])
                if key in excluded or key in seen: continue
                seen.add(key); choices.append({"row": row, "prefix_controls": (20, 40, 60)[len(choices) % 3]})
                break
            if len(choices) == per_task: break
        if len(choices) != per_task: raise ValueError("Not enough predeclared unique TRAIN target/scene cases")
        selected.extend(choices)
    return selected


class PrefixCorrectionCapture:
    def __init__(self, env, policy, source, row, controls, folder):
        self.env, self.policy, self.source, self.row = env, policy, source, row
        self.controls, self.folder = controls, folder
        self.prefix_trace = []; self.causal_states = []; self.held_before = []; self.held_after = []

    def __getattr__(self, name): return getattr(self.env, name)

    def reset(self, task):
        if task != TaskSpec.from_dict(self.row["task"]): raise ValueError("Original instruction must not change")
        obs = reset_source_episode(self.env, self.source, self.row)
        with np.load(self.source / self.row["file"], allow_pickle=False) as reference:
            validate_reset_observation(reset_observation_errors(obs, self.env.normalized_gripper_closure(),
                {k: reference[k][:1] for k in ("joint_positions", "gripper_closure", "front_rgb", "wrist_rgb")}))
        self.policy.reset(task)
        for count in range(self.controls):
            before = scene_state(self.env, obs)
            if count % 10 == 0:
                for camera in ("front", "wrist"): Image.fromarray(getattr(obs, camera + "_rgb")).save(self.folder / f"prefix{count:03d}_{camera}.png")
            action = self.policy.predict(obs)
            obs, _, done, info = self.env.step(action)
            self.prefix_trace.append({"step": count + 1, "pre_action_scene64": before.tolist(),
                "policy_requested_canonical": np.asarray(action, np.float32).tolist(),
                "executed_canonical": info["executed_canonical_action"],
                "native_requested_arm_targets": np.asarray(self.env.requested_arm_targets).tolist(),
                "interaction_after": _interaction_audit(self.env), "strict_success": bool(info["strict_raw_success"]),
                "safety_violation": info.get("safety_violation")})
            atomic_json(self.folder / "student_prefix.json", self.prefix_trace)
            if done or info.get("safety_violation"):
                raise ValueError("Predeclared prefix terminated: keep actual success/abort, do not replace its scene")
        self.initial = _interaction_audit(self.env)
        self.initial_scene64 = scene_state(self.env, obs).tolist()
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        if self.env.is_strict_success(): raise ValueError("No initially completed correction episode")
        for camera in ("front", "wrist"): Image.fromarray(getattr(obs, camera + "_rgb")).save(self.folder / f"handoff_{camera}.png")
        return obs

    def step(self, action):
        self.causal_states.append(scene_state(self.env, self.env.observe()))
        self.held_before.append(_interaction_audit(self.env)["held_object"] or "")
        result = self.env.step(action)
        self.held_after.append(_interaction_audit(self.env)["held_object"] or "")
        return result


class OfflineRecoveryTeacher(StagedAcquisitionTeacher):
    """Supervised-label producer ONLY, not an inference-time recovery policy."""
    def reset(self, task):
        self.task = task
        self.release_wrong = self.env.gripper.grasped_object not in (None, self.env.pick_id)
        # Do not plan a requested-target grasp while a different object is
        # attached. Release first, then initialize from the ACTUAL new scene.
        self.ready = False
        if not self.release_wrong: self._initialize(task)

    def _initialize(self, task):
        super().reset(task)
        self.ready = True
        if self.env.gripper.grasped_object == self.env.pick_id:
            self._advance("lift" if self.env.gripper._bilateral(self.env.pick_id) else "close")

    def predict(self, observation):
        if self.release_wrong:
            if self.env.gripper.grasped_object is not None:
                return np.asarray([0, 0, 0, 0, 0, 0, -1], np.float32)
            self.release_wrong = False
        if not self.ready: self._initialize(self.task)
        return super().predict(observation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "validation", "validation-audit", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args(); source, output = args.source.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError("Keep all attempted corrections, no reroll")
    _, manifest = verify_source_audit(source, args.source_audit)
    _, val = verify_source_audit(args.validation, args.validation_audit)
    if manifest["split"] != "train" or val["split"] not in ("val", "validation"): raise ValueError("TRAIN only")
    validate_holdout(source, args.validation)
    identity = verify_fingerprint(args.checkpoint)
    rows = json.loads((source / "index.json").read_text())
    excluded = {physical_scene_key(r["task"]) for r in json.loads((args.validation / "index.json").read_text())}
    cases = select_train_cases(rows, excluded)
    output.mkdir(parents=True)
    frozen = {str(path.resolve()): sha256_file(path) for path in (
        source / "index.json", source / "dataset_manifest.json", args.source_audit, args.validation / "index.json",
        args.validation_audit, Path(__file__), Path(__file__).with_name("simstate_scene64.py"),
        Path(__file__).with_name("staged_acquisition_teacher.py"), Path(__file__).with_name("collect_demos.py"))}
    plan = {"scope": "TRAIN_only_40_onpolicy_recovery_candidates_NOT_model_SR", "source": str(source),
        "selected_files": [c["row"]["file"] for c in cases], "cases": [{"source_file": c["row"]["file"], "task": c["row"]["task"]["name"],
            "pick_name": c["row"]["task"]["pick_name"], "prefix_controls": c["prefix_controls"]} for c in cases],
        "schema": SCHEMA, "planned_attempts": 40, "candidate_only": True, "training_qualified": False,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
        "selection_uses_neural_outcomes": False, "validation_dev_final_used_for_training": False,
        "student_inputs_state18_NOT_unfitted64": True, "expert_is_offline_labels_ONLY": True,
        "expert_is_never_evaluation_fallback": True, "prefix_actions_are_not_teacher_labels": True,
        "source_hashes": frozen, "student_noise_seed": 20261315, "pid": os.getpid()}
    atomic_json(output / "plan.json", plan); results = []
    def save(status, error=None):
        atomic_json(output / "progress.json", {"status": status, "completed_attempts": len(results), "planned_attempts": 40,
            "expert_correction_successes_NOT_student_SR": sum(r.get("success", False) for r in results),
            "labeled_frames": sum(r.get("steps", 0) for r in results if r.get("file")),
            "candidate_only": True, "training_qualified": False, "error": error})
    save("starting_bounded_actual_VLA_prefix_collection")
    import torch
    torch.cuda.set_per_process_memory_fraction(3000 * 2**20 / torch.cuda.get_device_properties(0).total_memory)
    env = audit_environment(source, image_size=384, control_hz=10, seed=0,
        required_success_steps=1, use_semantic_success=False)
    policy = SimStateSmolVLA(env, checkpoint=str(args.checkpoint.resolve()), replan_steps=1, seed=20261315)
    try:
        for number, case in enumerate(cases):
            row = case["row"]; task = TaskSpec.from_dict(row["task"])
            folder = output / f"case{number:03d}_{task.name}"; folder.mkdir()
            prefix = PrefixCorrectionCapture(env, policy, source, row, case["prefix_controls"], folder)
            wrapper = QualityPrefixTouch(prefix, None, None) if task.name == "touch" else prefix
            result = {"source_file": row["file"], "task": row["task"], "prefix_controls": case["prefix_controls"],
                "success": False, "error": None, "candidate_only": True, "correction_labels_only": True}
            try:
                arrays, metadata = collect_episode(wrapper, task, policy_factory=OfflineRecoveryTeacher)
                features = np.asarray(prefix.causal_states, np.float32)
                if features.shape != (len(arrays["canonical_actions"]), 64): raise ValueError("Before-label scene64 timeline differs")
                arrays.update(audit_pre_action_PRIVILEGED_state64=features,
                    audit_pre_action_PRIVILEGED_state18=features[:, :18].copy(),
                    audit_any_held_object_before=np.asarray(prefix.held_before), audit_any_held_object_after=np.asarray(prefix.held_after))
                filename = f"{task.name}_onpolicy_{number:03d}.npz"; np.savez_compressed(output / filename, **arrays)
                for camera in ("front", "wrist"): Image.fromarray(env.render(camera)).save(folder / f"terminal_{camera}.png")
                result.update(metadata, file=filename, recovery_start_kind="student_predeclared_late_handoff_scene64_v1",
                    recovery_start_evidence=prefix.initial, initial_scene64=prefix.initial_scene64,
                    recovery_start_actions_not_training_labels=[r["executed_canonical"] for r in prefix.prefix_trace],
                    prefix_native_commands_audit_only=[r["native_requested_arm_targets"] for r in prefix.prefix_trace],
                    new_model_input_schema=SCHEMA, student_uses_privileged_state18=True)
                result["payload_sha256"] = sha256_file(output / filename)
                if task.name == "touch": result["near_surface_quality_gap_and_force"] = wrapper.terminal_geometry()
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
                result["failed_prefix_controls_preserved"] = prefix.prefix_trace
            results.append(result); atomic_json(output / "index.json", results); save("collecting_PREDECLARED_train_corrections")
            print(json.dumps({"attempt": number + 1, "task": task.name, "prefix": case["prefix_controls"],
                "expert_correction_success_NOT_student_SR": result["success"], "error": result["error"]}), flush=True)
        if any(sha256_file(path) != digest for path, digest in frozen.items()): raise ValueError("Frozen source changed")
        verify_fingerprint(args.checkpoint)
        atomic_json(output / "audit.json", {**plan, "completed_attempts": len(results), "rows": results,
            "remaining_gates": ["Replay ALL successful canonical and native corrections, preserve failed attempts",
                "Check current state64 before EACH label and exact source18 prefix", "Actual model tensor/CUDA/reload qualification",
                "Dedicated onpolicy-start training readiness role; never inject candidate-only data into a live fit"]})
        save("completed40_attempts_NOT_training_qualified")
    except Exception as error:
        save("failed_preserve_all_actual_actions", f"{type(error).__name__}: {error}"); raise
    finally:
        policy.close(); env.close()


if __name__ == "__main__": main()
