"""Four preselected TRAIN-only student prefixes -> physical expert corrections.

Candidate only, never deployed fallback or an automatic training promotion.
The student runs20 controls from its exact source reset; only subsequent
teacher corrections are supervised labels. All attempted failures retained.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.audit_training_readiness import physical_scene_key
from VLA.benchmark import _interaction_audit
from VLA.collect_demos import collect_episode
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.semantic_touch_teacher import NearSurfaceTouchCollection
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.smolvla.audit_checkpoint_predictions import validate_holdout
from VLA.staged_acquisition_teacher import StagedAcquisitionTeacher


class StudentPrefixReset:
    def __init__(self, env, policy, source, row, controls):
        self.env, self.policy, self.source, self.row, self.controls = env, policy, source, row, controls

    def __getattr__(self, name): return getattr(self.env, name)

    def reset(self, task):
        if task != TaskSpec.from_dict(self.row["task"]):
            raise ValueError("Correction instruction must remain unchanged")
        obs = reset_source_episode(self.env, self.source, self.row)
        self.policy.reset(task)
        self.actions, self.native_actions = [], []
        for _ in range(self.controls):
            action = self.policy.predict(obs)
            obs, _, done, info = self.env.step(action)
            self.actions.append(info["executed_canonical_action"])
            self.native_actions.append(np.asarray(self.env.requested_arm_targets).tolist())
            if done or info.get("safety_violation"):
                raise ValueError("Predeclared student prefix terminated/aborted; preserve without replacement")
        self.initial = _interaction_audit(self.env)
        if self.initial["held_object"] is not None:
            raise ValueError("This handoff pilot supports NOT-held starts only")
        self.env.step_count, self.env.success_streak, self.env._last_success = 0, 0, False
        self.held_before, self.held_after = [], []
        return obs

    def step(self, action):
        self.held_before.append(_interaction_audit(self.env)["held_object"] or "")
        output = self.env.step(action)
        self.held_after.append(_interaction_audit(self.env)["held_object"] or "")
        return output


class QualityPrefixTouch(NearSurfaceTouchCollection):
    def reset(self, task): return self.env.reset(task)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-audit", "validation", "validation-audit", "checkpoint", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists(): raise FileExistsError("Preserve every candidate")
    audit, manifest = verify_source_audit(source, args.source_audit)
    _, val = verify_source_audit(args.validation, args.validation_audit)
    if manifest.get("split") != "train" or val.get("split") not in {"val", "validation"}:
        raise ValueError("Explicit TRAIN and excluded validation required")
    validate_holdout(source, args.validation)
    identity = verify_fingerprint(args.checkpoint)
    index = json.loads((source / "index.json").read_text())
    selected = [next(row for row in index if row["task"]["name"] == task and not row.get("paired_scene_source")
        and not row.get("recovery_start_actions_not_training_labels")) for task in ("touch", "pick", "move", "pick_place")]
    excluded = {physical_scene_key(row["task"]) for row in json.loads((args.validation / "index.json").read_text())}
    if any(physical_scene_key(row["task"]) in excluded for row in selected): raise ValueError("Validation leakage")
    hashes = {str(p.resolve()): sha256_file(p) for p in (source / "index.json", source / "dataset_manifest.json", args.source_audit,
        args.validation / "index.json", args.validation_audit, Path(__file__))}
    output.mkdir(parents=True)
    plan = {"scope": "TRAIN_only_onpolicy_handoff_correction_pilot_NOT_qualified", "selected_files": [r["file"] for r in selected],
        "prefix_controls": 20, "planned_episodes": 4, "source": str(source), "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_content_sha256": identity["content_tree_sha256"], "source_hashes": hashes, "student_noise_seed": 20261315,
        "selection_uses_neural_outcomes": False, "development_or_final_test_seeds_used": False, "candidate_only": True,
        "training_qualified": False, "oracle_deployed_fallback": False, "student_prefix_actions_are_teacher_labels": False}
    (output / "plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    env = audit_environment(source, robot="ur5e", image_size=384, control_hz=10, seed=0,
        required_success_steps=1, use_semantic_success=False)
    policy = SmolVLAPolicyAdapter(env, checkpoint=str(args.checkpoint.resolve()), replan_steps=1, seed=20261315)
    rows = []
    try:
        for number, row in enumerate(selected):
            prefix = StudentPrefixReset(env, policy, source, row, 20)
            wrapper = QualityPrefixTouch(prefix, None, None) if row["task"]["name"] == "touch" else prefix
            result = {"source_file": row["file"], "task": row["task"], "success": False, "error": None}
            try:
                arrays, metadata = collect_episode(wrapper, TaskSpec.from_dict(row["task"]), policy_factory=StagedAcquisitionTeacher)
                arrays["audit_any_held_object_before"] = np.asarray(prefix.held_before)
                arrays["audit_any_held_object_after"] = np.asarray(prefix.held_after)
                file = f"{row['task']['name']}_onpolicy_{number:03d}.npz"
                np.savez_compressed(output / file, **arrays)
                for camera in ("front", "wrist"):
                    Image.fromarray(arrays[camera+"_rgb"][0]).save(output / (Path(file).stem+"_handoff_"+camera+".png"))
                    Image.fromarray(env.render(camera)).save(output / (Path(file).stem+"_terminal_"+camera+".png"))
                result.update(metadata, file=file, candidate_only=True, recovery_start_kind="student_20_step_open_handoff_v1",
                    recovery_start_evidence=prefix.initial, recovery_start_actions_not_training_labels=prefix.actions,
                    prefix_native_commands_audit_only=prefix.native_actions,
                    correction_labels_only=True, initial_false_grasp=False)
                if row["task"]["name"] == "touch":
                    result["near_surface_quality_gap_and_force"] = wrapper.terminal_geometry()
            except Exception as exc:
                result["error"] = str(exc)
                result["failed_prefix_controls_preserved"] = getattr(prefix, "actions", [])
            rows.append(result)
            (output / "index.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(json.dumps({"task": row["task"]["name"], "expert_correction_success": result["success"], "error": result["error"], "qualified": False}), flush=True)
        if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()): raise ValueError("Source changed")
        verify_fingerprint(args.checkpoint)
        (output / "audit.json").write_text(json.dumps({**plan, "completed_attempts": len(rows),
            "expert_correction_successes": sum(row["success"] for row in rows), "rows": rows,
            "remaining_gates": ["deterministic physical student-prefix replay", "sensor/feedback/action/source audit",
                "dedicated onpolicy-start readiness role", "full canonical/native replay", "native camera and model input review"]}, indent=2)+"\n")
    finally:
        policy.close(); env.close()


if __name__ == "__main__":
    main()
