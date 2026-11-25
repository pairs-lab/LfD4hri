"""Train/held-out imitation and fixed-noise SimState input interventions.

Offline diagnostic only: never moves the robot or changes training/evaluation.
Counterfactual inputs measure sensitivity, NOT valid-scene task success.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.cross_vla120_trial import atomic_json
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.simstate_feedback import SCHEMA
from VLA.smolvla.adapter import SmolVLAPolicyAdapter
from VLA.smolvla.audit_checkpoint_predictions import summarize, validate_holdout
from VLA.smolvla.simstate_adapter import SimStateSmolVLA


class SourceStatePolicy(SimStateSmolVLA):
    """Replay the audited BEFORE-action state through the deployed processors."""
    def build_sensor_batch(self, observation):
        import torch
        batch = SmolVLAPolicyAdapter.build_sensor_batch(self, observation)
        batch["observation.state"] = torch.from_numpy(self.source_state.copy()).unsqueeze(0)
        return batch


def projection_metadata(root, split):
    proof = json.loads((root/"audit.json").read_text())
    if (not proof.get("passed") or proof.get("schema") != SCHEMA or proof.get("split") != split
            or proof.get("pilot_only") or proof.get("future_state_in_model_input") is not False
            or proof.get("teacher_actions_in_model_input") is not False
            or proof.get("images_or_action_labels_modified") is not False):
        raise ValueError("Complete causal projection proof required")
    source = Path(proof["source"])
    for path, field in ((source/"index.json", "source_index_sha256"),
                        (source/"dataset_manifest.json", "source_manifest_sha256"),
                        (root/"stats.json", "stats_sha256")):
        if sha256_file(path) != proof[field]:
            raise ValueError("Projection metadata changed: " + str(path))
    return source, proof, json.loads((source/"index.json").read_text())


def saved_normalization(checkpoint, projection):
    from safetensors import safe_open
    expected = json.loads((projection/"stats.json").read_text())
    found = 0
    for filename in checkpoint.glob("policy_*normalizer_processor.safetensors"):
        with safe_open(filename, framework="np") as saved:
            for stat in ("mean", "std", "min", "max"):
                key = "observation.state." + stat
                if key not in saved.keys():
                    continue
                actual = saved.get_tensor(key)
                if not np.array_equal(actual, np.asarray(expected[stat], dtype=np.float32)):
                    raise ValueError("Saved state normalizer is not the declared TRAIN18: " + str(filename))
                found += 1
    if found < 4:
        raise ValueError("No complete saved TRAIN18 normalization")
    return {"passed": True, "exact_saved_state_statistics_checked": found,
            "validation_statistics_used": False, "train_stats_sha256": sha256_file(projection/"stats.json")}


def diagnostic_frames(actions):
    """Include actual XY decisions; early vertical descent is not grounding."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite canonical action timeline required")
    selected = np.rint(np.linspace(0, len(values)-1, 4)).astype(int).tolist()
    xy = np.linalg.norm(values[:, :2], axis=1)
    moving = np.flatnonzero(xy > .002)
    if len(moving):
        selected.extend((int(moving[0]), int(xy.argmax())))
    return np.unique(selected)


def run(args):
    import torch
    if args.output.exists():
        raise FileExistsError("Preserve every diagnostic; no overwrite")
    args.output.mkdir(parents=True)
    train, train_proof, train_index = projection_metadata(args.train_projection, "train")
    val, val_proof, val_index = projection_metadata(args.validation_projection, "validation")
    validate_holdout(train, val)
    identity = verify_fingerprint(args.checkpoint, full=True)
    contract = json.loads((args.checkpoint/"simstate_contract.json").read_text())
    if contract["projection_audit_sha256"] != sha256_file(args.train_projection/"audit.json"):
        raise ValueError("Checkpoint belongs to a different training projection")
    normalization = saved_normalization(args.checkpoint, args.train_projection)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_mib*1024**2/total, device=0)
    sensor = SimpleNamespace(closure=0.)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SourceStatePolicy(sensor, str(args.checkpoint), seed=20261014, replan_steps=1)
    rows, interventions = [], []
    mean = np.asarray(json.loads((args.train_projection/"stats.json").read_text())["mean"], dtype=np.float32)
    plan = {"checkpoint": str(args.checkpoint), "checkpoint_content_sha256": identity["content_tree_sha256"],
            "measurement": "train_and_disjoint_validation_imitation_NOT_success_rate",
            "privileged_model_inputs": True, "robot_actions_executed": 0,
            "source_images_and_actions_unchanged": True, "normalization": normalization,
            "episodes_per_task_per_split": args.episodes_per_task,
            "frames_per_episode": "four evenly spaced plus first/max teacher XY motion >2mm; unique indices",
            "interventions": "validation first frame AND teacher XY-moving frames: neutralize target, goal or held inputs to TRAIN mean",
            "same_prompt_RGB_original7_and_noise_across_interventions": True,
            "interventions_NOT_valid_scene_success_trials": True, "final_test_used": False,
            "selection_uses_model_predictions_or_outcomes": False}
    atomic_json(args.output/"plan.json", plan)
    try:
        for split, root, source, proof, index in (("train", args.train_projection, train, train_proof, train_index),
                                                  ("validation", args.validation_projection, val, val_proof, val_index)):
            by_task = defaultdict(list)
            for item in index:
                by_task[item["task"]["name"]].append(item)
            payloads = {item["file"]: item for item in proof["rows"]}
            for task_name in ("touch", "pick", "move", "pick_place"):
                items = by_task[task_name]
                if not items:
                    raise ValueError("Missing task: " + task_name)
                selected = np.linspace(0, len(items)-1, min(args.episodes_per_task, len(items)), dtype=int)
                for number in selected:
                    item = items[int(number)]; name = item["file"]; payload = payloads[name]
                    state_path = root/"states"/name
                    if (sha256_file(source/name) != payload["source_payload_sha256"] or
                            sha256_file(state_path) != payload["state_payload_sha256"]):
                        raise ValueError("Selected observation/label/state bytes changed: " + name)
                    with np.load(source/name, allow_pickle=False) as episode, np.load(state_path, allow_pickle=False) as sidecar:
                        states = sidecar["state"]
                        if states.shape != (len(episode["canonical_actions"]), 18):
                            raise ValueError("Projected frame alignment changed")
                        frames = diagnostic_frames(episode["canonical_actions"])
                        task = TaskSpec.from_dict(item["task"])
                        for stage, frame in enumerate(frames):
                            sensor.closure = float(episode["gripper_closure"][frame])
                            observation = VLAObservation(front_rgb=episode["front_rgb"][frame],
                                wrist_rgb=episode["wrist_rgb"][frame], joint_positions=episode["joint_positions"][frame],
                                gripper_width=.085*(1-sensor.closure), gripper_open=float(episode["gripper_open"][frame]),
                                ee_position=episode["ee_position"][frame], ee_rotation_6d=episode["ee_rotation_6d"][frame],
                                prompt=str(episode["prompt"].item()), step=int(frame))
                            policy.source_state = states[frame].copy(); policy.reset(task)
                            predicted = policy.predict(observation)
                            rows.append({"split": split, "task": task_name, "episode": name, "frame": int(frame),
                                "stage": stage, "predicted": predicted.tolist(),
                                "expected": episode["canonical_actions"][frame].tolist(),
                                "commanded_closure_before_action": float(states[frame, 10]),
                                "verified_correct_held_before_action": bool(states[frame, 17]),
                                "pre_action_state18": states[frame].tolist()})
                            teacher_xy = float(np.linalg.norm(episode["canonical_actions"][frame, :2]))
                            if split == "validation" and (frame == 0 or teacher_xy > .002):
                                for group, columns in (("target", slice(11, 14)), ("goal", slice(14, 17)), ("held", slice(17, 18))):
                                    policy.source_state = states[frame].copy()
                                    policy.source_state[columns] = mean[columns]; policy.reset(task)
                                    changed = policy.predict(observation)
                                    interventions.append({"task": task_name, "episode": name, "group": group,
                                        "frame": int(frame), "teacher_xy_norm_m": teacher_xy,
                                        "early_vertical_frame_not_grounding_evidence": teacher_xy <= .002,
                                        "baseline": predicted.tolist(), "counterfactual": changed.tolist(),
                                        "translation_delta_m": float(np.linalg.norm(changed[:3]-predicted[:3])),
                                        "gripper_delta": float(changed[6]-predicted[6]),
                                        "input_changed": bool(np.any(states[frame, columns] != mean[columns]))})
                    atomic_json(args.output/"progress.json", {"status": "running", "predictions": len(rows),
                        "interventions": len(interventions), "rows": rows, "robot_actions_executed": 0})
                    print(split, task_name, name, len(rows), flush=True)
        verify_fingerprint(args.checkpoint, full=True)
        reports = {split: {"overall": summarize([r for r in rows if r["split"] == split]),
            "by_task": {task: summarize([r for r in rows if r["split"] == split and r["task"] == task])
                        for task in ("touch", "pick", "move", "pick_place")},
            "first_frame": summarize([r for r in rows if r["split"] == split and r["frame"] == 0]),
            "teacher_XY_moving_frames": (summarize([r for r in rows if r["split"] == split
                and np.linalg.norm(np.asarray(r["expected"])[:2]) > .002])
                if any(r["split"] == split and np.linalg.norm(np.asarray(r["expected"])[:2]) > .002 for r in rows) else None)}
                   for split in ("train", "validation")}
        atomic_json(args.output/"analysis.json", {**plan, "status": "completed_diagnostic_NOT_SR",
            "by_split": reports, "rows": rows, "intervention_rows": interventions})
        atomic_json(args.output/"progress.json", {"status": "completed_diagnostic_NOT_SR", "predictions": len(rows),
            "interventions": len(interventions), "robot_actions_executed": 0})
    except Exception as error:
        atomic_json(args.output/"progress.json", {"status": "preserved_error_no_reroll",
            "error": f"{type(error).__name__}: {error}", "predictions": len(rows), "rows": rows})
        raise
    finally:
        policy.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "train-projection", "validation-projection", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--gpu-memory-mib", type=int, default=3000)
    args = parser.parse_args()
    if args.episodes_per_task < 1 or not 1000 <= args.gpu_memory_mib <= 4000:
        raise ValueError("Positive sample count and bounded GPU budget required")
    run(args)
