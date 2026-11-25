"""Replay immutable qualified TRAIN/VAL data to extract causal state18.

Pixels/action labels are not changed or copied into a new fake-success source.
The sidecar is NOT qualified for training until ALL declared source frames,
real sensors, held feedback, hashes and terminal strict outcomes pass.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.cross_vla120_trial import atomic_json
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit, reset_observation_errors, validate_reset_observation
from VLA.simstate_feedback import SCHEMA, FEATURE_NAMES, feedback_state, state_stats

TASKS = ("touch", "pick", "move", "pick_place")


def select_pilot(rows):
    selected = []
    for task in TASKS:
        candidates = [r for r in rows if r["task"]["name"] == task]
        for recovery in (False, True):
            selected.append(next(r for r in candidates if bool(r.get("recovery_start_actions_not_training_labels")) == recovery))
    return selected


def project(source, audit_path, output, *, pilot=False, split="train"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Projection evidence is immutable; do not reroll failed episodes")
    audit, manifest = verify_source_audit(source, audit_path)
    rows = json.loads((source / "index.json").read_text())
    selected = select_pilot(rows) if pilot else rows
    output.mkdir(parents=True)
    (output / "states").mkdir()
    plan = {"schema": SCHEMA, "feature_names": FEATURE_NAMES, "source": str(source),
        "source_index_sha256": sha256_file(source / "index.json"),
        "source_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
        "source_audit": str(Path(audit_path).resolve()), "source_audit_sha256": sha256_file(Path(audit_path)),
        "source_episodes": len(rows), "declared_episodes": len(selected), "pilot_only": pilot, "split": split,
        "privileged_simulator_information_in_model_input": True, "teacher_actions_in_model_input": False,
        "future_state_in_model_input": False, "oracle_controller_at_deployment": False,
        "observation_profile_sha256": manifest["observation_profile_sha256"],
        "environment_runtime_fingerprint": environment_runtime_fingerprint(manifest["observation_profile"]),
        "extractor_sha256": sha256_file(Path(__file__)), "feature_builder_sha256": sha256_file(Path(__file__).with_name("simstate_feedback.py")),
        "selected_files": [r["file"] for r in selected], "candidate_only": True}
    atomic_json(output / "plan.json", plan)
    reports, all_states, frames, held_frames = [], [], 0, 0
    progress_path = output / "progress.json"
    def save(status, error=None):
        atomic_json(progress_path, {"status": status, "split": split, "pilot_only": pilot,
            "completed_episodes": len(reports), "planned_episodes": len(selected), "frames": frames,
            "held_frames": held_frames, "rows": reports, "error": error, "neural_episodes_executed": 0})
    save("extracting_causal_states_from_real_physics")
    try:
        for item in selected:
            payload = source / item["file"]
            digest = sha256_file(payload)
            with np.load(payload, allow_pickle=False) as data:
                with audit_environment(source, image_size=384, control_hz=10, seed=0,
                    required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
                    observation = reset_source_episode(env, source, item)
                    maximum = {"joint_max_abs_error": 0., "measured_closure_abs_error": 0.,
                        "front_rgb_max_abs_error": 0, "front_rgb_changed_pixel_fraction": 0.,
                        "wrist_rgb_max_abs_error": 0, "wrist_rgb_changed_pixel_fraction": 0.}
                    initial_check = validate_reset_observation(reset_observation_errors(observation, env.normalized_gripper_closure(), data))
                    states = []
                    for frame, action in enumerate(data["canonical_actions"]):
                        errors = reset_observation_errors(observation, env.normalized_gripper_closure(),
                            {k: data[k][frame:frame+1] for k in ("joint_positions", "gripper_closure", "front_rgb", "wrist_rgb")})
                        validate_reset_observation(errors)
                        for key in maximum:
                            maximum[key] = max(maximum[key], errors[key])
                        values = feedback_state(env, observation)
                        # Grasp feedback is PRE-action: compare to the previous
                        # recorded post-action value, never next_verified_grasp[t].
                        expected_hold = bool(data["next_verified_grasp"][frame-1]) if frame else False
                        if bool(values[-1]) != expected_hold:
                            raise ValueError("Current correct-body contact feedback differs from recorded previous state")
                        if frame and values[10] != np.float32(data["next_requested_gripper_closure"][frame-1]):
                            raise ValueError("Previous CLOSE intent is not measured width or current/future label")
                        original_state = np.r_[data["joint_positions"][frame], data["gripper_closure"][frame]].astype(np.float32)
                        if np.max(np.abs(values[:7] - original_state)) > 1e-6:
                            raise ValueError("Original state7 no longer matches the real causal replay")
                        states.append(values)
                        if frame in (0, len(data["canonical_actions"]) - 1):
                            folder = output / "visuals" / Path(item["file"]).stem
                            folder.mkdir(parents=True, exist_ok=True)
                            for camera in ("front", "wrist"):
                                Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"frame{frame:04d}_{camera}.png")
                            atomic_json(folder / f"frame{frame:04d}_state.json", {"schema": SCHEMA, "values": values.tolist(), "feature_names": FEATURE_NAMES})
                        observation, _, _, info = env.step(action)
                        if info.get("safety_violation"):
                            raise ValueError("Replay physics safety violation")
                    if not info["strict_raw_success"]:
                        raise ValueError("Original trajectory no longer reaches its strict terminal outcome")
                    states = np.asarray(states, dtype=np.float32)
                    state_path = output / "states" / item["file"]
                    np.savez_compressed(state_path, state=states)
                    frames += len(states); held_frames += int(states[:, -1].sum()); all_states.append(states)
                    reports.append({"file": item["file"], "task": item["task"]["name"], "frames": len(states),
                        "source_payload_sha256": digest, "state_payload_sha256": sha256_file(state_path),
                        "sensor_maxima": maximum, "initial_check": initial_check, "terminal_strict_success": True})
            if sha256_file(payload) != digest:
                raise ValueError("Immutable original source payload changed")
            save("extracting_causal_states_from_real_physics")
            print(f"Sim-state {split} {len(reports)}/{len(selected)} episodes, {frames} frames", flush=True)
        if (sha256_file(source / "index.json") != plan["source_index_sha256"]
                or sha256_file(source / "dataset_manifest.json") != plan["source_manifest_sha256"]):
            raise ValueError("Original source changed")
        if not pilot and frames != manifest["frames"]:
            raise ValueError("Incomplete full-source projection")
        stats = state_stats(np.concatenate(all_states))
        atomic_json(output / "stats.json", stats)
        atomic_json(output / "audit.json", {**plan, "passed": True, "rows": reports, "frames": frames,
            "stats_sha256": sha256_file(output / "stats.json"), "neural_episodes_executed": 0,
            "qualified_for_training": not pilot and split == "train", "candidate_only": pilot,
            "images_or_action_labels_modified": False, "current_state_not_future_hold_or_goal": True})
        save("pilot_passed_NOT_training_qualified" if pilot else "full_causal_projection_passed")
    except Exception as error:
        save("failed_projection_NO_training", f"{type(error).__name__}: {error}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    project(args.source, args.source_audit, args.output, pilot=args.pilot, split=args.split)


if __name__ == "__main__":
    main()
