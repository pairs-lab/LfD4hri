"""Preserve audited prefix; materialize NPZ members once; parallel CPU replay.

New immutable output. Old partial evidence is read-only and reused ONLY after
its receipt/source/runtime hashes pass. Failed projections cannot be retried.
Workers are independent PyBullet processes, not neural/oracle policy trials.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint
from VLA.cross_vla120_trial import atomic_json
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import verify_source_audit, reset_observation_errors, validate_reset_observation
from VLA.simstate_feedback import SCHEMA, FEATURE_NAMES, feedback_state, state_stats


def extract_episode(source, output, item):
    source, output = Path(source), Path(output)
    payload = source/item["file"]
    digest = sha256_file(payload)
    # Critical throughput fix: NPZ is compressed per member. Repeatedly
    # indexing data['front_rgb'] inside a frame loop decompresses ALL frames.
    with np.load(payload, allow_pickle=False) as archive:
        names = ("canonical_actions", "joint_positions", "gripper_closure", "front_rgb", "wrist_rgb",
                 "next_verified_grasp", "next_requested_gripper_closure")
        data = {k: archive[k] for k in names}
    with audit_environment(source, image_size=384, control_hz=10, seed=0,
            required_success_steps=1, success_threshold_scale=1., use_semantic_success=False) as env:
        observation = reset_source_episode(env, source, item)
        initial = validate_reset_observation(reset_observation_errors(observation, env.normalized_gripper_closure(), data))
        maximum = {"joint_max_abs_error": 0., "measured_closure_abs_error": 0., "front_rgb_max_abs_error": 0,
            "front_rgb_changed_pixel_fraction": 0., "wrist_rgb_max_abs_error": 0, "wrist_rgb_changed_pixel_fraction": 0.}
        states = []
        for frame, action in enumerate(data["canonical_actions"]):
            errors = reset_observation_errors(observation, env.normalized_gripper_closure(),
                {k: data[k][frame:frame+1] for k in ("joint_positions", "gripper_closure", "front_rgb", "wrist_rgb")})
            validate_reset_observation(errors)
            for key in maximum: maximum[key] = max(maximum[key], errors[key])
            values = feedback_state(env, observation)
            hold = bool(data["next_verified_grasp"][frame-1]) if frame else False
            if bool(values[17]) != hold or (frame and values[10] != np.float32(data["next_requested_gripper_closure"][frame-1])):
                raise ValueError("Causal grasp/previous-command mismatch")
            raw = np.r_[data["joint_positions"][frame], data["gripper_closure"][frame]].astype(np.float32)
            if np.abs(values[:7]-raw).max() > 1e-6:
                raise ValueError("Original causal state7 mismatch")
            states.append(values)
            if frame in (0, len(data["canonical_actions"])-1):
                folder = output/"visuals"/Path(item["file"]).stem
                folder.mkdir(parents=True, exist_ok=True)
                for camera in ("front", "wrist"):
                    Image.fromarray(getattr(observation, camera+"_rgb")).save(folder/f"frame{frame:04d}_{camera}.png")
                atomic_json(folder/f"frame{frame:04d}_state.json", {"schema": SCHEMA, "values": values.tolist(), "feature_names": FEATURE_NAMES})
            observation, _, _, info = env.step(action)
            if info.get("safety_violation"):
                raise ValueError("Replay safety violation; no retry")
        if not info["strict_raw_success"]:
            raise ValueError("Strict terminal outcome failed")
    if sha256_file(payload) != digest:
        raise ValueError("Immutable source changed")
    states = np.asarray(states, dtype=np.float32)
    path = output/"states"/item["file"]
    if path.exists(): raise FileExistsError("Never overwrite state evidence")
    np.savez_compressed(path, state=states)
    return {"file": item["file"], "task": item["task"]["name"], "frames": len(states),
        "source_payload_sha256": digest, "state_payload_sha256": sha256_file(path),
        "sensor_maxima": maximum, "initial_check": initial, "terminal_strict_success": True}


def reuse_prefix(previous, source, output, rows, plan):
    previous = Path(previous)
    old_plan = json.loads((previous/"plan.json").read_text())
    old = json.loads((previous/"progress.json").read_text())
    if old["status"] != "extracting_causal_states_from_real_physics":
        raise ValueError("Only interrupted, NOT failed, projections may continue")
    for field in ("schema", "source", "source_index_sha256", "source_manifest_sha256", "feature_builder_sha256",
                  "source_audit_sha256", "environment_runtime_fingerprint", "selected_files", "split"):
        if old_plan[field] != plan[field]:
            raise ValueError("Interrupted projection binding differs: "+field)
    receipts = old["rows"]
    if [r["file"] for r in receipts] != [r["file"] for r in rows[:len(receipts)]]:
        raise ValueError("Completed prefix/order changed")
    if sum(r["frames"] for r in receipts) != old["frames"]:
        raise ValueError("Interrupted prefix counts differ")
    for item, proof in zip(rows, receipts):
        state = previous/"states"/item["file"]
        if (sha256_file(source/item["file"]) != proof["source_payload_sha256"]
                or sha256_file(state) != proof["state_payload_sha256"] or not proof["terminal_strict_success"]):
            raise ValueError("Cannot reuse stale/failed receipt")
        validate_reset_observation(proof["sensor_maxima"])
        with np.load(state, allow_pickle=False) as archive:
            values = archive["state"]
            if values.shape != (item["steps"],18) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("Interrupted state schema/count changed")
        os.link(state, output/"states"/item["file"])
    return receipts, {"previous_projection": str(previous.resolve()), "reused_episodes": len(receipts),
        "previous_plan_sha256": sha256_file(previous/"plan.json"), "previous_progress_sha256": sha256_file(previous/"progress.json"),
        "old_evidence_never_mutated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.workers <= 4:
        raise ValueError("Fresh output/bounded1..4 independent CPU workers required")
    source, output = args.source.resolve(), args.output.resolve()
    _, manifest = verify_source_audit(source, args.source_audit)
    rows = json.loads((source/"index.json").read_text())
    output.mkdir(parents=True); (output/"states").mkdir()
    plan = {"schema": SCHEMA, "feature_names": FEATURE_NAMES, "source": str(source),
        "source_index_sha256": sha256_file(source/"index.json"), "source_manifest_sha256": sha256_file(source/"dataset_manifest.json"),
        "source_audit": str(args.source_audit.resolve()), "source_audit_sha256": sha256_file(args.source_audit),
        "source_episodes": len(rows), "declared_episodes": len(rows), "pilot_only": False, "split": args.split,
        "privileged_simulator_information_in_model_input": True, "teacher_actions_in_model_input": False,
        "future_state_in_model_input": False, "oracle_controller_at_deployment": False,
        "observation_profile_sha256": manifest["observation_profile_sha256"],
        "environment_runtime_fingerprint": environment_runtime_fingerprint(manifest["observation_profile"]),
        "extractor_sha256": sha256_file(Path(__file__)), "feature_builder_sha256": sha256_file(Path(__file__).with_name("simstate_feedback.py")),
        "selected_files": [r["file"] for r in rows], "candidate_only": True,
        "NPZ_members_materialized_once": True, "independent_CPU_workers": args.workers}
    atomic_json(output/"plan.json", plan)
    completed, inherited = {}, None
    def save(status, error=None):
        proofs = [completed[r["file"]] for r in rows if r["file"] in completed]
        atomic_json(output/"progress.json", {"status": status, "pid": os.getpid(), "split": args.split,
            "completed_episodes": len(proofs), "planned_episodes": len(rows), "frames": sum(r["frames"] for r in proofs),
            "rows": proofs, "error": error, "neural_episodes_executed": 0, "reuse": inherited})
    try:
        if args.previous:
            reused, inherited = reuse_prefix(args.previous, source, output, rows, plan)
            completed = {r["file"]:r for r in reused}
        save("real_causal_replay_in_progress")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = {pool.submit(extract_episode, source, output, row):row for row in rows if row["file"] not in completed}
            try:
                for future in as_completed(pending):
                    proof = future.result()
                    completed[proof["file"]] = proof
                    save("real_causal_replay_in_progress")
                    print(f"Causal {args.split}: {len(completed)}/{len(rows)} (reused {inherited['reused_episodes'] if inherited else 0})", flush=True)
            except Exception:
                for future in pending: future.cancel()
                raise
        proofs, states = [completed[r["file"]] for r in rows], []
        for row in rows:
            with np.load(output/"states"/row["file"], allow_pickle=False) as archive:
                states.append(archive["state"])
        states = np.concatenate(states)
        if len(states) != manifest["frames"]:
            raise ValueError("Not every declared frame is present")
        if sha256_file(source/"index.json") != plan["source_index_sha256"] or sha256_file(source/"dataset_manifest.json") != plan["source_manifest_sha256"]:
            raise ValueError("Source changed during projection")
        atomic_json(output/"stats.json", state_stats(states))
        atomic_json(output/"audit.json", {**plan, "passed": True, "rows": proofs, "frames": len(states), "reuse": inherited,
            "stats_sha256": sha256_file(output/"stats.json"), "neural_episodes_executed": 0,
            "qualified_for_training": args.split == "train", "candidate_only": False,
            "images_or_action_labels_modified": False, "current_state_not_future_hold_or_goal": True})
        save("full_causal_projection_passed")
    except Exception as error:
        save("failed_projection_NO_training", f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
