"""Measure RDT in-sample imitation through the actual deployed HTTP/FK path."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure
import pybullet as p
import requests

from VLA.common.observation import VLAObservation
from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
from VLA.common.source_payload_provenance import verify_source_payload_for_profile
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.pi0.audit_checkpoint_predictions import aggregate
from VLA.rdt.adapter import RDTPolicy


def observation_at(episode, frame):
    return VLAObservation(
        episode["front_rgb"][frame], episode["wrist_rgb"][frame], episode["joint_positions"][frame],
        .085 * (1 - float(episode["gripper_closure"][frame])), float(episode["gripper_open"][frame]),
        episode["ee_position"][frame], episode["ee_rotation_6d"][frame],
        str(episode["prompt"].item()), int(frame),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes_per_task < 1:
        raise ValueError("episodes-per-task must be positive")
    source = args.source.resolve()
    audit = json.loads(args.audit_report.read_text())
    verify_source_payload_for_profile(audit, source)
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    runtime_fingerprint = environment_runtime_fingerprint(manifest.get("observation_profile"))
    if (not audit.get("passed") or audit.get("failures")
            or Path(audit["dataset"]).resolve() != source
            or audit["index_sha256"] != sha256_file(source / "index.json")
            or audit["manifest_sha256"] != sha256_file(source / "dataset_manifest.json")):
        raise RuntimeError("Source audit is stale or failed")
    identity = verify_fingerprint(args.checkpoint.resolve())
    grouped = defaultdict(list)
    for item in json.loads((source / "index.json").read_text()):
        grouped[item["task"]["name"]].append(item)
    rows = []
    with audit_environment(source, robot="ur5e", image_size=384, control_hz=10, render_observations=False) as env:
        policy = RDTPolicy(env, checkpoint=str(args.checkpoint.resolve()), port=args.port, seed=20261007)
        policy._verify_server()
        for name in ("touch", "pick", "move", "pick_place"):
            items = grouped[name]
            for number in np.linspace(0, len(items) - 1, min(len(items), args.episodes_per_task), dtype=int):
                item = items[int(number)]
                reset_source_episode(env, source, item)
                with np.load(source / item["file"], allow_pickle=False) as episode:
                    frames = np.unique(np.rint(np.linspace(0, len(episode["canonical_actions"]) - 1, 4)).astype(int))
                    for frame in frames:
                        observation = observation_at(episode, frame)
                        previous = observation_at(episode, max(0, frame - 1))
                        response = requests.post(policy.url, json=policy.build_sensor_request(observation, previous), timeout=600)
                        response.raise_for_status()
                        native = np.asarray(response.json()["joint_actions"], dtype=np.float32)[0]
                        if native.shape != (7,) or not np.isfinite(native).all():
                            raise RuntimeError("Invalid RDT native prediction")
                        for joint, value in zip(env.robot_spec.arm_joint_ids, observation.joint_positions, strict=True):
                            p.resetJointState(env.robot_id, joint, float(value))
                        predicted = env.joint_targets_to_canonical(native[:6], 1 - float(native[6]))
                        expected = episode["canonical_actions"][frame]
                        joint_error = native[:6] - episode["next_joint_targets"][frame]
                        joint_error = (joint_error + np.pi) % (2 * np.pi) - np.pi
                        native_error = np.r_[joint_error, native[6] - (1 - native_gripper_closure(episode)[frame])]
                        rows.append({"task": name, "episode": item["file"], "frame": int(frame),
                                     "prompt": observation.prompt, "predicted_canonical": predicted.tolist(),
                                     "expected_canonical": expected.tolist(), "canonical_error": (predicted - expected).tolist(),
                                     "predicted_native": native.tolist(), "native_error": native_error.tolist()})
                print(f"{item['file']}: checked {len(rows)} observations", flush=True)
    native_errors = np.asarray([row["native_error"] for row in rows])
    verify_source_payload_for_profile(audit, source)
    verify_environment_runtime_fingerprint(runtime_fingerprint, manifest.get("observation_profile"))
    report = {"integration_passed": True, "measurement": "in_sample_imitation_not_success_rate",
              "server_metadata": policy.server_metadata,
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
              "source_index_sha256": sha256_file(source / "index.json"),
              "source_manifest_sha256": sha256_file(source / "dataset_manifest.json"),
              "source_payload_fingerprint": audit.get("source_payload_fingerprint"),
              "environment_runtime_fingerprint": runtime_fingerprint,
              "input_path": "deployed RDTPolicy.build_sensor_request with exact source t-1 history; actual server; exact common FK",
              "privileged_target_coordinates_consumed": False, "overall": aggregate(rows),
              "decoded_native_mse_active7": float(np.square(native_errors).mean()),
              "decoded_native_joint_rmse_rad": float(np.sqrt(np.square(native_errors[:, :6]).mean())),
              "training_loss_caveat": "Official training loss averages all 128 coordinates, not just the 7 executable UR5 controls; this decoded error is not the stochastic denoising loss",
              "by_task": {name: aggregate([row for row in rows if row["task"] == name]) for name in grouped},
              "initial_frame": aggregate([row for row in rows if row["frame"] == 0]), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
