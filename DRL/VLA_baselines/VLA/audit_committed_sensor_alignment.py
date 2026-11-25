"""Independent RGB/proprio time-alignment replay of immutable committed samples.

May run while collection appends OTHER files. Binds exact selected payloads,
an immutable index snapshot and the sensor/runtime profile. Never qualifies a
partial candidate or loads a neural policy. No teacher acts during this replay.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def sensor_errors(observation, env, data, frame):
    errors = {}
    for field in ("front_rgb", "wrist_rgb", "joint_positions", "ee_position", "ee_rotation_6d"):
        observed = getattr(observation, field)
        errors[field] = float(np.max(np.abs(np.asarray(observed, float)-np.asarray(data[field][frame], float))))
    errors["gripper_closure"] = float(abs(env.normalized_gripper_closure()-data["gripper_closure"][frame]))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--episode-files", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    snapshot_hash = sha256_file(args.snapshot)
    index = json.loads(args.snapshot.read_text())
    rows = [row for row in index if row["file"] in args.episode_files]
    if len(rows) != len(set(args.episode_files)):
        raise ValueError("Missing or duplicate committed sample")
    profile = json.loads((source/"dataset_manifest.json").read_text())["observation_profile"]
    runtime = environment_runtime_fingerprint(profile)
    payloads = {row["file"]: sha256_file(source/row["file"]) for row in rows}
    args.output.mkdir(parents=True)
    results = []
    with audit_environment(source, image_size=384, required_success_steps=1, use_semantic_success=False) as env:
        for row in rows:
            observation = reset_source_episode(env, source, row)
            with np.load(source/row["file"], allow_pickle=False) as data:
                checked = []
                for frame, action in enumerate(data["canonical_actions"]):
                    errors = sensor_errors(observation, env, data, frame)
                    passed = all(value <= (0 if key.endswith("rgb") else 1e-6)
                                 for key, value in errors.items())
                    checked.append({"frame": frame, "passed": passed, "errors": errors})
                    if frame == 0 or not passed:
                        for camera in ("front", "wrist"):
                            stem = f"{Path(row['file']).stem}_frame{frame:03d}_{camera}"
                            Image.fromarray(getattr(observation, camera+"_rgb")).save(args.output/(stem+"_replayed.png"))
                            Image.fromarray(data[camera+"_rgb"][frame]).save(args.output/(stem+"_recorded.png"))
                    if not passed:
                        # First mismatch is sufficient to reject alignment;
                        # preserve it without continuing a divergent replay.
                        break
                    observation, _, done, info = env.step(action)
                    if done:
                        break
                result = {"file": row["file"], "passed": all(x["passed"] for x in checked),
                    "checked_frames": len(checked), "stored_frames": len(data["canonical_actions"]),
                    "checks": checked, "task": row["task"]["name"]}
                results.append(result)
                print(row["file"], result["passed"], len(checked), checked[-1]["errors"], flush=True)
    verify_environment_runtime_fingerprint(runtime, profile)
    if (sha256_file(args.snapshot) != snapshot_hash
            or any(sha256_file(source/file) != digest for file, digest in payloads.items())):
        raise RuntimeError("Selected immutable inputs changed")
    passed = all(row["passed"] and row["checked_frames"] == row["stored_frames"] for row in results)
    report = {"scope": "Selected committed sensor alignment diagnostic_NOT_full_qualification_or_neural_SR",
        "passed": passed, "source": str(source), "snapshot_sha256": snapshot_hash,
        "payloads": payloads, "environment_runtime_fingerprint": runtime, "results": results,
        "neural_weights_loaded": False, "training_qualified": False}
    (args.output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    if not passed:
        raise RuntimeError("Sensor alignment discrepancy preserved; investigate before fitting")


if __name__ == "__main__":
    main()
