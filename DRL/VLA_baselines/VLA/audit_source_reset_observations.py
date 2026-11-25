"""Bounded CPU sensor-reset parity check in the actual deployment interpreter.

This checks scene reconstruction, not learned success. It never creates a
policy or supplies source labels to a policy, and never moves the robot after
the recorded reset/startup sequence.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.evaluate_smolvla_v9_source import (reset_observation_errors, select_episodes,
                                         validate_reset_observation, verify_source_audit)


def main():
    import pybullet

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    args = parser.parse_args()
    source = args.source.resolve()
    if args.output.exists() or args.output.with_suffix("").exists():
        raise FileExistsError("Refusing to overwrite reset-parity evidence")
    audit, manifest = verify_source_audit(source, args.audit_report)
    rows = select_episodes(json.loads((source / "index.json").read_text()), args.episodes_per_task)
    images = args.output.with_suffix("")
    images.mkdir(parents=True)
    results = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1., use_semantic_success=True) as env:
        for row in rows:
            observation = reset_source_episode(env, source, row)
            with np.load(source / row["file"], allow_pickle=False) as recorded:
                errors = reset_observation_errors(observation, env.normalized_gripper_closure(), recorded)
            initial_success = bool(env.is_semantic_success())
            failures = []
            try:
                validate_reset_observation(errors)
            except ValueError as exc:
                failures.append(str(exc))
            if initial_success:
                failures.append("Task already succeeds at reset")
            for camera in ("front", "wrist"):
                Image.fromarray(getattr(observation, camera + "_rgb")).save(
                    images / (Path(row["file"]).stem + "_" + camera + ".png"))
            results.append({"file": row["file"], "task": row["task"]["name"],
                            "recovery_startup_present": bool(row.get("recovery_start_actions_not_training_labels")),
                            "passed": not failures, "failures": failures,
                            "initial_semantic_success": initial_success, **errors})
            print(row["file"], "PASS" if not failures else failures, flush=True)
    verify_source_audit(source, args.audit_report)
    report = {"scope": "deployment_sensor_reset_parity_not_learned_policy_success",
              "passed": all(row["passed"] for row in results), "source": str(source),
              "source_index_sha256": audit["index_sha256"], "source_manifest_sha256": audit["manifest_sha256"],
              "source_payload_fingerprint": audit["source_payload_fingerprint"],
              "environment_runtime_fingerprint": manifest["environment_runtime_fingerprint"],
              "python": sys.executable, "pybullet_version": importlib.metadata.version("pybullet"),
              "pybullet_binary_sha256": sha256_file(Path(pybullet.__file__)),
              "trials": len(results), "results": results, "reset_images": str(images.resolve())}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
