"""Replay ALL preserved FAST-codec failures with original/decoded controls.

Offline label-path diagnosis, not learned policy evaluation. Saves actual RGB,
one combined video and grasp/safety feedback that NEVER enters a neural input.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.recorder import VideoRecorder
from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.pi0.codec_provenance import verify_fast_codec_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.failure_report.read_text())
    selected = [row["file"] for row in report["rows"] if not row["success"]]
    if report["passed"] or not selected or report["mode"] != "native":
        raise ValueError("Requires actual preserved native-codec failures")
    source, codec = Path(report["source"]), Path(report["native_actions_root"])
    manifest = json.loads((codec / "codec_manifest.json").read_text())
    verify_fast_codec_binding(manifest, source, codec)
    if sha256_file(codec / "codec_manifest.json") != report["codec_manifest_sha256"]:
        raise RuntimeError("Codec differs from the recorded failed experiment")
    index = {row["file"]: row for row in json.loads((source / "index.json").read_text())}
    profile = json.loads((source / "dataset_manifest.json").read_text())["observation_profile"]
    runtime = environment_runtime_fingerprint(profile)
    frozen = {str(path): sha256_file(path) for path in
              (args.failure_report, source / "index.json", source / "dataset_manifest.json", Path(__file__))}
    args.output.mkdir(parents=True)
    results = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           use_semantic_success=False) as env:
        with VideoRecorder(args.output / "original_vs_codec.mp4", fps=10,
                           overview_render=lambda: env.render("overview")) as recorder:
            for file in selected:
                for mode in ("original_native", "decoded_fast"):
                    observation = reset_source_episode(env, source, index[file])
                    with np.load(source / file, allow_pickle=False) as original:
                        canonical = original["canonical_actions"]
                        targets = original["next_joint_targets"].copy()
                        closures = native_gripper_closure(original).copy()
                    if mode == "decoded_fast":
                        with np.load(codec / file, allow_pickle=False) as decoded:
                            targets, closures = decoded["next_joint_targets"].copy(), decoded["next_gripper_closure"].copy()
                    folder = args.output / Path(file).stem / mode
                    folder.mkdir(parents=True)
                    trace = []
                    recorder.add(observation, f"OFFLINE {mode} | {index[file]['task']['prompt']}")
                    for step, (target, closure) in enumerate(zip(targets, closures)):
                        action = env.joint_targets_to_canonical(target, float(closure))
                        observation, _, done, info = env.step(action)
                        held = env.gripper.grasped_object
                        trace.append({"step": step + 1, "semantic_success": bool(info["semantic_raw_success"]),
                            "strict_success": bool(info["strict_raw_success"]),
                            "correct_target_held": held == env.pick_id,
                            "any_object_held": held is not None,
                            "measured_closure": env.normalized_gripper_closure(),
                            "safety_violation": info.get("safety_violation"),
                            "canonical_translation_error_m": float(abs(action[:3] - canonical[step, :3]).max()),
                            "ee_object_distance_m": float(info["ee_object_distance"]),
                            "object_goal_distance_m": float(info["object_goal_distance"])})
                        recorder.add(observation, f"OFFLINE {mode} | step {step + 1} held={held == env.pick_id}")
                        if step % 10 == 0 or closure >= .5 or done or step == len(targets) - 1:
                            for camera in ("front", "wrist"):
                                Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"step_{step+1:03d}_{camera}.png")
                        if done:
                            break
                    row = {"file": file, "mode": mode, "steps": len(trace), "trace": trace,
                           "success": any(t["semantic_success"] for t in trace),
                           "ever_correct_target_held": any(t["correct_target_held"] for t in trace),
                           "safety_abort": any(t["safety_violation"] for t in trace)}
                    results.append(row)
                    print(file, mode, row["success"], row["ever_correct_target_held"], row["safety_abort"], flush=True)
    verify_environment_runtime_fingerprint(runtime, profile)
    verify_fast_codec_binding(manifest, source, codec)
    if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
        raise RuntimeError("Diagnostic inputs changed during replay")
    (args.output / "results.json").write_text(json.dumps({
        "scope": "ALL preserved codec failures_OFFLINE_label_diagnosis_NOT_neural_SR",
        "frozen_files": frozen, "environment_runtime_fingerprint": runtime,
        "learned_policy_called": False, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
