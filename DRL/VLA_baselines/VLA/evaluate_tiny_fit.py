"""Learned-policy closed-loop reproduction of tiny training scenes, NOT SR.

The harness reconstructs source reset/spawn order. Only the normal RGB,
proprioception and text observation goes to the learned policy; teacher
actions, segmentation, object positions and goal coordinates are not used
to produce controls. No oracle fallback and no final/development seed sweep.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from VLA.common.observation_profile import profile_hash
from VLA.common.profile_binding import checkpoint_profile
from VLA.common.recorder import VideoRecorder
from VLA.common.tasks import TaskSpec
from VLA.common.provenance import sha256_file
from VLA.dataset_replay_context import audit_environment, reset_source_episode
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.adapter import SmolVLAPolicyAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    audit = json.loads(args.audit_report.read_text())
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    if (not manifest.get("tiny_fit_only") or not audit.get("passed") or audit.get("failures")
            or audit.get("index_sha256") != sha256_file(source / "index.json")
            or audit.get("manifest_sha256") != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Requires qualified tiny TRAIN source")
    profile = checkpoint_profile(args.checkpoint)
    if profile is None or profile_hash(profile) != manifest["observation_profile_sha256"]:
        raise ValueError("Source and checkpoint sensor/controller profiles differ")
    identity = verify_fingerprint(args.checkpoint)
    index = json.loads((source / "index.json").read_text())
    args.output.mkdir(parents=True)
    results = []
    with audit_environment(source, image_size=384, required_success_steps=1,
                           success_threshold_scale=1.25, use_semantic_success=True) as env:
        policy = SmolVLAPolicyAdapter(env, checkpoint=str(args.checkpoint.resolve()), seed=20261008)
        recorder = VideoRecorder(args.output / "rollout.mp4", fps=10,
                                 overview_render=lambda: env.render("overview"))
        try:
            for row in index:
                task = TaskSpec.from_dict(row["task"])
                observation = reset_source_episode(env, source, row)
                policy.reset(task)
                folder = args.output / Path(row["file"]).stem
                folder.mkdir()
                actions, correct, wrong, info = [], False, False, None
                initial_success = env.is_semantic_success()
                for step in range(task.max_steps):
                    if step % 10 == 0:
                        for camera in ("front", "wrist"):
                            Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"step_{step:04d}_{camera}.png")
                    recorder.add(observation, f"Tiny TRAIN fit only | {task.prompt}")
                    action = policy.predict(observation)
                    if action.shape != (7,) or not np.isfinite(action).all():
                        raise ValueError("Invalid learned action")
                    actions.append(action.tolist())
                    observation, _, done, info = env.step(action)
                    grasped = env.gripper.grasped_object
                    correct |= grasped == env.pick_id
                    wrong |= grasped is not None and grasped != env.pick_id
                    if done:
                        break
                # Save the terminal observation too: a 10-step logging period
                # otherwise leaves a short successful episode with only t=0.
                for camera in ("front", "wrist"):
                    Image.fromarray(getattr(observation, camera + "_rgb")).save(folder / f"terminal_{info['step']:04d}_{camera}.png")
                recorder.add(observation, f"Tiny TRAIN fit only | success={info['success']}")
                result = {"file": row["file"], "task": task.name, "prompt": task.prompt,
                          "steps": info["step"], "success": info["success"],
                          "initial_semantic_success": initial_success,
                          "correct_target_grasp_seen": correct, "wrong_target_grasp_seen": wrong,
                          "final_ee_object_distance_m": info["ee_object_distance"],
                          "final_goal_xy_distance_m": info["goal_xy_distance_m"], "actions": actions}
                results.append(result)
                (args.output / "progress.json").write_text(json.dumps(results, indent=2) + "\n")
                print(f"{row['file']}: learned success={result['success']} steps={result['steps']} wrong grasp={wrong}", flush=True)
        finally:
            recorder.close()
            policy.close()
    report = {"scope": "Tiny training-scene closed-loop sanity; NOT held-out/model benchmark success rate",
              "source": str(source), "source_index_sha256": sha256_file(source / "index.json"),
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_content_sha256": identity["content_tree_sha256"],
              "trials": len(results), "successes": sum(r["success"] for r in results),
              "privileged_information_in_controls": False, "oracle_fallback": False,
              "final_test_used": False, "results": results}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
