"""One honest combined video: corrected teacher source replay, NOT VLA success."""
import argparse
import json
from pathlib import Path
import numpy as np
from VLA.common.recorder import VideoRecorder
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="Optional FRONT-only camera candidate for the teacher video")
    parser.add_argument("--prefer-pick", help="Prefer the repaired hard object for each task when present")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows = json.loads((args.source/"index.json").read_text())
    selected = []
    for name in ("touch","pick","move","pick_place"):
        candidates = [r for r in rows if r["task"]["name"]==name]
        preferred = [r for r in candidates if r["task"]["pick_name"] == args.prefer_pick]
        selected.append((preferred or candidates)[0])
    results = []
    if args.profile:
        from VLA.rerender_body_v9_front_candidate import front_only_profile
        from VLA.common.body_environment import BodyGraspEnvironment
        old_profile = json.loads((args.source/"dataset_manifest.json").read_text())["observation_profile"]
        profile = front_only_profile(old_profile, json.loads(args.profile.read_text()))
        environment = BodyGraspEnvironment(profile, image_size=384, required_success_steps=1, use_semantic_success=False)
    else:
        environment = audit_environment(args.source, image_size=384, required_success_steps=1, use_semantic_success=False)
    with environment as env:
        recorder = VideoRecorder(args.output/"rollout.mp4", fps=10, overview_render=lambda: env.render("overview"))
        try:
            for row in selected:
                observation = reset_source_episode(env, args.source, row)
                with np.load(args.source/row["file"]) as data:
                    for action in data["canonical_actions"]:
                        recorder.add(observation, "TEACHER replay only | "+row["task"]["prompt"])
                        observation, _, done, info = env.step(action)
                        if done:
                            break
                recorder.add(observation, f"TEACHER ONLY | success={info['success']}", repeat=10)
                results.append({"task":row["task"]["name"],"source_episode":row["file"],
                    "target":row["task"]["pick_name"],"success":info["success"],"safety":info.get("safety_violation"),
                    "held_requested_object": env.gripper.grasped_object == env.pick_id,
                    "bilateral_pad_contact":env.gripper._bilateral(env.pick_id)})
        finally:
            recorder.close()
    (args.output/"results.json").write_text(json.dumps({"scope":"Corrected teacher control replay video, NOT learned success rate",
        "results":results,"source":str(args.source.resolve()),
        "video_profile":None if args.profile is None else str(args.profile.resolve()),
        "video_does_not_certify_new_source_images":True,"final_test_used":False},indent=2)+"\n")


if __name__=="__main__":
    main()
