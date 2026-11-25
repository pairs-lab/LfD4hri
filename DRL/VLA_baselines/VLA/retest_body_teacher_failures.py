"""Retest the identical failed physical scenes with the repaired teacher."""
import argparse
import json
from pathlib import Path
from PIL import Image

from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.tasks import TaskSpec
from VLA.collision_planned_teacher import CollisionPlannedTeacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic", action="store_true", help="Use the already defined VLA semantic regions, not strict RL buffer collection")
    parser.add_argument("--gear", action="store_true", help="Separate explicit physical mimic-coupling experiment")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    profile = json.loads(Path("VLA/profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
    if args.gear:
        profile.update(name=profile["name"]+"_gear_candidate", mimic_coupling="gear")
    rows = json.loads((args.source/"index.json").read_text())
    results = []
    with BodyGraspEnvironment(profile, image_size=384, required_success_steps=1 if args.semantic else 10,
                              use_semantic_success=args.semantic) as env:
        for row in rows:
            if row["success"]:
                continue
            task = TaskSpec.from_dict(row["task"])
            observation = env.reset(task)
            teacher = CollisionPlannedTeacher(env)
            teacher.reset(task)
            for _ in range(task.max_steps):
                observation, _, done, info = env.step(teacher.predict(observation))
                if done:
                    break
            for camera in ("front", "wrist"):
                Image.fromarray(env.render(camera)).save(args.output / (row["file"]+"_"+camera+".png"))
            results.append({"file":row["file"],"success":info["success"],"steps":env.step_count,
                "safety_violation":info.get("safety_violation"),"max_mimic_error_rad":env.safety_max_mimic_error,
                "safety_diagnostics_before_rollback":info.get("safety_diagnostics_before_rollback"),
                "bilateral_contact":env.gripper._bilateral(env.pick_id),
                "held_requested_object":env.gripper.grasped_object==env.pick_id,
                "semantic_metrics":env.semantic_success_metrics()})
            print(json.dumps(results[-1]),flush=True)
    (args.output/"audit.json").write_text(json.dumps({"results":results,"source_modified":False,
        "object_poses_modified":False,"final_test_used":False},indent=2)+"\n")


if __name__ == "__main__":
    main()
