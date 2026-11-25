"""Compare commanded gripper proprioception with actual actuator measurements."""
import argparse
import json
from pathlib import Path

import numpy as np
import pybullet as p

from VLA.common.environment import ManipulationEnv
from VLA.common.scripted_policy import ScriptedPolicy
from VLA.common.tasks import TaskSpec
from VLA.dataset_replay_context import audit_environment, reset_source_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((args.source / "index.json").read_text())
    episodes = []
    with audit_environment(args.source, robot="ur5e", image_size=384, required_success_steps=1,
                         render_observations=False) as env:
        for name in ("touch", "pick", "move", "pick_place"):
            item = next(item for item in index if item["task"]["name"] == name)
            task = TaskSpec.from_dict(item["task"])
            observation = reset_source_episode(env, args.source, item)
            policy = ScriptedPolicy(env)
            policy.reset(task)
            rows = []
            for step in range(task.max_steps):
                drive = p.getJointState(env.gripper.body, 1)
                closure = np.clip((drive[0] - env.gripper.OPEN_ANGLE) /
                    (env.gripper.CLOSED_ANGLE - env.gripper.OPEN_ANGLE), 0, 1)
                rows.append({"step": step, "commanded_open": float(observation.gripper_open),
                    "measured_drive_angle_rad": drive[0], "measured_drive_velocity_rad_s": drive[1],
                    "measured_drive_open_fraction": float(1 - closure),
                    "pad_link_center_distance_m": env.gripper.width()})
                observation, reward, done, info = env.step(policy.predict(observation))
                if done:
                    break
            error = [abs(row["commanded_open"] - row["measured_drive_open_fraction"]) for row in rows]
            episodes.append({"task": name, "source_episode": item["file"], "expert_success": info["success"],
                             "max_command_vs_measured_open_difference": max(error),
                             "mean_command_vs_measured_open_difference": float(np.mean(error)), "rows": rows})
    report = {"episodes": episodes, "source": str(args.source.resolve()),
        "current_training_and_inference_gripper_state": "last command, not measured opening",
        "measured_drive_fraction_caveat": "Normalized motor angle is not physical finger aperture; no width calibration assumed",
        "pad_center_distance_caveat": "Link COM distance is not inner-finger aperture; do not divide it by 85mm",
        "original_rdt_contract": "Measured gripper width normalized to [0,1] as per official fine-tuning guide",
        "original_rdt_guide": "https://github.com/thu-ml/RoboticsDiffusionTransformer#fine-tuning-on-your-own-dataset",
        "production_state_changed": False, "scope": "Four training-scene sensor probes; no learned policy scored"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": [{key: value for key, value in ep.items() if key != "rows"}
                                   for ep in episodes], "production_state_changed": False}, indent=2))


if __name__ == "__main__":
    main()
