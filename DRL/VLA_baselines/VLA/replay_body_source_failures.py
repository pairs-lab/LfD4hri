"""Reproduce committed failed candidate controls and report their real cause."""
import argparse
import json
from pathlib import Path
import numpy as np

from VLA.common.body_environment import BodyGraspEnvironment
from VLA.common.tasks import TaskSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(Path("VLA/profiles/body_contact_v9_wrist65_f80_encoder.json").read_text())
    rows = json.loads((args.source / "index.json").read_text())
    results = []
    with BodyGraspEnvironment(profile, image_size=32, render_observations=False, required_success_steps=10) as env:
        for row in rows:
            if row["success"]:
                continue
            env.reset(TaskSpec.from_dict(row["task"]))
            with np.load(args.source / row["file"]) as data:
                for action in data["canonical_actions"]:
                    _, _, done, info = env.step(action)
                    if done:
                        break
            result = {"file": row["file"], "steps": env.step_count, "success": info["success"],
                      "safety_violation": info.get("safety_violation"), "rollback": info.get("rollback_restored")}
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": results, "source_modified": False, "final_test_used": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
