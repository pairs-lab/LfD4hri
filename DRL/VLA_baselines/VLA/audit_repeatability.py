"""Compare duplicate seeded rollouts without treating repetitions as new trials."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifests = [json.loads((root / "manifest.json").read_text()) for root in (args.first, args.second)]
    for key in ("robot", "gripper", "control_hz", "replan_steps", "seed", "scene_profile", "image_size", "cameras", "tasks", "checkpoint", "reward_contract"):
        if manifests[0][key] != manifests[1][key]:
            raise RuntimeError("Different rollout condition: " + key)
    traces = [json.loads((root / "trajectory.json").read_text()) for root in (args.first, args.second)]
    results = [json.loads((root / "results.json").read_text()) for root in (args.first, args.second)]
    summaries, passed = {}, True
    for name in traces[0]:
        arrays = [np.asarray([step["action"] for step in trace[name]], dtype=np.float32) for trace in traces]
        exact = arrays[0].shape == arrays[1].shape and np.array_equal(*arrays)
        outcomes = [next(task for task in result["tasks"] if task["name"] == name) for result in results]
        same_outcome = all(outcomes[0][key] == outcomes[1][key] for key in ("success", "steps", "error"))
        summaries[name] = {"action_sequence_bitwise_identical": exact, "outcome_identical": same_outcome,
                           "steps": [len(array) for array in arrays],
                           "action_sha256": [hashlib.sha256(array.tobytes()).hexdigest() for array in arrays],
                           "success": outcomes[0]["success"]}
        passed &= exact and same_outcome
    report = {"passed": passed, "first": str(args.first.resolve()), "second": str(args.second.resolve()),
              "tasks": summaries, "measurement": "seeded_repeatability_not_population_success_rate",
              "duplicates_count_as_independent_trials": False,
              "latency_comparison_omitted": "GPU is shared with training"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Duplicate rollout differs; see saved report")


if __name__ == "__main__":
    main()
