"""Compare actual PyBullet binaries on identical four-task action replays."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def emit(source: Path, output: Path):
    import pybullet
    from VLA.common.environment import ManipulationEnv
    from VLA.common.tasks import TaskSpec

    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    index = json.loads((source / "index.json").read_text())
    arrays, episodes = {}, []
    for name in ("touch", "pick", "move", "pick_place"):
        item = next(row for row in index if row["task"]["name"] == name)
        task = TaskSpec.from_dict(item["task"])
        with np.load(source / item["file"], allow_pickle=False) as archive:
            actions = archive["canonical_actions"]
        states, frames, successes = [], [], []
        with ManipulationEnv(robot="ur5e", image_size=384, control_hz=10, seed=0,
                             required_success_steps=1, use_semantic_success=True,
                             success_threshold_scale=1.25) as env:
            observation = env.reset(task)
            selected = {0, len(actions) // 2, len(actions)}
            for step in range(len(actions) + 1):
                if step in selected:
                    frames.append(np.stack((observation.front_rgb, observation.wrist_rgb)))
                states.append(np.r_[observation.joint_positions, observation.ee_position,
                                    observation.ee_rotation_6d, observation.gripper_width])
                if step < len(actions):
                    observation, _, _, info = env.step(actions[step])
                    successes.append([info["semantic_raw_success"], info["strict_raw_success"],
                                      info["grasped_object"] is not None])
        arrays[name + "__sensor_state"] = np.asarray(states)
        arrays[name + "__rgb"] = np.asarray(frames)
        arrays[name + "__outcomes"] = np.asarray(successes, dtype=bool)
        episodes.append({"task": name, "file": item["file"], "actions": len(actions),
                         "saved_image_steps": sorted(selected)})
    np.savez_compressed(output / "replay.npz", **arrays)
    binary = Path(pybullet.__file__)
    metadata = {"python": sys.version, "python_executable": sys.executable,
                "pybullet_version": importlib.metadata.version("pybullet"),
                "pybullet_binary": str(binary), "pybullet_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "source": str(source.resolve()), "source_index_sha256": hashlib.sha256((source / "index.json").read_bytes()).hexdigest(),
                "environment_sha256": hashlib.sha256(Path("VLA/common/environment.py").read_bytes()).hexdigest(),
                "episodes": episodes, "measurement": "runtime_equivalence_not_learned_policy_success"}
    (output / "runtime.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def compare(first: Path, second: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    a, b = [json.loads((root / "runtime.json").read_text()) for root in (first, second)]
    for key in ("pybullet_version", "source_index_sha256", "environment_sha256", "episodes"):
        if a[key] != b[key]:
            raise RuntimeError("Different replay contract: " + key)
    differences = {}
    with np.load(first / "replay.npz") as x, np.load(second / "replay.npz") as y:
        if set(x.files) != set(y.files):
            raise RuntimeError("Different replay fields")
        for key in x.files:
            if x[key].shape != y[key].shape:
                raise RuntimeError("Different replay shape: " + key)
            delta = np.abs(x[key].astype(np.float64) - y[key].astype(np.float64))
            differences[key] = {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean()),
                                "different_values": int(np.count_nonzero(delta)), "values": int(delta.size)}
            if key.endswith("__sensor_state"):
                differences[key]["per_column_max_abs_error"] = delta.max(axis=0).tolist()
                # Last field is pad-centre separation, unused by every current
                # adapter. All robot joint/EEF fields actually consumed are first.
                differences[key]["consumed_state_max_abs_error"] = float(delta[:, :15].max())
    exact = all(row["different_values"] == 0 for row in differences.values())
    model_equivalent = all(
        row.get("consumed_state_max_abs_error", row["max_abs_error"]) == 0
        for row in differences.values()
    )
    report = {"passed_bitwise": exact, "passed_model_observation_and_outcome_equivalence": model_equivalent,
              "first": a, "second": b, "differences": differences,
              "interpretation": "Different binary hashes alone do not establish different simulation behavior; actual replay is compared"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed_bitwise": exact, "differences": differences}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("emit", "compare"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--first", type=Path)
    parser.add_argument("--second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "emit":
        if args.source is None:
            raise ValueError("--source is required")
        emit(args.source, args.output)
    else:
        if args.first is None or args.second is None:
            raise ValueError("--first and --second are required")
        compare(args.first, args.second, args.output)


if __name__ == "__main__":
    main()
