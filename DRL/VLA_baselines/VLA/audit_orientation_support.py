"""Measure teacher-label rotation support and actual rollout tool tilt.

Read-only diagnostic: correlation is not a root-cause or model-SR certificate.
No robot/world modification, no normalization change, no new neural trial.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file
from VLA.cross_vla120_trial import atomic_json


def tilt(values):
    # Repository's OBS contract is two COLUMNS transposed then flattened,
    # not a row-major3x2 array. Incorrect reshaping invents tool tilt.
    columns = np.asarray(values, np.float64).reshape(-1, 2, 3)
    axis = np.cross(columns[:, 0, :], columns[:, 1, :])
    return np.degrees(np.arccos(np.clip(np.abs(axis[:, 2]), 0, 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "progress", "output"): p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    source = args.source.resolve(); index = json.loads((source / "index.json").read_text())
    orientations = []; actions = []
    for row in index:
        with np.load(source / row["file"], allow_pickle=False) as data:
            orientations.append(tilt(data["ee_rotation_6d"]))
            actions.append(data["canonical_actions"][:, 3:6].copy())
    angles = np.concatenate(orientations); rotations = np.concatenate(actions)
    progress = json.loads(args.progress.read_text()); rollout = []
    for row in progress["rows"]:
        folder = Path(row["trajectory"]).parent / "observations" / row["task"]
        observed = []; paths = []
        for path in sorted(folder.glob("step_*.json")):
            values = json.loads(path.read_text())
            if "ee_rotation_6d" not in values: raise ValueError("Actual recorded pose required, no action integration approximation")
            observed.append(float(tilt(values["ee_rotation_6d"])[0])); paths.append(str(path))
        if not observed: raise ValueError("Missing actual physical orientation snapshots")
        rollout.append({"seed": row["seed"], "task": row["task"], "success": row["success"],
            "maximum_observed_tilt_degrees": max(observed), "nonplanar_seen": bool(max(observed) > np.degrees(np.arccos(.9))),
            "actual_pose_samples": len(observed), "paths": paths})
    report = {"source_index_sha256": sha256_file(source / "index.json"), "source_frames": len(angles),
        "training_roll_pitch_nonzero_labels": int(np.any(np.abs(rotations[:, :2]) > 1e-8, axis=1).sum()),
        "training_rotation_min": rotations.min(0).tolist(), "training_rotation_max": rotations.max(0).tolist(),
        "training_rotation_std": rotations.astype(np.float64).std(0).tolist(),
        "training_tilt_degrees_quantiles": dict(zip(["p50", "p90", "p99", "max"], np.quantile(angles, [.5, .9, .99, 1]).tolist())),
        "training_nonplanar_frames": int((angles > np.degrees(np.arccos(.9))).sum()),
        "rollout_episodes": len(rollout), "success_with_nonplanar_seen": int(sum(r["success"] and r["nonplanar_seen"] for r in rollout)),
        "failure_with_nonplanar_seen": int(sum(not r["success"] and r["nonplanar_seen"] for r in rollout)),
        "actual_orientation_snapshots_NOT_full_frame_exhaustive": True, "rows": rollout,
        "neural_controls_executed": 0, "causal_failure_attribution_not_established": True}
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}), flush=True)


if __name__ == "__main__": main()
