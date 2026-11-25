"""Export an explicit TRAIN-population min/max action-codec diagnostic.

No model fit is authorized and no production normalization asset is changed.
The official affine normalizer reads q01/q99 schema fields; here those fields
explicitly store 0th/100th percentile bounds, NOT 1st/99th percentiles. This is
an alternative encoding recipe to test against the original quantile recipe.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.source_payload_provenance import verify_source_payload_for_profile


def main():
    from openpi.shared import normalize
    from VLA.pi0.training import calculate_npz_stats
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, choices=(5, 10), default=5)
    parser.add_argument("--per-horizon-step", action="store_true",
                        help="Separate each future step's bounds; do not let large late deltas coarsen the deployed first action")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    audit = json.loads(args.source_audit.read_text())
    frozen = {str(path): sha256_file(path) for path in (source / "index.json", source / "dataset_manifest.json", args.source_audit)}
    if (not audit.get("passed") or audit.get("failures")
            or audit.get("index_sha256") != sha256_file(source / "index.json")
            or audit.get("manifest_sha256") != sha256_file(source / "dataset_manifest.json")):
        raise ValueError("Requires a bound, passed source audit")
    verify_source_payload_for_profile(audit, source)
    stats, frames = calculate_npz_stats(source, args.action_horizon, None)
    shape = (args.action_horizon, 7) if args.per_horizon_step else (7,)
    minimum, maximum = np.full(shape, np.inf), np.full(shape, -np.inf)
    action_chunks = []
    for row in json.loads((source / "index.json").read_text()):
        with np.load(source / row["file"], allow_pickle=False) as episode:
            joints = (episode["joint_positions"].astype(float) + np.pi) % (2*np.pi) - np.pi
            targets = (episode["next_joint_targets"].astype(float) + np.pi) % (2*np.pi) - np.pi
            absolute = np.c_[targets, native_gripper_closure(episode)]
            for frame in range(len(joints)):
                action = absolute[np.minimum(np.arange(frame, frame + args.action_horizon), len(joints)-1)].copy()
                action[:, :6] -= joints[frame]
                minimum = np.minimum(minimum, action if args.per_horizon_step else action.min(axis=0))
                maximum = np.maximum(maximum, action if args.per_horizon_step else action.max(axis=0))
                if args.per_horizon_step:
                    action_chunks.append(action)
    previous = stats["actions"]
    mean, std = previous.mean, previous.std
    if args.per_horizon_step:
        chunks = np.stack(action_chunks)
        mean, std = chunks.mean(axis=0), chunks.std(axis=0)
    stats["actions"] = normalize.NormStats(mean=mean, std=std, q01=minimum, q99=maximum)
    verify_source_payload_for_profile(audit, source)
    if any(sha256_file(Path(path)) != digest for path, digest in frozen.items()):
        raise RuntimeError("Calibration inputs changed")
    args.output.mkdir(parents=True)
    normalize.save(args.output, stats)
    (args.output / "calibration.json").write_text(json.dumps({
        "scope": "encoding_calibration_diagnostic_NOT_training_authorization",
        "source": str(source), "frozen_files": frozen, "frames": frames,
        "action_horizon": args.action_horizon, "action_bounds_percentiles": [0., 100.],
        "separate_bounds_per_horizon_step": args.per_horizon_step,
        "state_bounds_percentiles": [1., 99.], "action_minimum": minimum.tolist(),
        "action_maximum": maximum.tolist(), "neural_weights_loaded": False,
        "production_stats_modified": False, "norm_stats_sha256": sha256_file(args.output / "norm_stats.json")}, indent=2) + "\n")


if __name__ == "__main__":
    main()
