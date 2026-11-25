#!/usr/bin/env python3
"""Export FAST encode/decode targets for real, separate PyBullet replay.

This stage runs in the isolated openpi environment without importing PyBullet.
The exported absolute joint targets are replayed by audit_closed_loop_replay in
the simulator environment, so dependency isolation cannot skip the robot test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    from openpi.shared import normalize
    from VLA.pi0.fast_tokenizer import UR5FASTTokenizer
    from VLA.pi0.training import calculate_npz_stats

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--action-horizon", type=int, choices=(5, 10), default=10)
    parser.add_argument("--max-episodes-per-task", type=int)
    parser.add_argument("--tokenizer", default="physical-intelligence/fast")
    parser.add_argument("--codec-scale", type=float, default=10.0)
    parser.add_argument("--episode-files", nargs="+", help="Exact source filenames for diagnosis")
    args = parser.parse_args()
    source = args.source.resolve()
    audit = json.loads(args.source_audit.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile, source_payload_fingerprint, verify_source_payload_fingerprint
    from VLA.common.runtime_provenance import environment_runtime_fingerprint, verify_environment_runtime_fingerprint
    verify_source_payload_for_profile(audit, source)
    source_payload = source_payload_fingerprint(source)
    profile = json.loads((source / "dataset_manifest.json").read_text()).get("observation_profile")
    runtime_fingerprint = environment_runtime_fingerprint(profile)
    index_hash, manifest_hash = sha256(source / "index.json"), sha256(source / "dataset_manifest.json")
    if (not audit.get("passed") or audit.get("failures")
        or audit.get("index_sha256") != index_hash
        or audit.get("manifest_sha256") != manifest_hash
        or Path(audit.get("dataset", "")).resolve() != source):
        raise RuntimeError("FAST codec source audit is failed or stale")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.stats_path:
        stats = normalize.load(args.stats_path.parent)
        stats_path = args.stats_path
    else:
        stats, count = calculate_npz_stats(source, args.action_horizon, None)
        if count != audit["frames"]:
            raise RuntimeError("FAST codec normalization did not cover the full source")
        normalize.save(args.output, stats)
        stats_path = args.output / "norm_stats.json"
    token = UR5FASTTokenizer(180, fast_tokenizer_path=args.tokenizer, codec_scale=args.codec_scale)
    if not np.isfinite(args.codec_scale) or args.codec_scale <= 0:
        raise ValueError("codec_scale must be positive and finite")
    token._fast_tokenizer.scale = args.codec_scale
    records = json.loads((source / "index.json").read_text())
    counts = {}
    selected = []
    first_error = np.zeros(7)
    chunk_error = np.zeros(7)
    squared_error, values, truncations, zero_chunks = 0.0, 0, 0, 0
    rows = []
    clipped_coefficients = 0
    from scipy.fft import dct
    span = (np.asarray(stats["actions"].q99) - np.asarray(stats["actions"].q01) + 1e-6) / 2

    def normalize_value(value, key):
        s = stats[key]
        return (value - np.asarray(s.q01)) / (np.asarray(s.q99) - np.asarray(s.q01) + 1e-6) * 2 - 1

    for item in records:
        task = item["task"]["name"]
        if args.episode_files is not None and item["file"] not in args.episode_files:
            continue
        if args.max_episodes_per_task is not None and counts.get(task, 0) >= args.max_episodes_per_task:
            continue
        counts[task] = counts.get(task, 0) + 1
        with np.load(source / item["file"]) as episode:
            joints = ((episode["joint_positions"].astype(np.float32) + np.pi) % (2 * np.pi) - np.pi)
            targets = ((episode["next_joint_targets"].astype(np.float32) + np.pi) % (2 * np.pi) - np.pi)
            closures = native_gripper_closure(episode).astype(np.float32)
            state = np.c_[joints, episode["gripper_closure"].astype(np.float32)]
        absolute = np.c_[targets, closures]
        decoded_targets = []
        for frame in range(len(joints)):
            future = np.minimum(np.arange(frame, frame + args.action_horizon), len(joints) - 1)
            delta = absolute[future].copy()
            delta[:, :6] -= joints[frame]
            normalized = normalize_value(delta, "actions")
            coefficients = np.around(dct(normalized, axis=0, norm="ortho") * args.codec_scale)
            clipped_coefficients += int(np.count_nonzero(
                coefficients < token._fast_tokenizer.min_token
            ))
            ids, valid, _, _ = token.tokenize(item["task"]["prompt"], normalize_value(state[frame], "state"), normalized)
            truncations += int(bool(np.all(valid)))
            decoded = token.extract_actions(ids[valid], args.action_horizon, 7).astype(np.float64)
            zero_chunks += int(not np.any(decoded))
            error = (decoded - normalized) * span
            first_error = np.maximum(first_error, np.abs(error[0]))
            chunk_error = np.maximum(chunk_error, np.max(np.abs(error), axis=0))
            squared_error += float(np.sum(error * error)); values += error.size
            # Exactly openpi's quantile Unnormalize + AbsoluteActions.
            recovered = (decoded + 1) * span + np.asarray(stats["actions"].q01)
            recovered[:, :6] += joints[frame]
            decoded_targets.append(recovered[0])
        decoded_targets = np.asarray(decoded_targets, dtype=np.float32)
        np.savez_compressed(args.output / item["file"],
                            next_joint_targets=decoded_targets[:, :6],
                            next_gripper_closure=decoded_targets[:, 6])
        selected.append(item["file"])
        rows.append({"file": item["file"], "frames": len(joints)})
        print(f"codec {len(selected)} {item['file']}", flush=True)
    passed = bool(not truncations and not zero_chunks and not clipped_coefficients and values)
    verify_source_payload_fingerprint(source_payload, source)
    verify_environment_runtime_fingerprint(runtime_fingerprint, profile)
    codec_payload = source_payload_fingerprint(args.output, [{"file": name} for name in selected])
    report = {
        "passed": passed, "source": str(source), "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash, "source_audit_sha256": sha256(args.source_audit),
        "source_payload_fingerprint": source_payload,
        "codec_payload_fingerprint": codec_payload,
        "environment_runtime_fingerprint": runtime_fingerprint,
        "action_horizon": args.action_horizon, "stats_path": str(stats_path.resolve()),
        "stats_sha256": sha256(stats_path), "tokenizer": args.tokenizer,
        "tokenizer_source_sha256": sha256(Path(__file__).parent / "fast_tokenizer.py"),
        "all_episodes": len(selected) == len(records), "selected_files": selected,
        "codec_scale": args.codec_scale,
        "clipped_dct_coefficients": clipped_coefficients,
        "episodes": len(selected), "frames": sum(row["frames"] for row in rows),
        "truncated_chunks": truncations, "all_zero_decoded_chunks": zero_chunks,
        "first_action_max_abs_error_by_dim": first_error.tolist(),
        "chunk_max_abs_error_by_dim": chunk_error.tolist(),
        "physical_codec_rmse": float(np.sqrt(squared_error / max(values, 1))),
        "note": "Numeric codec report; completion is established by separate actual PyBullet replay",
    }
    (args.output / "codec_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "selected_files"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
