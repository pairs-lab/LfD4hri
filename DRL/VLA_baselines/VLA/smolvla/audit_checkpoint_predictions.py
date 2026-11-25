"""Audit exact deployed SmolVLA predictions on audited expert observations.

This measures *in-sample imitation*, not held-out or closed-loop success.
It reuses the rollout adapter, including its checkpoint processors and noise.
No target coordinates or segmentation masks are provided to the model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from VLA.common.observation import VLAObservation
from VLA.common.native_action_fields import native_gripper_closure
from VLA.common.provenance import sha256_file
from VLA.common.tasks import TaskSpec
from VLA.fingerprint_checkpoint import verify_fingerprint
from VLA.smolvla.adapter import SmolVLAPolicyAdapter


def validate_saved_normalization(checkpoint: Path, source: Path, audit: dict, identity: dict) -> dict:
    """Check the processors actually deployed, not just the original data stats."""
    from safetensors import safe_open

    local_manifest = checkpoint / "vla_checkpoint_manifest.json"
    lineage_path = local_manifest if local_manifest.is_file() else checkpoint.parents[2] / "training_lineage.json"
    lineage = json.loads(lineage_path.read_text())
    if (Path(lineage["checkpoint"]).resolve() != checkpoint
            or Path(lineage["source"]).resolve() != source
            or lineage["source_index_sha256"] != audit["index_sha256"]
            or lineage["source_manifest_sha256"] != audit["manifest_sha256"]
            or lineage["frames"] != audit["frames"]):
        raise RuntimeError("Final checkpoint lineage does not match the audited source")
    records = {row["path"]: row["sha256"] for row in identity["records"]}
    if (records["config.json"] != lineage["checkpoint_config_sha256"]
            or records["model.safetensors"] != lineage["checkpoint_model_sha256"]):
        raise RuntimeError("Final weights/config differ from completed training lineage")
    stats_path = Path(lineage["dataset_root"]) / "meta/stats.json"
    if sha256_file(stats_path) != lineage["stats_sha256"]:
        raise RuntimeError("Training normalization statistics changed")
    stats = json.loads(stats_path.read_text())
    errors = {}
    for filename in (
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        with safe_open(checkpoint / filename, framework="np") as tensors:
            for feature in ("observation.state", "action"):
                for statistic in ("mean", "std", "min", "max"):
                    key = f"{feature}.{statistic}"
                    expected = np.asarray(stats[feature][statistic], dtype=np.float32)
                    actual = tensors.get_tensor(key)
                    if actual.shape != expected.shape or not np.array_equal(actual, expected):
                        raise RuntimeError(f"Saved processor stats mismatch: {filename}:{key}")
                    errors[f"{filename}:{key}"] = 0.0
    return {"passed": True, "stats_sha256": lineage["stats_sha256"], "max_abs_errors": errors}


def summarize(rows: list[dict]) -> dict:
    predicted = np.asarray([row["predicted"] for row in rows])
    expected = np.asarray([row["expected"] for row in rows])
    error = predicted - expected
    lengths = np.linalg.norm(predicted[:, :3], axis=1) * np.linalg.norm(expected[:, :3], axis=1)
    moving = lengths > 1e-8
    cosines = np.sum(predicted[moving, :3] * expected[moving, :3], axis=1) / lengths[moving]
    applied_agreement = None
    if all("commanded_closure_before_action" in row for row in rows):
        previous = np.array([1 if row["commanded_closure_before_action"] > .5 else -1 for row in rows])
        applied = np.where(predicted[:, 6] > .2, 1, np.where(predicted[:, 6] < -.2, -1, previous))
        expected_applied = np.where(expected[:, 6] > .2, 1, np.where(expected[:, 6] < -.2, -1, previous))
        applied_agreement = float(np.mean(applied == expected_applied))
    return {
        "samples": len(rows),
        "canonical_mae": np.abs(error).mean(axis=0).tolist(),
        "translation_axis_rmse_m": float(np.sqrt(np.square(error[:, :3]).mean())),
        "translation_vector_error_p50_p90_m": np.quantile(np.linalg.norm(error[:, :3], axis=1), [.5, .9]).tolist(),
        "gripper_sign_accuracy": float(np.mean(np.sign(predicted[:, 6]) == np.sign(expected[:, 6]))),
        "gripper_applied_command_agreement": applied_agreement,
        "gripper_deadband_fraction": float(np.mean(np.abs(predicted[:, 6]) <= .2)),
        "translation_cosine_mean": None if not len(cosines) else float(cosines.mean()),
        "translation_cosine_positive_fraction": None if not len(cosines) else float(np.mean(cosines > 0)),
        "moving_direction_samples": int(moving.sum()),
        "translation_would_clip_fraction": float(np.any(np.abs(predicted[:, :3]) > .05, axis=1).mean()),
    }


def gripper_diagnostic_fields(episode, item: dict, frame: int, *, body_v9: bool) -> dict:
    """Recover audit-only command history without replacing the model's FK input.

    A CLOSE request can remain active while the encoder reports a partially
    closed, object-filled hand. Deadband decisions therefore use request
    history, not aperture feedback. Legacy command-state diagnostics retain
    their historical interpretation.
    """
    measured = float(episode["gripper_closure"][frame])
    requested = measured
    if body_v9:
        if "next_requested_gripper_closure" not in episode:
            raise ValueError("V9 prediction diagnostics require requested gripper action labels")
        requests = native_gripper_closure(episode)
        if requests.ndim != 1 or len(requests) != len(episode["gripper_closure"]):
            raise ValueError("V9 requested gripper action labels must match the episode frames")
        requested = 0.0  # Canonical reset requests OPEN before any startup action.
        if frame:
            requested = float(requests[frame - 1])
        else:
            for startup in item.get("recovery_start_actions_not_training_labels", ()):
                action = np.asarray(startup, dtype=float)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("Invalid startup action in V9 diagnostic command history")
                if action[-1] > .2:
                    requested = 1.0
                elif action[-1] < -.2:
                    requested = 0.0
    closed = measured >= .95 if body_v9 else measured > .5
    return {
        "commanded_closure_before_action": requested,
        "measured_closure_before_action": measured,
        "gripper_diagnostic_state_contract": "encoder_fk_aperture_closure_v1" if body_v9 else "legacy_command_state",
        "closed_empty_audit_only": bool(closed and
            (frame == 0 or not episode["next_grasped_object"][frame - 1])),
    }


def validate_holdout(training_source, evaluation_source):
    from VLA.audit_training_readiness import physical_scene_key
    def groups(source):
        rows = json.loads((source / "index.json").read_text())
        return {physical_scene_key(row["task"]) for row in rows}
    train_manifest = json.loads((training_source / "dataset_manifest.json").read_text())
    eval_manifest = json.loads((evaluation_source / "dataset_manifest.json").read_text())
    if (not train_manifest.get("observation_profile_sha256")
            or train_manifest["observation_profile_sha256"] != eval_manifest.get("observation_profile_sha256")
            or groups(training_source) & groups(evaluation_source)):
        raise RuntimeError("Held-out prediction audit requires the same sensor/controller profile and disjoint physical scenes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes-per-task", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--file-prefix", default=None, help="Explicit qualified-source subset, e.g. corrective_")
    parser.add_argument("--normalization-source", type=Path,
                        help="Explicit training source for a held-out audit; NEVER normalize using validation statistics")
    parser.add_argument("--normalization-audit-report", type=Path)
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        raise ValueError("episodes-per-task must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.source.resolve()
    manifest = json.loads((source / "dataset_manifest.json").read_text())
    body_v9 = (manifest.get("observation_profile") or {}).get("gripper_controller") == "body_contact_v9"
    audit = json.loads(args.audit_report.read_text())
    if (not audit.get("passed") or audit.get("failures")
            or Path(audit.get("dataset", "")).resolve() != source
            or audit.get("index_sha256") != sha256_file(source / "index.json")
            or audit.get("manifest_sha256") != sha256_file(source / "dataset_manifest.json")):
        raise RuntimeError("Source audit is stale or failed")
    identity = verify_fingerprint(args.checkpoint.resolve())
    if bool(args.normalization_source) != bool(args.normalization_audit_report):
        raise ValueError("Held-out normalization requires both training source and its audit")
    normalization_source, normalization_audit = source, audit
    if args.normalization_source:
        normalization_source = args.normalization_source.resolve()
        normalization_audit = json.loads(args.normalization_audit_report.read_text())
        if (not normalization_audit.get("passed") or normalization_audit.get("failures")
                or Path(normalization_audit.get("dataset", "")).resolve() != normalization_source
                or normalization_audit.get("index_sha256") != sha256_file(normalization_source / "index.json")
                or normalization_audit.get("manifest_sha256") != sha256_file(normalization_source / "dataset_manifest.json")):
            raise RuntimeError("Training normalization source audit is stale or failed")
        validate_holdout(normalization_source, source)
    normalization = validate_saved_normalization(args.checkpoint.resolve(), normalization_source, normalization_audit, identity)
    grouped = defaultdict(list)
    for item in json.loads((source / "index.json").read_text()):
        if args.file_prefix is not None and not item["file"].startswith(args.file_prefix):
            continue
        grouped[item["task"]["name"]].append(item)
    if any(not grouped[name] for name in ("touch", "pick", "move", "pick_place")):
        raise ValueError("Diagnostic subset must contain all four tasks")
    # Model input remains the source's measured FK state for V9 (legacy uses
    # command state). Requested intent below is strictly a diagnostic label.
    sensor = SimpleNamespace(closure=0.0)
    sensor.normalized_gripper_closure = lambda: sensor.closure
    policy = SmolVLAPolicyAdapter(sensor, checkpoint=str(args.checkpoint.resolve()), seed=args.seed)
    rows = []
    for name in ("touch", "pick", "move", "pick_place"):
        episodes = grouped[name]
        for number in np.linspace(0, len(episodes) - 1, min(len(episodes), args.episodes_per_task), dtype=int):
            item = episodes[int(number)]
            with np.load(source / item["file"], allow_pickle=False) as episode:
                task = TaskSpec.from_dict(item["task"])
                indices = np.unique(np.rint(np.linspace(0, len(episode["canonical_actions"]) - 1, 4)).astype(int))
                for stage, frame in enumerate(indices):
                    gripper_fields = gripper_diagnostic_fields(episode, item, int(frame), body_v9=body_v9)
                    # Independent predictions: no preceding expert/future actions in the model queue.
                    policy.reset(task)
                    sensor.closure = float(episode["gripper_closure"][frame])
                    observation = VLAObservation(
                        front_rgb=episode["front_rgb"][frame], wrist_rgb=episode["wrist_rgb"][frame],
                        joint_positions=episode["joint_positions"][frame],
                        gripper_width=.085 * (1.0 - sensor.closure), gripper_open=float(episode["gripper_open"][frame]),
                        ee_position=episode["ee_position"][frame], ee_rotation_6d=episode["ee_rotation_6d"][frame],
                        prompt=str(episode["prompt"].item()), step=int(frame),
                    )
                    predicted = policy.predict(observation)
                    if predicted.shape != (7,) or not np.isfinite(predicted).all():
                        raise RuntimeError("Invalid deployed prediction")
                    rows.append({
                        "task": name, "episode": item["file"], "frame": int(frame), "stage": stage,
                        "prompt": observation.prompt, "predicted": predicted.tolist(),
                        "expected": episode["canonical_actions"][frame].tolist(),
                        **gripper_fields,
                    })
            print(f"{name} {item['file']} | {len(rows)} observations", flush=True)
    policy.close()
    report = {
        "measurement": "held_out_imitation_not_success_rate" if args.normalization_source else "in_sample_imitation_not_success_rate", "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_content_sha256": identity["content_tree_sha256"], "source": str(source),
        "source_index_sha256": sha256_file(source / "index.json"), "seed": args.seed,
        "source_file_prefix": args.file_prefix,
        "sensor_and_adapter": "exact deployed SmolVLAPolicyAdapter; source RGB, raw joints, "
            + ("measured encoder/FK gripper closure" if body_v9 else "legacy gripper command state") + ", text",
        "privileged_target_coordinates_consumed": False,
        "saved_normalization_audit": normalization,
        "normalization_source": str(normalization_source),
        "overall": summarize(rows),
        "by_task": {name: summarize([row for row in rows if row["task"] == name]) for name in grouped},
        "initial_frame": summarize([row for row in rows if row["frame"] == 0]),
        "closed_empty_frames": (summarize([row for row in rows if row["closed_empty_audit_only"]])
            if any(row["closed_empty_audit_only"] for row in rows) else None),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
