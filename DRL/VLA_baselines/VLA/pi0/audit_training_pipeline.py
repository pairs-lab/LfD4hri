#!/usr/bin/env python3
"""Audit the exact openpi training path, including numeric and token contracts.

This intentionally goes beyond an image-only preview.  It exercises the
official LeRobot loader, action chunking, UR5 transforms, normalization,
camera-slot masks, model resize, prompt/action tokenization, and final JAX
batch collation.  Audit-only segmentation masks choose difficult examples;
they are never exposed to the policy or stored in the converted dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TASK_NAMES = ("touch", "pick", "move", "pick_place")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("pi0", "pi0_fast"), required=True)
    parser.add_argument("--action-horizon", type=int, choices=(5, 10), default=10)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--fast-codec-report", type=Path, default=None)
    parser.add_argument("--fast-codec-scale", type=float, default=10.0)
    parser.add_argument("--fast-tokenizer-path", default="physical-intelligence/fast")
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            PROJECT_ROOT
            / "VLA/datasets/pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
        ),
    )
    parser.add_argument(
        "--lerobot-home", type=Path, default=PROJECT_ROOT / "VLA/datasets"
    )
    parser.add_argument("--repo-id", default="openpi_lerobot_cam_v4_vla_oneshot")
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=PROJECT_ROOT / "VLA/audits/camera_v4_vla_oneshot/audit.json",
    )
    parser.add_argument(
        "--roundtrip-report",
        type=Path,
        default=PROJECT_ROOT / "VLA/audits/camera_v4_vla_oneshot/openpi_roundtrip.json",
    )
    parser.add_argument(
        "--action-adapter-report",
        type=Path,
        default=PROJECT_ROOT / "VLA/audits/camera_v4_vla_oneshot/pi0_action_adapter.json",
    )
    parser.add_argument(
        "--closed-loop-adapter-report",
        type=Path,
        default=(
            PROJECT_ROOT
            / "VLA/audits/camera_v4_vla_oneshot/pi0_closed_loop_replay.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def wrap_joints(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def expected_numeric(episode, frame: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    state = np.concatenate(
        (
            wrap_joints(episode["joint_positions"][frame]),
            np.asarray((episode["gripper_closure"][frame],), dtype=np.float64),
        )
    )
    length = len(episode["joint_positions"])
    future = np.minimum(np.arange(frame, frame + horizon), length - 1)
    actions = np.concatenate(
        (
            wrap_joints(episode["next_joint_targets"][future]),
            native_gripper_closure(episode)[future, None].astype(np.float64),
        ),
        axis=1,
    )
    actions[:, :6] -= state[:6]
    return state, actions


def normalize_array(value: np.ndarray, stats, use_quantiles: bool) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if use_quantiles:
        q01 = np.asarray(stats.q01, dtype=np.float64)[..., : value.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float64)[..., : value.shape[-1]]
        return (value - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = np.asarray(stats.mean, dtype=np.float64)[..., : value.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float64)[..., : value.shape[-1]]
    return (value - mean) / (std + 1e-6)


def difficult_samples(source: Path, index: list[dict]) -> list[dict]:
    offsets = np.cumsum([0, *[int(item["steps"]) for item in index[:-1]]]).tolist()
    selected: list[dict] = []
    for task_name in TASK_NAMES:
        candidates: list[tuple[int, int, dict]] = []
        for episode_index, item in enumerate(index):
            if item["task"]["name"] != task_name:
                continue
            with np.load(source / item["file"]) as episode:
                candidates.append((int(episode["front_pick_mask"][0].sum()), episode_index, item))
        _, episode_index, item = min(candidates, key=lambda value: value[0])
        for phase, frame in (
            ("start", 0),
            ("middle", int(item["steps"]) // 2),
            ("success", int(item["steps"]) - 1),
        ):
            selected.append(
                {
                    "task": task_name,
                    "phase": phase,
                    "episode_index": episode_index,
                    "frame": frame,
                    "global_index": offsets[episode_index] + frame,
                    "file": item["file"],
                    "pick_name": item["task"]["pick_name"],
                }
            )
    return selected


def uint8_image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype == np.uint8:
        return value
    if value.min() >= -1.0001 and value.max() <= 1.0001:
        value = (value + 1.0) * 127.5
    return np.rint(np.clip(value, 0.0, 255.0)).astype(np.uint8)


def main() -> int:
    args = parse_args()
    args.source = args.source.resolve()
    args.lerobot_home = args.lerobot_home.resolve()
    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home)

    from openpi import transforms
    from openpi_client import image_tools
    from openpi.training import data_loader

    from VLA.pi0.training import make_config, validate_lineage

    config_args = SimpleNamespace(
        model=args.model,
        action_horizon=args.action_horizon,
        asset_id=args.asset_id,
        fast_codec_report=args.fast_codec_report,
        fast_codec_scale=args.fast_codec_scale,
        fast_tokenizer_path=args.fast_tokenizer_path,
        repo_id=args.repo_id,
        steps=1,
        batch_size=1,
        exp_name="pipeline_audit",
        max_frames=None,
        overwrite=False,
        source_dataset=args.source,
        lerobot_home=args.lerobot_home,
        audit_report=args.audit_report,
        roundtrip_report=args.roundtrip_report,
        action_adapter_report=args.action_adapter_report,
        closed_loop_adapter_report=args.closed_loop_adapter_report,
    )
    train_config = make_config(config_args)
    lineage = validate_lineage(config_args, train_config, require_stats=True)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
    )
    pre_normalized = data_loader.TransformedDataset(
        raw_dataset,
        [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs],
    )
    normalized = data_loader.TransformedDataset(
        pre_normalized,
        [transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)],
    )
    model_dataset = data_loader.transform_dataset(raw_dataset, data_config)

    index = json.loads((args.source / "index.json").read_text())
    selected = difficult_samples(args.source, index)
    failures: list[str] = []
    numeric_error = 0.0
    normalized_error = 0.0
    resized_pixel_error = 0
    cells: list[Image.Image] = []
    selected_report: list[dict] = []
    if args.model == "pi0":
        real_keys = ("base_0_rgb", "left_wrist_0_rgb")
        padding_key = None
        expected_masks = {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
        }
        expected_state_dim = 32
        expected_action_dim = 32
    else:
        real_keys = ("base_0_rgb", "wrist_0_rgb")
        padding_key = "base_1_rgb"
        expected_masks = {"base_0_rgb": True, "base_1_rgb": True, "wrist_0_rgb": True}
        expected_state_dim = 7
        expected_action_dim = 7

    for selection in selected:
        with np.load(args.source / selection["file"]) as episode:
            expected_state, expected_actions = expected_numeric(
                episode, selection["frame"], train_config.model.action_horizon
            )
            expected_images = (
                np.asarray(
                    image_tools.resize_with_pad(
                        episode["front_rgb"][selection["frame"]], 224, 224
                    )
                ),
                np.asarray(
                    image_tools.resize_with_pad(
                        episode["wrist_rgb"][selection["frame"]], 224, 224
                    )
                ),
            )
        pre = pre_normalized[selection["global_index"]]
        norm = normalized[selection["global_index"]]
        model_item = model_dataset[selection["global_index"]]
        numeric_error = max(
            numeric_error,
            float(np.max(np.abs(np.asarray(pre["state"]) - expected_state))),
            float(np.max(np.abs(np.asarray(pre["actions"]) - expected_actions))),
        )
        # Compare Normalize against the exact float32 values emitted by the
        # LeRobot/UR5 transform.  The independent source comparison above
        # already bounds conversion error; reusing float64 source values here
        # would amplify harmless sub-microradian rounding on low-variance joints.
        expected_norm_state = normalize_array(
            np.asarray(pre["state"]),
            data_config.norm_stats["state"],
            data_config.use_quantile_norm,
        )
        expected_norm_actions = normalize_array(
            np.asarray(pre["actions"]),
            data_config.norm_stats["actions"],
            data_config.use_quantile_norm,
        )
        normalized_error = max(
            normalized_error,
            float(np.max(np.abs(np.asarray(norm["state"]) - expected_norm_state))),
            float(np.max(np.abs(np.asarray(norm["actions"]) - expected_norm_actions))),
        )
        masks = {key: bool(value) for key, value in model_item["image_mask"].items()}
        if masks != expected_masks:
            failures.append(f"{selection['file']}:{selection['frame']} camera masks {masks}")
        if tuple(np.asarray(model_item["state"]).shape) != (expected_state_dim,):
            failures.append(f"{selection['file']}:{selection['frame']} state shape")
        if tuple(np.asarray(model_item["actions"]).shape) != (
            train_config.model.action_horizon,
            expected_action_dim,
        ):
            failures.append(f"{selection['file']}:{selection['frame']} action shape")
        if not all(
            np.isfinite(np.asarray(value)).all()
            for value in (
                model_item["state"],
                model_item["actions"],
                model_item["tokenized_prompt"],
            )
        ):
            failures.append(f"{selection['file']}:{selection['frame']} non-finite model input")
        output_images = [np.asarray(model_item["image"][key]) for key in real_keys]
        for actual, expected in zip(output_images, expected_images, strict=True):
            resized_pixel_error = max(
                resized_pixel_error,
                int(np.max(np.abs(actual.astype(np.int16) - expected.astype(np.int16)))),
            )
        if padding_key is None:
            padding_image = np.zeros_like(output_images[0])
            if set(model_item["image"]) != set(real_keys):
                failures.append(
                    f"{selection['file']}:{selection['frame']} unexpected synthetic camera"
                )
        else:
            padding_image = np.asarray(model_item["image"][padding_key])
            if np.any(padding_image):
                failures.append(f"{selection['file']}:{selection['frame']} nonzero padding camera")
        if args.model == "pi0":
            if np.any(np.asarray(model_item["state"])[7:]) or np.any(
                np.asarray(model_item["actions"])[..., 7:]
            ):
                failures.append(f"{selection['file']}:{selection['frame']} nonzero numeric padding")

        views = [*output_images, padding_image]
        cell = Image.new("RGB", (672, 248), "black")
        labels = (
            *real_keys,
            "right_wrist_0_rgb (omitted)"
            if padding_key is None
            else f"{padding_key} (padding)",
        )
        for view_index, (view, label) in enumerate(zip(views, labels, strict=True)):
            cell.paste(Image.fromarray(uint8_image(view)), (view_index * 224, 24))
            ImageDraw.Draw(cell).text((view_index * 224 + 4, 5), label, fill="white")
        ImageDraw.Draw(cell).text(
            (4, 228),
            f"{selection['task']} | {selection['phase']} | {selection['pick_name']}",
            fill="white",
        )
        cells.append(cell)
        selected_report.append(
            {
                **selection,
                "camera_masks": masks,
                "state_shape": list(np.asarray(model_item["state"]).shape),
                "action_shape": list(np.asarray(model_item["actions"]).shape),
                "prompt_tokens": int(np.asarray(model_item["tokenized_prompt_mask"]).sum()),
            }
        )

    # Audit every frame numerically without repeatedly decoding RGB.  This is
    # also the exact FAST tokenizer used by ModelTransformFactory.
    token_transform = next(
        (
            item
            for item in data_config.model_transforms.inputs
            if isinstance(item, transforms.TokenizeFASTInputs)
        ),
        None,
    )
    total_state_values = 0
    total_action_values = 0
    state_outside = 0
    action_outside = 0
    max_abs_state = 0.0
    max_abs_action = 0.0
    state_outside_by_dim = np.zeros(7, dtype=np.int64)
    action_outside_by_dim = np.zeros(7, dtype=np.int64)
    max_abs_state_by_dim = np.zeros(7, dtype=np.float64)
    max_abs_action_by_dim = np.zeros(7, dtype=np.float64)
    max_raw_joint_delta = 0.0
    token_valid_min = 10**9
    token_valid_max = 0
    token_loss_min = 10**9
    token_loss_max = 0
    token_truncations = 0
    codec_squared_error = 0.0
    codec_values = 0
    codec_max_error = 0.0
    codec_physical_squared_error = 0.0
    codec_physical_max_per_dim = np.zeros(7, dtype=np.float64)
    codec_physical_first_step_max_per_dim = np.zeros(7, dtype=np.float64)
    codec_worst: dict | None = None
    frames = 0
    for item in index:
        with np.load(args.source / item["file"]) as episode:
            prompt = str(episode["prompt"].item())
            for frame in range(int(item["steps"])):
                state, actions = expected_numeric(
                    episode, frame, train_config.model.action_horizon
                )
                max_raw_joint_delta = max(
                    max_raw_joint_delta, float(np.max(np.abs(actions[:, :6])))
                )
                norm_state = normalize_array(
                    state, data_config.norm_stats["state"], data_config.use_quantile_norm
                )
                norm_actions = normalize_array(
                    actions, data_config.norm_stats["actions"], data_config.use_quantile_norm
                )
                limit = 1.0 if data_config.use_quantile_norm else 3.0
                state_outside += int(np.count_nonzero(np.abs(norm_state) > limit))
                action_outside += int(np.count_nonzero(np.abs(norm_actions) > limit))
                state_outside_by_dim += np.count_nonzero(
                    np.abs(norm_state[..., :7]) > limit, axis=tuple(range(norm_state.ndim - 1))
                )
                action_outside_by_dim += np.count_nonzero(
                    np.abs(norm_actions[..., :7]) > limit,
                    axis=tuple(range(norm_actions.ndim - 1)),
                )
                max_abs_state_by_dim = np.maximum(
                    max_abs_state_by_dim,
                    np.max(np.abs(norm_state[..., :7]), axis=tuple(range(norm_state.ndim - 1))),
                )
                max_abs_action_by_dim = np.maximum(
                    max_abs_action_by_dim,
                    np.max(np.abs(norm_actions[..., :7]), axis=tuple(range(norm_actions.ndim - 1))),
                )
                total_state_values += norm_state.size
                total_action_values += norm_actions.size
                max_abs_state = max(max_abs_state, float(np.max(np.abs(norm_state))))
                max_abs_action = max(max_abs_action, float(np.max(np.abs(norm_actions))))
                if token_transform is not None:
                    tokens, token_mask, _, loss_mask = token_transform.tokenizer.tokenize(
                        prompt, norm_state, norm_actions
                    )
                    valid = int(np.count_nonzero(token_mask))
                    loss = int(np.count_nonzero(loss_mask))
                    token_valid_min = min(token_valid_min, valid)
                    token_valid_max = max(token_valid_max, valid)
                    token_loss_min = min(token_loss_min, loss)
                    token_loss_max = max(token_loss_max, loss)
                    if bool(np.all(token_mask)):
                        token_truncations += 1
                    decoded = token_transform.tokenizer.extract_actions(
                        tokens[token_mask],
                        train_config.model.action_horizon,
                        train_config.model.action_dim,
                    )
                    error = np.asarray(decoded, dtype=np.float64) - norm_actions
                    codec_squared_error += float(np.sum(error**2))
                    codec_values += error.size
                    frame_codec_max = float(np.max(np.abs(error)))
                    if frame_codec_max > codec_max_error:
                        codec_max_error = frame_codec_max
                        worst_location = np.unravel_index(np.argmax(np.abs(error)), error.shape)
                        codec_worst = {
                            "file": item["file"],
                            "frame": frame,
                            "chunk_step": int(worst_location[0]),
                            "action_dim": int(worst_location[1]),
                            "normalized_error": float(error[worst_location]),
                        }
                    # Quantile normalization is affine, so convert codec error
                    # back to radians/closure units before judging its effect.
                    span = (
                        np.asarray(data_config.norm_stats["actions"].q99, dtype=np.float64)[:7]
                        - np.asarray(data_config.norm_stats["actions"].q01, dtype=np.float64)[:7]
                    ) / 2.0
                    physical_error = error * span
                    codec_physical_squared_error += float(np.sum(physical_error**2))
                    codec_physical_max_per_dim = np.maximum(
                        codec_physical_max_per_dim,
                        np.max(np.abs(physical_error), axis=0),
                    )
                    # Evaluation executes only the first action before
                    # replanning.  Report that stricter closed-loop contract
                    # separately without hiding errors later in the target.
                    codec_physical_first_step_max_per_dim = np.maximum(
                        codec_physical_first_step_max_per_dim,
                        np.abs(physical_error[0]),
                    )
                frames += 1

    if numeric_error > 1e-5:
        failures.append(f"pre-normalization numeric error {numeric_error}")
    if normalized_error > 1e-5:
        failures.append(f"normalization error {normalized_error}")
    if resized_pixel_error != 0:
        failures.append(f"model resize mismatch {resized_pixel_error} pixels")
    if max_raw_joint_delta >= np.pi:
        failures.append(f"wrapped-joint discontinuity {max_raw_joint_delta}")
    if token_transform is not None:
        codec_rmse = float(np.sqrt(codec_squared_error / codec_values))
        codec_physical_rmse = float(np.sqrt(codec_physical_squared_error / codec_values))
        if token_truncations:
            failures.append(f"FAST token truncation on {token_truncations} frames")
        if (
            codec_rmse > 0.05
            # FAST is intentionally lossy over the whole action chunk. Reject
            # any joint reconstruction error above five degrees, including
            # actions that are training targets but are not executed open-loop.
            or float(np.max(codec_physical_max_per_dim[:6])) > np.deg2rad(5.0)
            or float(codec_physical_max_per_dim[6]) > 0.1
        ):
            failures.append(
                "FAST codec error too large: "
                f"normalized_rmse={codec_rmse}, physical_max={codec_physical_max_per_dim.tolist()}"
            )
    else:
        codec_rmse = None
        codec_physical_rmse = None
        token_valid_min = token_valid_max = token_loss_min = token_loss_max = None

    # Exercise the final collator/conversion used by the training loop.
    loader = data_loader.create_data_loader(
        train_config, shuffle=False, num_batches=1, framework="jax"
    )
    batch_observation, batch_actions = next(iter(loader))
    batch = {
        "state_shape": list(batch_observation.state.shape),
        "state_dtype": str(batch_observation.state.dtype),
        "action_shape": list(batch_actions.shape),
        "action_dtype": str(batch_actions.dtype),
        "image_shapes": {key: list(value.shape) for key, value in batch_observation.images.items()},
        "image_dtypes": {key: str(value.dtype) for key, value in batch_observation.images.items()},
        "camera_masks": {
            key: np.asarray(value).astype(bool).tolist()
            for key, value in batch_observation.image_masks.items()
        },
    }
    if batch["state_dtype"] != "float32" or batch["action_dtype"] != "float32":
        failures.append(f"unexpected final batch dtype: {batch}")

    sheet = Image.new("RGB", (672 * 3, 248 * 4), "black")
    for cell_index, cell in enumerate(cells):
        sheet.paste(cell, ((cell_index % 3) * 672, (cell_index // 3) * 248))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    report = {
        "passed": not failures,
        "failures": failures,
        "model": args.model,
        "lineage": lineage,
        "use_quantile_norm": data_config.use_quantile_norm,
        "numeric_error": numeric_error,
        "normalized_error": normalized_error,
        "resized_pixel_error": resized_pixel_error,
        "frames_audited_numerically": frames,
        "max_raw_joint_delta_rad": max_raw_joint_delta,
        "normalized_distribution": {
            "outside_limit": 1.0 if data_config.use_quantile_norm else 3.0,
            "state_fraction_outside": state_outside / total_state_values,
            "action_fraction_outside": action_outside / total_action_values,
            "max_abs_state": max_abs_state,
            "max_abs_action": max_abs_action,
            "state_fraction_outside_by_dim": (
                state_outside_by_dim / frames
            ).tolist(),
            "action_fraction_outside_by_dim": (
                action_outside_by_dim / (frames * train_config.model.action_horizon)
            ).tolist(),
            "max_abs_state_by_dim": max_abs_state_by_dim.tolist(),
            "max_abs_action_by_dim": max_abs_action_by_dim.tolist(),
        },
        "fast_tokens": {
            "valid_min": token_valid_min,
            "valid_max": token_valid_max,
            "loss_min": token_loss_min,
            "loss_max": token_loss_max,
            "truncated_frames": token_truncations,
            "codec_rmse": codec_rmse,
            "codec_max_abs_error": codec_max_error if token_transform is not None else None,
            "codec_physical_rmse": codec_physical_rmse,
            "codec_physical_max_per_dim": (
                codec_physical_max_per_dim.tolist() if token_transform is not None else None
            ),
            "codec_physical_first_step_max_per_dim": (
                codec_physical_first_step_max_per_dim.tolist()
                if token_transform is not None
                else None
            ),
            "codec_pass_limits": (
                {
                    "normalized_rmse": 0.05,
                    "joint_error_rad": float(np.deg2rad(5.0)),
                    "gripper_error": 0.1,
                }
                if token_transform is not None
                else None
            ),
            "codec_worst": codec_worst,
        },
        "final_batch": batch,
        "selected_difficult_frames": selected_report,
        "contact_sheet": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
