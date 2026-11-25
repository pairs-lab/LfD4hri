#!/usr/bin/env python3
"""Target-domain LoRA fine-tuning for official openpi pi0 / pi0-FAST.

This module keeps project-specific glue outside the upstream clone.  It uses
the official UR5 transform recipe, openpi's pinned LeRobot loader and training
loop, and the audited camera-v4 trajectories used by every VLA baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from VLA.common.native_action_fields import native_gripper_closure


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OPENPI_ROOT = PROJECT_ROOT / "VLA" / "third_party" / "openpi"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "VLA"
    / "datasets"
    / "pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_LEROBOT_HOME = PROJECT_ROOT / "VLA" / "datasets"
DEFAULT_AUDIT = PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "audit.json"
DEFAULT_ROUNDTRIP_AUDIT = (
    PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "openpi_roundtrip.json"
)
DEFAULT_ACTION_ADAPTER_AUDIT = (
    PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "pi0_action_adapter.json"
)
DEFAULT_CLOSED_LOOP_ADAPTER_AUDIT = (
    PROJECT_ROOT
    / "VLA"
    / "audits"
    / "camera_v4_vla_oneshot"
    / "pi0_closed_loop_replay.json"
)
CHECKPOINT_MANIFEST_NAME = "vla_checkpoint_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stats", "smoke", "train", "manifest"))
    parser.add_argument("--model", choices=("pi0", "pi0_fast"), required=True)
    parser.add_argument("--repo-id", default="openpi_lerobot_cam_v4_vla_oneshot")
    parser.add_argument(
        "--lerobot-home",
        type=Path,
        default=DEFAULT_LEROBOT_HOME,
        help="Local LeRobot cache root. The converted dataset must be at <root>/<repo-id>.",
    )
    parser.add_argument(
        "--steps", type=int, default=5000,
        help="Optimizer updates; actual loader microsteps = steps * gradient accumulation",
    )
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--color-safe-augmentation", action="store_true")
    parser.add_argument("--augmentation-report", type=Path,
                        help="Required qualified colour-safe image audit before a long colour-task fine-tune")
    parser.add_argument("--fast-codec-scale", type=float, default=10.0)
    parser.add_argument("--fast-tokenizer-path", default="physical-intelligence/fast")
    parser.add_argument(
        "--asset-id", default=None,
        help="Normalization asset id; use separate ids for h5 and h10 on the same dataset",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        choices=(5, 10),
        default=10,
        help="Native action chunk length. It is part of the checkpoint contract.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--exp-name", default="lora_5000")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Audited source NPZ dataset used to verify lineage and compute numeric statistics",
    )
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--roundtrip-report", type=Path, default=DEFAULT_ROUNDTRIP_AUDIT)
    parser.add_argument("--action-adapter-report", type=Path, default=DEFAULT_ACTION_ADAPTER_AUDIT)
    parser.add_argument(
        "--closed-loop-adapter-report",
        type=Path,
        default=DEFAULT_CLOSED_LOOP_ADAPTER_AUDIT,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory for mode=manifest.",
    )
    parser.add_argument("--manifest-overwrite", action="store_true")
    parser.add_argument(
        "--fast-codec-report", type=Path, default=None,
        help="Full FAST encode/decode PyBullet replay, required for FAST smoke/train/manifest",
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        default=None,
        help="JSON output for a one-update GPU smoke test (mode=smoke)",
    )
    return parser.parse_args()


def validate_lineage(args: argparse.Namespace, train_config, *, require_stats: bool) -> dict:
    """Bind the actual converted files and loader audit to the source split."""

    from VLA.common.provenance import verify_directory_fingerprint

    source = args.source_dataset.resolve()
    from VLA.common.source_training_gate import require_training_source
    if getattr(args, "mode", None) in {"train", "smoke"}:
        require_training_source(source, model=args.model, optimizer_updates=args.steps)
    converted = (args.lerobot_home / args.repo_id).resolve()
    required_source = (source / "index.json", source / "dataset_manifest.json")
    required_converted = (converted / "meta" / "info.json", converted / "conversion_manifest.json")
    action_adapter_report = getattr(
        args, "action_adapter_report", DEFAULT_ACTION_ADAPTER_AUDIT
    )
    closed_loop_adapter_report = getattr(
        args, "closed_loop_adapter_report", DEFAULT_CLOSED_LOOP_ADAPTER_AUDIT
    )
    for path in (
        *required_source,
        *required_converted,
        args.audit_report,
        args.roundtrip_report,
        action_adapter_report,
        closed_loop_adapter_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    index_hash = sha256(required_source[0])
    manifest_hash = sha256(required_source[1])
    index = json.loads(required_source[0].read_text())
    if not index or any(not item.get("success") for item in index):
        raise RuntimeError("Source dataset must contain only successful demonstrations")

    audit = json.loads(args.audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, source)
    if (
        not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("dataset", "")).resolve() != source
        or audit.get("index_sha256") != index_hash
        or audit.get("manifest_sha256") != manifest_hash
    ):
        raise RuntimeError("Camera/data audit does not bind to the current source dataset")

    conversion = json.loads(required_converted[1].read_text())
    info = json.loads(required_converted[0].read_text())
    if (
        conversion.get("version") != 2
        or Path(conversion.get("source", "")).resolve() != source
        or Path(conversion.get("converted", "")).resolve() != converted
        or conversion.get("source_index_sha256") != index_hash
        or conversion.get("source_manifest_sha256") != manifest_hash
        or conversion.get("episodes") != audit.get("episodes")
        or conversion.get("frames") != audit.get("frames")
        or info.get("total_episodes") != audit.get("episodes")
        or info.get("total_frames") != audit.get("frames")
        or info.get("fps") != 10
        or info.get("robot_type") != "ur5e_robotiq_2f85_pybullet"
    ):
        raise RuntimeError("Converted LeRobot metadata does not match the audited source")
    converted_fingerprint = conversion.get("converted_fingerprint", {})
    verified_conversion = verify_directory_fingerprint(converted, converted_fingerprint)

    roundtrip = json.loads(args.roundtrip_report.read_text())
    if (
        not roundtrip.get("passed")
        or roundtrip.get("failures")
        or roundtrip.get("episodes") != audit.get("episodes")
        or roundtrip.get("frames") != audit.get("frames")
        or roundtrip.get("joint_max_abs_error") != 0.0
        or roundtrip.get("gripper_max_abs_error") != 0.0
        or roundtrip.get("action_max_abs_error") != 0.0
        or roundtrip.get("front_rgb_max_pixel_error") != 0
        or roundtrip.get("wrist_rgb_max_pixel_error") != 0
        or roundtrip.get("prompt_mismatches") != 0
        or Path(roundtrip.get("source", "")).resolve() != source
        or Path(roundtrip.get("converted", "")).resolve() != converted
        or roundtrip.get("source_index_sha256") != index_hash
        or roundtrip.get("source_manifest_sha256") != manifest_hash
        or roundtrip.get("conversion_manifest_sha256") != sha256(required_converted[1])
        or roundtrip.get("converted_fingerprint", {}).get("content_tree_sha256")
        != verified_conversion["content_tree_sha256"]
        or roundtrip.get("converted_fingerprint", {}).get("layout_sha256")
        != verified_conversion["layout_sha256"]
        or roundtrip.get("loader_samples") != roundtrip.get("loader_expected_samples")
        or roundtrip.get("loader_samples", 0) < 1
        or any(roundtrip.get(key) != 0 for key in (
            "loader_joint_max_abs_error", "loader_gripper_max_abs_error",
            "loader_action_max_abs_error", "loader_front_rgb_max_pixel_error",
            "loader_wrist_rgb_max_pixel_error", "loader_prompt_mismatches",
            "index_mismatches",
        ))
    ):
        raise RuntimeError("Converted LeRobot round-trip audit is missing or no longer exact")

    adapter_audit = json.loads(action_adapter_report.read_text())
    from VLA.common.runtime_provenance import verify_source_environment_runtime
    verify_source_environment_runtime(adapter_audit, source)
    adapter_source = PROJECT_ROOT / "VLA/common/environment.py"
    if (
        not adapter_audit.get("passed")
        or Path(adapter_audit.get("source", "")).resolve() != source
        or adapter_audit.get("source_index_sha256") != index_hash
        or adapter_audit.get("source_manifest_sha256") != manifest_hash
        or adapter_audit.get("frames") != audit.get("frames")
        or Path(adapter_audit.get("adapter_source", "")).resolve() != adapter_source
        or adapter_audit.get("adapter_source_sha256") != sha256(adapter_source)
    ):
        raise RuntimeError("Native-joint to canonical-action adapter audit is stale or failed")

    closed_loop = json.loads(closed_loop_adapter_report.read_text())
    verify_source_environment_runtime(closed_loop, source)
    adapter_source = PROJECT_ROOT / "VLA/pi0/adapter.py"
    native_summary = closed_loop.get("summaries", {}).get("native", {})
    if (
        not closed_loop.get("passed")
        or closed_loop.get("failures")
        or Path(closed_loop.get("source", "")).resolve() != source
        or closed_loop.get("source_index_sha256") != index_hash
        or closed_loop.get("source_manifest_sha256") != manifest_hash
        or closed_loop.get("environment_sha256")
        != sha256(PROJECT_ROOT / "VLA/common/environment.py")
        or closed_loop.get("adapter_sha256") != sha256(adapter_source)
        or closed_loop.get("mode") != "native"
        or closed_loop.get("native_success_mode") != "semantic"
        or closed_loop.get("all_episodes") is not True
        or native_summary.get("episodes") != audit.get("episodes")
        or native_summary.get("successes") != audit.get("episodes")
        or native_summary.get("initial_false_positives") != 0
    ):
        raise RuntimeError(
            "Full closed-loop native-action replay is stale, incomplete, or failed"
        )

    fast_codec_hash = None
    if require_stats:
        asset_id = train_config.data.assets.asset_id or args.repo_id
        stats_dir = train_config.assets_dirs / asset_id
        stats_manifest_path = stats_dir / "stats_manifest.json"
        stats_path = stats_dir / "norm_stats.json"
        if not stats_manifest_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(
                f"Normalization assets are missing at {stats_dir}; run mode=stats first"
            )
        stats_manifest = json.loads(stats_manifest_path.read_text())
        if (
            Path(stats_manifest.get("source", "")).resolve() != source
            or stats_manifest.get("source_index_sha256") != index_hash
            or stats_manifest.get("source_manifest_sha256") != manifest_hash
            or stats_manifest.get("frames") != audit.get("frames")
            or stats_manifest.get("action_horizon") != train_config.model.action_horizon
            or stats_manifest.get("model_type") != str(train_config.model.model_type)
            or stats_manifest.get("accumulator_dtype") != "float64"
            or stats_manifest.get("statistics_method") != "exact_numpy"
            or stats_manifest.get("norm_stats_sha256") != sha256(stats_path)
            or min(
                float(stats_manifest.get("minimum_std", {}).get(key, 0.0))
                for key in ("state", "actions")
            )
            <= 1e-5
        ):
            raise RuntimeError("Normalization statistics do not match this model and dataset")
        if args.model == "pi0_fast":
            codec_report_path = getattr(args, "fast_codec_report", None)
            if codec_report_path is None or not codec_report_path.is_file():
                raise FileNotFoundError("FAST requires --fast-codec-report for the full codec replay")
            codec_report = json.loads(codec_report_path.read_text())
            codec_root = Path(codec_report.get("native_actions_root", ""))
            codec_manifest_path = codec_root / "codec_manifest.json"
            if not codec_manifest_path.is_file():
                raise FileNotFoundError(codec_manifest_path)
            codec_manifest = json.loads(codec_manifest_path.read_text())
            from VLA.pi0.codec_provenance import verify_fast_codec_binding
            verify_fast_codec_binding(codec_manifest, source, codec_root)
            verify_source_environment_runtime(codec_report, source)
            native_codec = codec_report.get("summaries", {}).get("native", {})
            # A lossy published action codec is not an exact float-label
            # adapter. Keep every residual failure visible, but require a
            # measured >=99.5% preservation overall AND for each task, plus
            # zero malformed/truncated/clamped encodings. The exact native
            # adapter above still requires 100% success. This is a setup
            # fidelity gate, never a learned-policy success-rate claim.
            minimum_codec_preservation = 0.995
            codec_by_task = native_codec.get("by_task", {})
            codec_preserved = (
                native_codec.get("episodes") == audit["episodes"]
                and float(native_codec.get("success_rate", 0.0)) >= minimum_codec_preservation
                and set(codec_by_task) == {"touch", "pick", "move", "pick_place"}
                and all(
                    float(value.get("success_rate") or 0.0) >= minimum_codec_preservation
                    for value in codec_by_task.values()
                )
            )
            if (
                not codec_preserved
                or codec_report.get("all_episodes") is not True
                or Path(codec_report.get("source", "")).resolve() != source
                or codec_report.get("source_index_sha256") != index_hash
                or codec_report.get("source_manifest_sha256") != manifest_hash
                or codec_report.get("environment_sha256") != sha256(PROJECT_ROOT / "VLA/common/environment.py")
                or codec_report.get("adapter_sha256") != sha256(PROJECT_ROOT / "VLA/pi0/adapter.py")
                or codec_report.get("mode") != "native"
                or codec_report.get("native_success_mode") != "semantic"
                or native_codec.get("episodes") != audit["episodes"]
                or native_codec.get("initial_false_positives") != 0
                or codec_report.get("codec_manifest_sha256") != sha256(codec_manifest_path)
                or codec_manifest.get("stats_sha256") != sha256(stats_path)
                or codec_manifest.get("action_horizon") != train_config.model.action_horizon
                or codec_manifest.get("truncated_chunks") != 0
                or codec_manifest.get("all_zero_decoded_chunks") != 0
                or codec_manifest.get("tokenizer_source_sha256") != sha256(PROJECT_ROOT / "VLA/pi0/fast_tokenizer.py")
                or codec_manifest.get("codec_scale", 10.0) != getattr(args, "fast_codec_scale", 10.0)
                or codec_manifest.get("clipped_dct_coefficients", 0) != 0
            ):
                raise RuntimeError("FAST codec-to-PyBullet replay is stale, incomplete, or failed")
            fast_codec_hash = sha256(codec_report_path)

    result = {
        "source": str(source),
        "converted": str(converted),
        "episodes": audit["episodes"],
        "frames": audit["frames"],
        "index_sha256": index_hash,
        "manifest_sha256": manifest_hash,
        "model": args.model,
        "action_horizon": train_config.model.action_horizon,
        "closed_loop_adapter_report_sha256": sha256(closed_loop_adapter_report),
        "conversion_manifest_sha256": sha256(required_converted[1]),
        "converted_content_tree_sha256": verified_conversion["content_tree_sha256"],
        "fast_codec_report_sha256": fast_codec_hash,
    }
    print("Validated training lineage:\n" + json.dumps(result, indent=2), flush=True)
    return result


def make_config(args: argparse.Namespace):
    from flax import nnx
    from openpi.models import model as model_api
    from openpi.models import pi0_config, pi0_fast
    from openpi.shared import nnx_utils
    from openpi.training import config, optimizer, weight_loaders
    import openpi.transforms as transforms
    from typing_extensions import override
    from VLA.pi0.fast_tokenizer import UR5FASTTokenizer
    from VLA.pi0.accumulation import GradientAccumulatingOptimizer

    @dataclasses.dataclass(frozen=True)
    class UR5Inputs(transforms.DataTransformFn):
        model_type: model_api.ModelType
        action_dim: int

        def __call__(self, data: dict) -> dict:
            state = np.concatenate(
                (
                    np.asarray(data["joints"], dtype=np.float32).reshape(6),
                    np.asarray(data["gripper"], dtype=np.float32).reshape(1),
                )
            )
            # Unlike pi0, FAST has no PadStatesAndActions model transform.
            if self.model_type == model_api.ModelType.PI0_FAST:
                state = np.pad(state, (0, self.action_dim - state.shape[0]))
            base = _parse_image(data["base_rgb"])
            wrist = _parse_image(data["wrist_rgb"])
            if self.model_type == model_api.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base, np.zeros_like(base), wrist)
                masks = (np.True_, np.True_, np.True_)
            else:
                # Do not run SigLIP on a synthetic right-wrist tensor.  The
                # official recipe masks that slot False, so removing those
                # already-invisible tokens preserves the two real modalities
                # while cutting a third of the vision encoding cost.
                names = ("base_0_rgb", "left_wrist_0_rgb")
                images = (base, wrist)
                masks = (np.True_, np.True_)
            output = {
                "state": state,
                "image": dict(zip(names, images, strict=True)),
                "image_mask": dict(zip(names, masks, strict=True)),
            }
            if "actions" in data:
                output["actions"] = np.asarray(data["actions"], dtype=np.float32)
            if "prompt" in data:
                output["prompt"] = data["prompt"]
            return output

    @dataclasses.dataclass(frozen=True)
    class UR5Outputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"])[..., :7]}

    @dataclasses.dataclass(frozen=True)
    class LeRobotUR5DataConfig(config.DataConfigFactory):
        @override
        def create(self, assets_dirs: Path, model_config: model_api.BaseModelConfig):
            repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "base_rgb": "image",
                            "wrist_rgb": "wrist_image",
                            "joints": "joints",
                            "gripper": "gripper",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = transforms.Group(
                inputs=[UR5Inputs(model_config.model_type, model_config.action_dim)],
                outputs=[UR5Outputs()],
            )
            delta_mask = transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[transforms.DeltaActions(delta_mask)],
                outputs=[transforms.AbsoluteActions(delta_mask)],
            )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=config.ModelTransformFactory()(model_config),
            )

    action_horizon = int(getattr(args, "action_horizon", 10))
    accumulation = int(getattr(args, "gradient_accumulation", 1))
    if accumulation < 1 or args.steps < 1:
        raise ValueError("steps and gradient_accumulation must be positive")
    if action_horizon not in (5, 10):
        raise ValueError("action_horizon must be 5 or 10")
    if args.model == "pi0":
        model = pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=action_horizon,
        )
        freeze_filter = nnx.Any(
            model.get_freeze_filter(),
            nnx_utils.PathRegex(".*img.*"),
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi0_base/params"
    else:
        model = pi0_fast.Pi0FASTConfig(
            action_dim=7,
            # The benchmark replans after one action.  A 10-step chunk covers
            # one second at 10 Hz, matches the official single-arm fine-tuning
            # examples, and avoids needless DCT/BPE error from compressing a
            # full 32-step manipulation trajectory into every target.
            action_horizon=action_horizon,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
            fast_model_tokenizer=UR5FASTTokenizer,
            fast_model_tokenizer_kwargs={
                "codec_scale": float(getattr(args, "fast_codec_scale", 10.0)),
                "fast_tokenizer_path": getattr(args, "fast_tokenizer_path", "physical-intelligence/fast"),
            },
        )
        freeze_filter = nnx.Any(
            model.get_freeze_filter(),
            nnx_utils.PathRegex(".*img.*"),
        )
        checkpoint = "gs://openpi-assets/checkpoints/pi0_fast_base/params"

    name = f"{args.model}_pybullet_ur5e_robotiq"
    return config.TrainConfig(
        name=name,
        exp_name=args.exp_name,
        model=model,
        data=LeRobotUR5DataConfig(
            repo_id=args.repo_id,
            assets=config.AssetsConfig(
                asset_id=getattr(args, "asset_id", None) or f"{args.repo_id}_h{action_horizon}"
            ),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint),
        # Keep the useful cosine portion aligned with longer balanced-dataset
        # runs.  The upstream default ends at 30k; silently reusing it for a
        # 60k/90k run would train the unseen half of the data only at the
        # terminal learning rate.  Short smoke/dev runs retain the official
        # 30k schedule.
        lr_schedule=optimizer.CosineDecaySchedule(
            decay_steps=max(int(args.steps), 30_000)
        ),
        optimizer=GradientAccumulatingOptimizer(optimizer.AdamW(), steps=accumulation),
        num_train_steps=args.steps * accumulation,
        batch_size=args.batch_size,
        num_workers=0,
        freeze_filter=freeze_filter,
        ema_decay=None,
        assets_base_dir=str(PROJECT_ROOT / "VLA" / "pi0" / "assets"),
        checkpoint_base_dir=str(PROJECT_ROOT / "VLA" / "pi0" / "finetune"),
        log_interval=100,
        save_interval=getattr(args, "save_interval", 1000) * accumulation,
        keep_period=getattr(args, "save_interval", 1000) * accumulation,
        wandb_enabled=False,
        overwrite=args.overwrite,
        resume=getattr(args, "resume", False),
        policy_metadata={
            "robot": "ur5e",
            "gripper": "Robotiq 2F-85",
            "control_hz": 10,
            "action_horizon": model.action_horizon,
            "gradient_accumulation": accumulation,
            "effective_batch_size": args.batch_size * accumulation,
            "peft": "LoRA plus action heads; frozen pretrained SigLIP backbone",
        },
    )


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def compute_stats(train_config, max_frames: int | None) -> None:
    from openpi.shared import normalize
    from openpi.training import data_loader
    import openpi.transforms as transforms

    class RemoveStrings(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {
                key: value
                for key, value in data.items()
                if not np.issubdtype(np.asarray(value).dtype, np.str_)
            }

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = data_loader.create_torch_dataset(
        data_config, train_config.model.action_horizon, train_config.model
    )
    dataset = data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    frame_limit = len(dataset) if max_frames is None else min(max_frames, len(dataset))
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for index in range(frame_limit):
        sample = dataset[index]
        for key, running in stats.items():
            # RunningStats uses E[x^2] - E[x]^2.  Accumulating float32 can
            # catastrophically cancel low-variance UR5 joints and report a
            # zero standard deviation, which then explodes z-normalization.
            running.update(np.asarray(sample[key], dtype=np.float64))
        if (index + 1) % 500 == 0 or index + 1 == frame_limit:
            print(f"stats {index + 1}/{frame_limit}", flush=True)
    result = {key: running.get_statistics() for key, running in stats.items()}
    output = train_config.assets_dirs / data_config.repo_id
    normalize.save(output, result)
    print(f"Wrote normalization statistics to {output}")


def calculate_npz_stats(source: Path, action_horizon: int, max_frames: int | None):
    """Reproduce openpi normalization without unnecessarily decoding RGB.

    The pinned LeRobot loader clips out-of-episode future indices to the final
    frame.  DeltaActions then subtracts the current state from every absolute
    joint target in that future chunk.  This implements those two operations
    exactly and feeds the same grouped arrays to openpi's RunningStats.
    """
    index = json.loads((source / "index.json").read_text())
    failed = [item["file"] for item in index if not item.get("success")]
    if failed:
        raise RuntimeError(f"Refusing failed demonstrations: {failed[:10]}")
    # The dataset is small enough to compute exact statistics in memory.  Do
    # not use openpi.RunningStats here: its E[x^2]-E[x]^2 variance loses
    # precision on low-variance joints, and repeated histogram rebinning can
    # move nominal 1st/99th percentiles far away from the true quantiles.
    state_values: list[np.ndarray] = []
    action_values: list[np.ndarray] = []
    processed = 0

    def finish():
        from openpi.shared import normalize

        arrays = {
            "state": np.concatenate(state_values, axis=0),
            "actions": np.concatenate(action_values, axis=0),
        }
        return {
            key: normalize.NormStats(
                mean=np.mean(value, axis=0, dtype=np.float64),
                std=np.std(value, axis=0, dtype=np.float64),
                q01=np.quantile(value, 0.01, axis=0),
                q99=np.quantile(value, 0.99, axis=0),
            )
            for key, value in arrays.items()
        }

    for item in index:
        with np.load(source / item["file"]) as episode:
            # Accumulate statistics in float64.  openpi's RunningStats derives
            # variance from E[x^2] - E[x]^2; float32 made the nearly-fixed
            # wrist joint's nonzero ~2.5e-3 rad std collapse to exactly zero.
            joints = (
                (episode["joint_positions"].astype(np.float64) + np.pi) % (2 * np.pi)
                - np.pi
            )
            next_joints = (
                (episode["next_joint_targets"].astype(np.float64) + np.pi) % (2 * np.pi)
                - np.pi
            )
            state = np.concatenate(
                (joints, episode["gripper_closure"].astype(np.float64)[:, None]), axis=1
            )
            absolute_actions = np.concatenate(
                (
                    next_joints,
                    native_gripper_closure(episode).astype(np.float64)[:, None],
                ),
                axis=1,
            )
            length = len(state)
            for frame in range(length):
                if max_frames is not None and processed >= max_frames:
                    return finish(), processed
                future = np.minimum(np.arange(frame, frame + action_horizon), length - 1)
                actions = absolute_actions[future].copy()
                actions[:, :6] -= state[frame, :6]
                state_values.append(state[frame : frame + 1])
                action_values.append(actions)
                processed += 1
                if processed % 500 == 0:
                    print(f"stats {processed}", flush=True)
    return finish(), processed


def compute_stats_from_npz(
    train_config, source: Path, max_frames: int | None, *, overwrite: bool = False
) -> None:
    from openpi.shared import normalize

    result, processed = calculate_npz_stats(source, train_config.model.action_horizon, max_frames)
    # Avoid constructing ModelTransformFactory here: FAST would download and
    # instantiate its tokenizer even though normalization only needs numeric
    # state/action arrays.
    repo_id = train_config.data.repo_id
    if repo_id is None:
        raise ValueError("repo_id is required for normalization assets")
    asset_id = train_config.data.assets.asset_id or repo_id
    output = train_config.assets_dirs / asset_id
    if (output / "norm_stats.json").exists() and not overwrite:
        raise FileExistsError(
            f"Normalization statistics already exist at {output}; pass --overwrite to replace them"
        )
    normalize.save(output, result)
    stats_path = output / "norm_stats.json"
    (output / "stats_manifest.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_index_sha256": sha256(source / "index.json"),
                "source_manifest_sha256": sha256(source / "dataset_manifest.json"),
                "frames": processed,
                "action_horizon": train_config.model.action_horizon,
                "asset_id": asset_id,
                "model_type": str(train_config.model.model_type),
                "accumulator_dtype": "float64",
                "statistics_method": "exact_numpy",
                "minimum_std": {
                    key: float(np.min(np.asarray(value.std))) for key, value in result.items()
                },
                "norm_stats_sha256": sha256(stats_path),
                "native_contract": "wrapped absolute UR5e joints converted to first-six deltas; absolute Robotiq closure",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote normalization statistics from {processed} frames to {output}")


def write_checkpoint_manifest(
    checkpoint: Path,
    args: argparse.Namespace,
    train_config,
    lineage: dict,
    *,
    overwrite: bool = False,
) -> Path:
    """Bind an otherwise config-less openpi checkpoint to its VLA contract."""

    checkpoint = checkpoint.resolve()
    if not checkpoint.is_dir() or not (checkpoint / "_CHECKPOINT_METADATA").is_file():
        raise FileNotFoundError(f"Not an openpi checkpoint: {checkpoint}")
    asset_id = train_config.data.assets.asset_id or args.repo_id
    stats_dir = train_config.assets_dirs / asset_id
    source_stats = stats_dir / "norm_stats.json"
    stats_manifest = stats_dir / "stats_manifest.json"
    checkpoint_stats = checkpoint / "assets" / asset_id / "norm_stats.json"
    for path in (source_stats, stats_manifest, checkpoint_stats):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_stats_hash = sha256(source_stats)
    if sha256(checkpoint_stats) != source_stats_hash:
        raise RuntimeError("Checkpoint normalization assets differ from audited training assets")

    camera_slots = (
        ["base_0_rgb", "left_wrist_0_rgb"]
        if args.model == "pi0"
        else ["base_0_rgb", "base_1_rgb_zero_padding", "wrist_0_rgb"]
    )
    tokenizer_path = PROJECT_ROOT / "VLA/pi0/fast_tokenizer.py"
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "model": args.model,
        "model_type": str(train_config.model.model_type),
        "action_horizon": int(train_config.model.action_horizon),
        "action_dim": int(train_config.model.action_dim),
        "model_config": {
            key: getattr(train_config.model, key, None)
            for key in ("paligemma_variant", "action_expert_variant", "max_token_len", "fast_model_tokenizer_kwargs")
        },
        "camera_slots": camera_slots,
        "repo_id": args.repo_id,
        "asset_id": asset_id,
        "control_hz": 10,
        "native_action_contract": (
            "wrapped absolute UR5e q0..q5 targets plus Robotiq closure 0=open,1=closed; "
            "first six are delta-normalized internally"
        ),
        "source": lineage,
        "audit_reports": {
            "data": {"path": str(args.audit_report.resolve()), "sha256": sha256(args.audit_report)},
            "roundtrip": {
                "path": str(args.roundtrip_report.resolve()),
                "sha256": sha256(args.roundtrip_report),
            },
            "action_adapter": {
                "path": str(args.action_adapter_report.resolve()),
                "sha256": sha256(args.action_adapter_report),
            },
            "closed_loop_adapter": {
                "path": str(args.closed_loop_adapter_report.resolve()),
                "sha256": sha256(args.closed_loop_adapter_report),
            },
        },
        "normalization": {
            "stats_manifest": str(stats_manifest.resolve()),
            "stats_manifest_sha256": sha256(stats_manifest),
            "norm_stats_sha256": source_stats_hash,
            "checkpoint_norm_stats": str(checkpoint_stats),
        },
        "interpretation_code": {
            "training_sha256": sha256(Path(__file__)),
            "adapter_sha256": sha256(PROJECT_ROOT / "VLA/pi0/adapter.py"),
            "fast_tokenizer_sha256": (
                sha256(tokenizer_path) if args.model == "pi0_fast" else None
            ),
        },
        "optimization": {
            "requested_optimizer_updates": int(args.steps),
            "requested_microsteps": int(train_config.num_train_steps),
            "checkpoint_label": int(checkpoint.name),
            "completed_microsteps": int(checkpoint.name) + 1,
            "completed_optimizer_updates": (int(checkpoint.name) + 1)
            // int(train_config.optimizer.steps),
            "partial_accumulation_microsteps": (int(checkpoint.name) + 1)
            % int(train_config.optimizer.steps),
            "microbatch_size": int(args.batch_size),
            "gradient_accumulation": int(train_config.optimizer.steps),
            "effective_batch_size": int(args.batch_size) * int(train_config.optimizer.steps),
            "lr_schedule": dataclasses.asdict(train_config.lr_schedule),
        },
    }
    output = checkpoint / CHECKPOINT_MANIFEST_NAME
    if args.model == "pi0_fast":
        manifest["audit_reports"]["fast_codec"] = {
            "path": str(args.fast_codec_report.resolve()),
            "sha256": sha256(args.fast_codec_report),
        }
    if output.exists() and not overwrite:
        existing = json.loads(output.read_text())
        if {k: v for k, v in existing.items() if k != "created_at"} == {
            k: v for k, v in manifest.items() if k != "created_at"
        }:
            return output
        raise RuntimeError(f"Existing checkpoint contract conflicts with this configuration: {output}")
    if (checkpoint / "benchmark_fingerprint.json").exists():
        raise RuntimeError(
            "This checkpoint is already fingerprinted. Its contract must be written "
            "before final fingerprinting; finalized checkpoints are immutable."
        )
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote checkpoint contract to {output}", flush=True)
    return output


def write_experiment_manifests(args, train_config, lineage: dict) -> list[Path]:
    experiment = (
        Path(train_config.checkpoint_base_dir)
        / train_config.name
        / train_config.exp_name
    )
    checkpoints = sorted(
        path.parent for path in experiment.glob("*/_CHECKPOINT_METADATA")
    )
    if not checkpoints:
        raise RuntimeError(f"Training returned without a saved checkpoint under {experiment}")
    outputs = []
    for checkpoint in checkpoints:
        if (checkpoint / CHECKPOINT_MANIFEST_NAME).exists() and not args.manifest_overwrite:
            continue
        outputs.append(
            write_checkpoint_manifest(
                checkpoint,
                args,
                train_config,
                lineage,
                overwrite=args.manifest_overwrite,
            )
        )
    return outputs


def run_train(train_config) -> None:
    path = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_official_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official training loop from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stock_init = module.init_train_state

    def audited_init(*args, **kwargs):
        state, state_sharding = stock_init(*args, **kwargs)
        print("Trainable master dtype audit: " + json.dumps(assert_fp32_training_state(state, train_config)), flush=True)
        return state, state_sharding

    module.init_train_state = audited_init
    module.main(train_config)


def assert_fp32_training_state(state, train_config) -> dict:
    import jax
    import jax.numpy as jnp

    master_dtypes, moment_dtypes = {}, {}
    for leaf in jax.tree_util.tree_leaves(state.params.filter(train_config.trainable_filter)):
        dtype = str(leaf.dtype)
        master_dtypes[dtype] = master_dtypes.get(dtype, 0) + int(np.prod(leaf.shape))
        if leaf.dtype != jnp.float32:
            raise RuntimeError(f"Openpi trainable master is not FP32: {dtype}")
    for leaf in jax.tree_util.tree_leaves(state.opt_state):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            dtype = str(leaf.dtype)
            moment_dtypes[dtype] = moment_dtypes.get(dtype, 0) + int(np.prod(leaf.shape))
            if leaf.dtype != jnp.float32:
                raise RuntimeError(f"Openpi floating optimizer state is not FP32: {dtype}")
    return {"passed": True, "trainable_master_dtype_counts": master_dtypes,
            "floating_optimizer_state_dtype_counts": moment_dtypes}


def validate_training_run_contract(args, train_config, lineage: dict) -> None:
    """Reject resume settings that reinterpret a shape-compatible checkpoint."""

    asset_id = train_config.data.assets.asset_id or args.repo_id
    contract = {
        "version": 1,
        "model": args.model,
        "model_config": {
            key: getattr(train_config.model, key, None)
            for key in (
                "action_horizon", "action_dim", "max_token_len",
                "paligemma_variant", "action_expert_variant",
                "fast_model_tokenizer_kwargs",
            )
        },
        "repo_id": args.repo_id,
        "asset_id": asset_id,
        "source_index_sha256": lineage["index_sha256"],
        "source_manifest_sha256": lineage["manifest_sha256"],
        "converted_content_tree_sha256": lineage["converted_content_tree_sha256"],
        "norm_stats_sha256": sha256(train_config.assets_dirs / asset_id / "norm_stats.json"),
        "gradient_accumulation": int(train_config.optimizer.steps),
        "microbatch_size": int(args.batch_size),
        "num_microsteps": int(train_config.num_train_steps),
        "augmentation_profile": lineage.get("augmentation_profile", {"profile": "upstream_default", "hue_jitter": .1}),
        "lr_schedule": dataclasses.asdict(train_config.lr_schedule),
        "fast_tokenizer_sha256": (
            sha256(PROJECT_ROOT / "VLA/pi0/fast_tokenizer.py")
            if args.model == "pi0_fast" else None
        ),
    }
    experiment_parent = Path(train_config.checkpoint_base_dir) / train_config.name
    contract_path = experiment_parent / f"{train_config.exp_name}.vla_training_contract.json"
    if args.resume:
        if not contract_path.is_file() or json.loads(contract_path.read_text()) != contract:
            raise RuntimeError(
                "Resume contract is missing or changed (model/horizon/data/stats/optimizer). "
                "Use the original configuration for an exact continuation."
            )
    elif contract_path.exists() and not args.overwrite:
        raise FileExistsError(f"Training experiment contract already exists: {contract_path}")
    else:
        # Keep the sidecar outside the checkpoint experiment directory; the
        # official trainer expects that directory not to exist for a new run.
        experiment_parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")


def run_smoke(train_config, output: Path | None) -> None:
    """Run one real forward/backward/update without creating a checkpoint."""

    import jax
    import jax.numpy as jnp

    path = OPENPI_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("openpi_official_train_smoke", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official training loop from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if train_config.batch_size % jax.device_count() != 0:
        raise ValueError("Batch size must be divisible by JAX device count")
    rng = jax.random.key(train_config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = module.sharding.make_mesh(train_config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(module.sharding.DATA_AXIS)
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    accumulation = int(train_config.optimizer.steps)
    loader = module._data_loader.create_data_loader(
        train_config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=accumulation,
    )
    batches = iter(loader)
    batch = next(batches)
    state, state_sharding = module.init_train_state(
        train_config, init_rng, mesh, resume=False
    )
    jax.block_until_ready(state)
    master_dtype_audit = assert_fp32_training_state(state, train_config)
    trainable = state.params.filter(train_config.trainable_filter)
    trainable_parameters = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(trainable)
    )
    total_parameters = sum(
        int(np.prod(leaf.shape)) for leaf in jax.tree_util.tree_leaves(state.params)
    )
    ptrain_step = jax.jit(
        functools.partial(module.train_step, train_config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    # Hash the real trainable weights on CPU to prove an optimizer update is
    # emitted only after the full accumulation group (without doubling VRAM).
    def parameter_digest(current_state):
        digest = hashlib.sha256()
        leaves = jax.tree_util.tree_leaves(
            current_state.params.filter(train_config.trainable_filter)
        )
        for leaf in leaves:
            digest.update(np.asarray(jax.device_get(leaf)).tobytes())
        return digest.hexdigest()

    initial_digest = parameter_digest(state)
    premature_update = False
    new_state = state
    for microstep in range(accumulation):
        if microstep:
            batch = next(batches)
        new_state, info = ptrain_step(train_rng, new_state, batch)
        jax.block_until_ready(new_state)
        if microstep < accumulation - 1:
            premature_update |= parameter_digest(new_state) != initial_digest
    changed_after_group = parameter_digest(new_state) != initial_digest
    values = {key: float(value) for key, value in jax.device_get(info).items()}
    passed = (
        int(new_state.step) == accumulation
        and not premature_update
        and changed_after_group
        and all(np.isfinite(value) for value in values.values())
        and values["loss"] > 0.0
        and values["grad_norm"] > 0.0
    )
    report = {
        "passed": passed,
        "master_dtype_audit": master_dtype_audit,
        "model": str(train_config.model.model_type),
        "device": [str(device) for device in jax.devices()],
        "step_before": 0,
        "step_after": int(new_state.step),
        "gradient_accumulation": accumulation,
        "optimizer_updates": 1,
        "premature_parameter_update": premature_update,
        "parameters_changed_after_group": changed_after_group,
        "metrics": values,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "batch": {
            "state": list(batch[0].state.shape),
            "actions": list(batch[1].shape),
            "images": {key: list(value.shape) for key, value in batch[0].images.items()},
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise RuntimeError("GPU smoke update did not pass")


def main() -> None:
    args = parse_args()
    if args.mode == "train" and not args.color_safe_augmentation:
        raise ValueError("These task labels specify colours: qualify and select --color-safe-augmentation before fine-tuning; default hue jitter can contradict the instruction")
    if args.color_safe_augmentation:
        from VLA.pi0.augmentation import install_colour_safe_augmentation
        augmentation_profile = install_colour_safe_augmentation()
        augmentation_profile["patch_source_sha256"] = sha256(Path(__file__).with_name("augmentation.py"))
    else:
        augmentation_profile = {"profile": "upstream_default", "hue_jitter": .1}
    # LeRobot resolves local datasets from this variable at import time. Set it
    # before make_config or the official data loader can silently inspect the
    # user's global cache instead of this benchmark's immutable conversion.
    args.lerobot_home = args.lerobot_home.resolve()
    os.environ["HF_LEROBOT_HOME"] = str(args.lerobot_home)
    train_config = make_config(args)
    lineage = validate_lineage(
        args,
        train_config,
        require_stats=args.mode in {"smoke", "train", "manifest"},
    )
    lineage["augmentation_profile"] = augmentation_profile
    if args.mode == "train" and args.color_safe_augmentation:
        if args.augmentation_report is None:
            raise ValueError("Colour-safe training requires its actual augmentation audit")
        audit = json.loads(args.augmentation_report.read_text())
        if (not audit.get("passed") or audit.get("failures")
                or audit.get("source_index_sha256") != lineage["index_sha256"]
                or audit.get("source_manifest_sha256") != lineage["manifest_sha256"]
                or audit.get("augmentation_profile") != augmentation_profile):
            raise ValueError("Colour-safe augmentation audit is failed or stale")
    if args.mode == "stats":
        compute_stats_from_npz(
            train_config,
            args.source_dataset.resolve(),
            args.max_frames,
            overwrite=args.overwrite,
        )
    elif args.mode == "smoke":
        run_smoke(train_config, args.smoke_report)
    elif args.mode == "manifest":
        if args.checkpoint is None:
            raise ValueError("mode=manifest requires --checkpoint")
        write_checkpoint_manifest(
            args.checkpoint,
            args,
            train_config,
            lineage,
            overwrite=args.manifest_overwrite,
        )
    else:
        validate_training_run_contract(args, train_config, lineage)
        run_train(train_config)
        write_experiment_manifests(args, train_config, lineage)


if __name__ == "__main__":
    main()
