#!/usr/bin/env python3
"""Target-domain RDT-170M fine-tuning on audited UR5e/Robotiq demos.

The model, diffusion objective, SigLIP tower, state-vector layout and 64-step
action horizon come from the pinned official RDT checkout.  Project-specific
code is limited to the audited PyBullet data adapter and checkpoint provenance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from VLA.rdt.training_data import (
    ACTION_HORIZON,
    NUM_CAMERAS,
    RDTBatchCollator,
    PyBulletRDTDataset,
    sha256,
    validate_language_embeddings,
    validate_lineage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RDT_ROOT = PROJECT_ROOT / "VLA" / "third_party" / "rdt"
if str(RDT_ROOT) not in sys.path:
    sys.path.insert(0, str(RDT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "smoke", "train"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument(
        "--action-contract-report",
        type=Path,
        help="Exhaustive FK report from VLA.rdt.audit_action_contract",
    )
    parser.add_argument(
        "--language-embeddings",
        type=Path,
        help="T5 embedding dictionary produced by VLA.rdt.prepare_language",
    )
    parser.add_argument(
        "--language-manifest",
        type=Path,
        help="Defaults to <language-embeddings stem>.manifest.json",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "VLA" / "checkpoints" / "rdt-170m",
    )
    parser.add_argument(
        "--vision-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "VLA" / "checkpoints" / "siglip-so400m-patch14-384",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--action-horizon", type=int, choices=(5, 10, 16, 64), default=64)
    parser.add_argument(
        "--episode-batch-size",
        type=int,
        default=1,
        help="Episodes decoded per worker batch; effective frames are multiplied by samples-per-episode",
    )
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--signal-precision", choices=("bf16", "fp32_state_action_readout_latent_v1"), default="bf16",
                        help="Opt-in numeric adaptation; requires a separate smoke/parity/qualification before a new fit")
    parser.add_argument("--signal-qualification", type=Path,
                        help="SHA-bound actual two-update GPU, reload and four-task integration proof for the opt-in long fit")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=("adamw8bit", "adamw"), default="adamw8bit")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--inspect-episodes", type=int, default=2)
    parser.add_argument("--smoke-report", type=Path, default=None)
    return parser.parse_args()


def _language_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.language_embeddings is None:
        raise ValueError(f"mode={args.mode} requires --language-embeddings")
    manifest = args.language_manifest or args.language_embeddings.with_suffix(".manifest.json")
    return args.language_embeddings.resolve(), manifest.resolve()


def _make_dataset(args: argparse.Namespace) -> PyBulletRDTDataset:
    return PyBulletRDTDataset(
        args.source,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        max_episodes=args.max_episodes,
        action_horizon=args.action_horizon,
    )


def _inspect(args: argparse.Namespace, lineage: dict) -> int:
    dataset = PyBulletRDTDataset(
        args.source,
        samples_per_episode=args.samples_per_episode,
        seed=args.seed,
        max_episodes=min(args.inspect_episodes, len(json.loads((args.source / "index.json").read_text()))),
        action_horizon=args.action_horizon,
    )
    summaries = []
    for episode_id in range(len(dataset)):
        samples = dataset[episode_id]
        summaries.append(
            {
                "episode_file": samples.episode_file,
                "step_ids": samples.step_ids.tolist(),
                "images_shape": list(samples.images.shape),
                "states_shape": list(samples.states.shape),
                "actions_shape": list(samples.actions.shape),
                "valid_action_lengths": samples.action_valid_lengths.tolist(),
                "state_nonzero_indices": np.flatnonzero(samples.state_mask[0]).tolist(),
                "joint_min": float(samples.states[..., :6].min()),
                "joint_max": float(samples.states[..., :6].max()),
                "gripper_open_values": sorted(
                    set(float(value) for value in samples.states[..., 10].reshape(-1))
                ),
                "prompt": samples.prompts[0],
            }
        )
    report = {
        "mode": "inspect",
        "passed": True,
        "lineage": lineage,
        "samples_per_episode": args.samples_per_episode,
        "episodes_inspected": summaries,
        "policy_inputs": ["front_rgb", "wrist_rgb", "joint_positions", "gripper_open", "text"],
        "privileged_pose_consumed": False,
    }
    print(json.dumps(report, indent=2))
    return 0


def _load_training_inputs(args: argparse.Namespace, dataset: PyBulletRDTDataset):
    from transformers import SiglipImageProcessor

    embeddings_path, language_manifest = _language_paths(args)
    embeddings, language_metadata = validate_language_embeddings(
        embeddings_path,
        language_manifest,
        args.source.resolve() / "index.json",
        dataset.prompts,
    )
    processor = SiglipImageProcessor.from_pretrained(args.vision_checkpoint)
    collator = RDTBatchCollator(processor, embeddings)
    return collator, embeddings_path, language_manifest, language_metadata


def _smoke(args: argparse.Namespace, lineage: dict) -> int:
    dataset = _make_dataset(args)
    collator, embeddings_path, language_manifest, language_metadata = _load_training_inputs(
        args, dataset
    )
    episodes = [dataset[index] for index in range(min(args.inspect_episodes, len(dataset)))]
    batch = collator(episodes)
    expected_batch = len(episodes) * args.samples_per_episode
    checks = {
        "images": tuple(batch["images"].shape) == (expected_batch, 6, 3, 384, 384),
        "states": tuple(batch["states"].shape) == (expected_batch, 1, 128),
        "actions": tuple(batch["actions"].shape)
        == (expected_batch, args.action_horizon, 128),
        "state_mask": tuple(batch["state_elem_mask"].shape) == (expected_batch, 128),
        "language": batch["lang_tokens"].shape[0] == expected_batch
        and batch["lang_tokens"].shape[2] == 4096,
        "finite_images": bool(torch.isfinite(batch["images"]).all()),
        "finite_state_actions": bool(
            torch.isfinite(batch["states"]).all() and torch.isfinite(batch["actions"]).all()
        ),
        "control_hz": bool(torch.all(batch["ctrl_freqs"] == 10)),
        "padding_lengths": bool(
            torch.all((batch["action_valid_lengths"] >= 1) & (batch["action_valid_lengths"] <= args.action_horizon))
        ),
    }
    report = {
        "mode": "input_smoke",
        "passed": all(checks.values()),
        "checks": checks,
        "lineage": lineage,
        "language_embeddings": str(embeddings_path),
        "language_embeddings_sha256": sha256(embeddings_path),
        "language_manifest": str(language_manifest),
        "language_metadata": language_metadata,
        "batch": {
            key: list(batch[key].shape)
            for key in (
                "images",
                "states",
                "actions",
                "state_elem_mask",
                "lang_tokens",
                "lang_attn_mask",
            )
        },
        "prompts": list(batch["prompts"]),
        "episode_files": list(batch["episode_files"]),
        "step_ids": batch["step_ids"].tolist(),
        "note": "CPU input-contract smoke; no policy update was performed",
    }
    if args.smoke_report is not None:
        args.smoke_report.parent.mkdir(parents=True, exist_ok=True)
        args.smoke_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def _make_optimizer(args: argparse.Namespace, model: torch.nn.Module):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
    return torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )


def _target_contract(configuration: dict, *, step: int | None = None) -> dict:
    language = configuration["language_metadata"]
    lineage = configuration["lineage"]
    return {
        "format_version": 1,
        "model_family": "rdt",
        "base_model": "rdt-170m",
        "checkpoint_step": step,
        "control_hz": 10,
        "state_dim": 128,
        "action_horizon": configuration["action_horizon"],
        "joint": {
            "representation": "absolute_position_rad",
            "convention": "wrap_to_minus_pi_pi",
            "offset": [0.0] * 6,
            "arm_indices": list(range(6)),
        },
        "gripper": {
            "index": 10,
            "representation": "absolute_open_fraction",
            "open": 1.0,
            "closed": 0.0,
            "clip": [0.0, 1.0],
        },
        "valid_indices": [0, 1, 2, 3, 4, 5, 10],
        "scaling": {"state": "identity_SI", "action": "identity_SI"},
        "signal_precision": configuration.get("signal_precision", "bf16"),
        "cameras": {
            "image_size": 384,
            "history": 2,
            "count": 3,
            "slots": [
                "front_t-1",
                "right_wrist_t-1",
                "processor_mean_t-1",
                "front_t",
                "right_wrist_t",
                "processor_mean_t",
            ],
        },
        "language": {
            "encoder": language["model"],
            "embeddings_sha256": configuration["language_embeddings_sha256"],
            "manifest_sha256": configuration["language_manifest_sha256"],
            "prompt_count": language["prompt_count"],
            "source_index_sha256": language["source_index_sha256"],
        },
        "lineage": {
            "source_index_sha256": lineage["source_index_sha256"],
            "source_manifest_sha256": lineage["source_manifest_sha256"],
            "audit_report_sha256": lineage["audit_report_sha256"],
            "action_contract_report_sha256": lineage.get(
                "action_contract_report_sha256"
            ),
        },
        "adapter": {
            "native_output": "absolute_joints_plus_gripper_open",
            "lazy_replan_required": True,
        },
    }


def _save_checkpoint(
    model,
    optimizer,
    output: Path,
    step: int,
    epoch: int,
    configuration: dict,
) -> Path:
    checkpoint = output / "checkpoints" / f"step-{step:06d}"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite RDT checkpoint: {checkpoint}")
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint)
    contract_path = checkpoint / "pybullet_rdt_config.json"
    contract_path.write_text(
        json.dumps(_target_contract(configuration, step=step), indent=2) + "\n"
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        },
        checkpoint / "training_state.pt",
    )
    manifest = {
        **configuration,
        "step": step,
        "epoch": epoch,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_config_sha256": sha256(checkpoint / "config.json"),
        "model_weights_sha256": sha256(checkpoint / "pytorch_model.bin"),
        "target_contract_sha256": sha256(contract_path),
        "training_state_sha256": sha256(checkpoint / "training_state.pt"),
    }
    (checkpoint / "vla_checkpoint_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return checkpoint


def _train(args: argparse.Namespace, lineage: dict) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("RDT fine-tuning requires CUDA")
    if args.output is None:
        raise ValueError("mode=train requires --output")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite RDT output: {args.output}")
    if args.steps < 1 or args.save_interval < 1:
        raise ValueError("--steps and --save-interval must be positive")
    if args.episode_batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")

    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from models.rdt_runner import RDTRunner

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    dataset = _make_dataset(args)
    collator, embeddings_path, language_manifest, language_metadata = _load_training_inputs(
        args, dataset
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.episode_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        persistent_workers=False,
        pin_memory=True,
        generator=generator,
    )

    # Keep trainable master weights in float32.  Casting them to bf16 makes
    # small Adam updates round away near typical UR5 joint-valued weights;
    # autocast still provides bf16 compute without losing those updates.
    from VLA.rdt.checkpoint_loading import load_checkpoint
    model, horizon_adaptation = load_checkpoint(
        args.base_checkpoint, args.action_horizon, dtype=torch.float32
    )
    model = model.to(device).train()
    from VLA.rdt.signal_precision import PROFILE, install_fp32_signal_islands, signal_dtype
    precision_profile = getattr(args, "signal_precision", "bf16")
    input_signal_dtype = signal_dtype(precision_profile)
    if precision_profile == PROFILE:
        install_fp32_signal_islands(model)
    vision = SiglipVisionTower(str(args.vision_checkpoint), args=None).to(
        device, dtype=dtype
    ).eval()
    vision.requires_grad_(False)
    optimizer = _make_optimizer(args, model)

    args.output.mkdir(parents=True)
    configuration = {
        "model": "rdt-170m",
        "official_source_revision": "cd79363a1387e8f81c7724d070ef7e45fd23150f",
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_config_sha256": sha256(args.base_checkpoint / "config.json"),
        "base_weights_sha256": sha256(args.base_checkpoint / "pytorch_model.bin"),
        "vision_checkpoint": str(args.vision_checkpoint.resolve()),
        "vision_config_sha256": sha256(args.vision_checkpoint / "config.json"),
        "language_embeddings": str(embeddings_path),
        "language_embeddings_sha256": sha256(embeddings_path),
        "language_manifest": str(language_manifest),
        "language_manifest_sha256": sha256(language_manifest),
        "language_metadata": language_metadata,
        "lineage": lineage,
        "steps": args.steps,
        "episode_batch_size": args.episode_batch_size,
        "samples_per_episode": args.samples_per_episode,
        "effective_frame_batch_size": args.episode_batch_size * args.samples_per_episode,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "mixed_precision": "bf16",
        "trainable_parameter_dtype": "float32",
        "signal_precision": precision_profile,
        "frozen_vision_parameter_dtype": "bfloat16",
        "vision_encoder_frozen": True,
        "image_augmentation": False,
        "condition_dropout_probability": 0.0,
        "state_noise_snr": None,
        "ema": False,
        "ema_reason": "Official RDT training source marks EMA deprecated and does not sample from it",
        "loss": "official RDTRunner.compute_loss (DDPM prediction_type=sample)",
        "policy_inputs": ["front_rgb", "wrist_rgb", "joint_positions", "gripper_open", "text"],
        "privileged_pose_consumed": False,
        "action_representation": "absolute wrapped UR5e joints + absolute gripper-open in sparse 128D RDT vector",
        "action_horizon": args.action_horizon,
        "horizon_adaptation": horizon_adaptation,
        "camera_slots": NUM_CAMERAS,
        "adapter_execution": "lazy absolute-joint conversion against latest measured state",
        "recommended_replan_steps": 1,
    }
    (args.output / "train_config.json").write_text(json.dumps(configuration, indent=2) + "\n")
    (args.output / "pybullet_rdt_config.json").write_text(
        json.dumps(_target_contract(configuration), indent=2) + "\n"
    )
    log_path = args.output / "train_metrics.jsonl"

    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    epoch = 0
    last_checkpoint: Path | None = None
    started = time.perf_counter()
    while optimizer_step < args.steps:
        dataset.set_epoch(epoch)
        for batch in loader:
            micro_step += 1
            images = batch["images"].to(device=device, dtype=dtype, non_blocking=True)
            states = batch["states"].to(device=device, dtype=input_signal_dtype, non_blocking=True)
            actions = batch["actions"].to(device=device, dtype=input_signal_dtype, non_blocking=True)
            state_mask = batch["state_elem_mask"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            lang_tokens = batch["lang_tokens"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            lang_mask = batch["lang_attn_mask"].to(device=device, non_blocking=True)
            ctrl_freqs = batch["ctrl_freqs"].to(device=device, non_blocking=True)
            batch_size, views, channels, height, width = images.shape
            with torch.no_grad():
                image_tokens = vision(
                    images.reshape(batch_size * views, channels, height, width)
                ).reshape(batch_size, -1, vision.hidden_size)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    lang_tokens=lang_tokens,
                    lang_attn_mask=lang_mask,
                    img_tokens=image_tokens,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_mask.unsqueeze(1),
                    ctrl_freqs=ctrl_freqs,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite RDT loss at microstep {micro_step}")
            (loss / args.gradient_accumulation).backward()
            if micro_step % args.gradient_accumulation != 0:
                continue
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite RDT gradient at microstep {micro_step}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            elapsed = time.perf_counter() - started
            record = {
                "step": optimizer_step,
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(gradient_norm.detach().cpu()),
                "elapsed_seconds": elapsed,
                "updates_per_second": optimizer_step / max(elapsed, 1e-9),
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "mean_valid_action_horizon": float(
                    batch["action_valid_lengths"].float().mean()
                ),
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if optimizer_step == 1 or optimizer_step % 10 == 0:
                print(json.dumps(record), flush=True)
            if optimizer_step % args.save_interval == 0 or optimizer_step == args.steps:
                last_checkpoint = _save_checkpoint(
                    model,
                    optimizer,
                    args.output,
                    optimizer_step,
                    epoch,
                    configuration,
                )
                print(f"Saved {last_checkpoint}", flush=True)
            if optimizer_step >= args.steps:
                break
        epoch += 1

    summary = {
        "status": "complete",
        "steps": optimizer_step,
        "epochs_started": epoch,
        "last_checkpoint": str(last_checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.mode in {"smoke", "train"}:
        from VLA.common.source_training_gate import require_training_source
        require_training_source(args.source, model="rdt", optimizer_updates=args.steps)
        from VLA.rdt.signal_qualification import require_signal_qualification
        require_signal_qualification(args)
    # Thousands of short T5 tensors are checked on CPU at startup.  A large
    # OpenMP pool costs more than these operations and competes with replay
    # workers; GPU compute does not need that host thread count.
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if args.mode in {"smoke", "train"} and args.action_contract_report is None:
        raise ValueError(f"mode={args.mode} requires --action-contract-report")
    lineage = validate_lineage(
        args.source, args.audit_report, args.action_contract_report
    )
    lineage["action_horizon"] = args.action_horizon
    if args.mode == "inspect":
        return _inspect(args, lineage)
    if args.mode == "smoke":
        return _smoke(args, lineage)
    return _train(args, lineage)


if __name__ == "__main__":
    raise SystemExit(main())
