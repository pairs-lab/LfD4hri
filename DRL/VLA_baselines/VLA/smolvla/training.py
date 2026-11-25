#!/usr/bin/env python3
"""Fail-closed SmolVLA launcher for an audited target-domain training split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "VLA"
    / "datasets"
    / "pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "VLA"
    / "datasets"
    / "lerobot_pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_AUDIT = PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "audit.json"
DEFAULT_ROUNDTRIP = (
    PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "lerobot_roundtrip.json"
)
DEFAULT_STATS_AUDIT = (
    PROJECT_ROOT
    / "VLA"
    / "audits"
    / "camera_v4_vla_oneshot"
    / "smolvla_exact_numeric_stats.json"
)
DEFAULT_PREPROCESS_AUDIT = (
    PROJECT_ROOT
    / "VLA"
    / "audits"
    / "camera_v4_vla_oneshot"
    / "smolvla_exact_worst_preprocess.json"
)
DEFAULT_HF_HOME = PROJECT_ROOT / "VLA" / "checkpoints" / "huggingface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("validate", "smoke", "train"))
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--roundtrip-report", type=Path, default=DEFAULT_ROUNDTRIP)
    parser.add_argument("--stats-report", type=Path, default=DEFAULT_STATS_AUDIT)
    parser.add_argument("--preprocess-report", type=Path, default=DEFAULT_PREPROCESS_AUDIT)
    parser.add_argument("--repo-id")
    parser.add_argument("--base-model", default="lerobot/smolvla_base")
    parser.add_argument(
        "--train-vlm", action="store_true",
        help="Use official train_expert_only=false: adapt language/visual connector while keeping vision encoder frozen",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-updates", type=int, default=None,
                        help="Optimizer updates; auto=min(1000, 10%% of run), not all of a short repair")
    parser.add_argument("--decay-updates", type=int, default=None,
                        help="Cosine schedule length in optimizer updates; default=max(run,30000)")
    parser.add_argument("--train-visual-connector", action="store_true",
                        help="Adapt the existing visual projection as well as the expert; vision/text backbones stay frozen unless --train-vlm is set")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=10_000, help="Optimizer updates")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=1_000)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--initial-frame-sampling-probability", type=float, default=None,
                        help="Optional task/episode-balanced sampling ablation; reserve this mass for initial frames")
    parser.add_argument("--target-balanced-sampling", action="store_true",
                        help="Within each task equalize picked-object names; supervised sampling only, never policy input")
    parser.add_argument("--paired-grounding-probability", type=float, default=0.,
                        help="Optional extra probability mass on audited same-image/different-target pairs at t=0")
    parser.add_argument("--phase-balanced-sampling", action="store_true",
                        help="Explicit target_phase_balanced_v1 training ablation on unchanged qualified labels")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def schedule_contract(optimizer_updates, accumulation, warmup_updates=None, decay_updates=None):
    """Convert update-unit settings to LeRobot's microstep scheduler units."""
    if optimizer_updates < 1 or accumulation < 1:
        raise ValueError("Run and accumulation must be positive")
    warmup = min(1000, optimizer_updates // 10) if warmup_updates is None else warmup_updates
    decay = max(optimizer_updates, 30000) if decay_updates is None else decay_updates
    if warmup < 0 or warmup >= optimizer_updates or decay < optimizer_updates or decay <= warmup:
        raise ValueError("Warmup must end before the run; decay must cover the run")
    return {"warmup_optimizer_updates": warmup, "decay_optimizer_updates": decay,
            "warmup_microsteps": warmup * accumulation, "decay_microsteps": decay * accumulation}


def validate_lineage(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    from VLA.common.source_training_gate import require_training_source
    if getattr(args, "mode", None) in {"train", "smoke"}:
        require_training_source(source, model="smolvla", optimizer_updates=args.steps)
    dataset = args.dataset_root.resolve()
    index_path = source / "index.json"
    source_manifest_path = source / "dataset_manifest.json"
    conversion_path = dataset / "conversion_manifest.json"
    info_path = dataset / "meta" / "info.json"
    stats_path = dataset / "meta" / "stats.json"
    for path in (
        index_path,
        source_manifest_path,
        conversion_path,
        info_path,
        stats_path,
        args.audit_report,
        args.roundtrip_report,
        args.stats_report,
        args.preprocess_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    index_hash = sha256(index_path)
    manifest_hash = sha256(source_manifest_path)
    index = json.loads(index_path.read_text())
    if not index or any(not item.get("success") for item in index):
        raise RuntimeError("Source must contain only successful demonstrations")

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
        raise RuntimeError("Camera/data audit is missing or stale")

    conversion = json.loads(conversion_path.read_text())
    info = json.loads(info_path.read_text())
    if (
        Path(conversion.get("source", "")).resolve() != source
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

    features = info.get("features", {})
    expected_action = {
        "dtype": "float32",
        "shape": [7],
        "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
    }
    expected_state = {
        "dtype": "float32",
        "shape": [7],
        "names": ["q0", "q1", "q2", "q3", "q4", "q5", "gripper_closure"],
    }
    image_keys = sorted(key for key in features if key.startswith("observation.images."))
    if (
        features.get("action") != expected_action
        or features.get("observation.state") != expected_state
        or image_keys != ["observation.images.front", "observation.images.wrist"]
        or any(
            features[key].get("dtype") != "image"
            or len(features[key].get("shape", ())) != 3
            or features[key]["shape"][-1] != 3
            for key in image_keys
        )
    ):
        raise RuntimeError(
            "LeRobot feature schema must be canonical 7D state/action plus exactly front and wrist RGB"
        )

    roundtrip = json.loads(args.roundtrip_report.read_text())
    exact_fields = (
        "state_max_abs_error",
        "action_max_abs_error",
        "front_rgb_max_pixel_error",
        "wrist_rgb_max_pixel_error",
        "prompt_mismatches",
        "index_mismatches",
        "loader_state_max_abs_error",
        "loader_action_max_abs_error",
        "loader_front_rgb_max_pixel_error",
        "loader_wrist_rgb_max_pixel_error",
    )
    if (
        not roundtrip.get("passed")
        or roundtrip.get("failures")
        or roundtrip.get("episodes") != audit.get("episodes")
        or roundtrip.get("frames") != audit.get("frames")
        or any(roundtrip.get(key) != 0 for key in exact_fields)
    ):
        raise RuntimeError("LeRobot source-to-loader round trip is missing or not exact")

    stats_audit = json.loads(args.stats_report.read_text())
    if (
        not stats_audit.get("passed")
        or Path(stats_audit.get("source", "")).resolve() != source
        or Path(stats_audit.get("dataset_root", "")).resolve() != dataset
        or stats_audit.get("source_index_sha256") != index_hash
        or stats_audit.get("source_manifest_sha256") != manifest_hash
        or stats_audit.get("frames") != audit.get("frames")
        or stats_audit.get("statistics_method") != "exact_numpy_float64"
        or stats_audit.get("stats_sha256") != sha256(stats_path)
        or conversion.get("numeric_stats_sha256") != sha256(stats_path)
        or any(
            error != 0.0
            for errors in stats_audit.get("after_max_abs_error", {}).values()
            for error in errors.values()
        )
    ):
        raise RuntimeError("Exact numeric normalization statistics are missing or stale")

    preprocess = json.loads(args.preprocess_report.read_text())
    if (
        not preprocess.get("passed")
        or preprocess.get("model") != "smolvla"
        or Path(preprocess.get("dataset", "")).resolve() != source
        or preprocess.get("transform", {}).get("method") != "lerobot resize_with_pad"
        or preprocess.get("transform", {}).get("size") != [512, 512]
        or any(not sample.get("finite") for sample in preprocess.get("samples", ()))
    ):
        raise RuntimeError("Exact SmolVLA preprocessing audit is missing or stale")

    result = {
        "source": str(source),
        "dataset_root": str(dataset),
        "episodes": audit["episodes"],
        "frames": audit["frames"],
        "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash,
        "stats_sha256": sha256(stats_path),
        "dataset_info_sha256": sha256(info_path),
        "preprocess_audit_sha256": sha256(args.preprocess_report),
        "action_horizon": args.action_horizon,
        "camera_mapping": {
            "observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
    }
    print("Validated SmolVLA lineage:\n" + json.dumps(result, indent=2), flush=True)
    return result


def validate_checkpoint_schema(
    output: Path,
    steps: int,
    info: dict,
    action_horizon: int,
    base_model: str,
    train_vlm: bool = False,
) -> dict:
    """Fail closed if LeRobot saved a base-domain rather than target-domain schema."""

    width = max(6, len(str(steps)))
    checkpoint = output / "checkpoints" / f"{steps:0{width}d}" / "pretrained_model"
    required = (
        checkpoint / "config.json",
        checkpoint / "train_config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
        checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete SmolVLA checkpoint; missing {missing}")

    config = json.loads((checkpoint / "config.json").read_text())
    train_config = json.loads((checkpoint / "train_config.json").read_text())
    features = info["features"]
    expected_inputs = {
        "observation.state": {"type": "STATE", "shape": [7]},
        "observation.images.camera1": {
            "type": "VISUAL",
            "shape": [3, *features["observation.images.front"]["shape"][:2]],
        },
        "observation.images.camera2": {
            "type": "VISUAL",
            "shape": [3, *features["observation.images.wrist"]["shape"][:2]],
        },
    }
    expected_mapping = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }
    if (
        config.get("input_features") != expected_inputs
        or config.get("output_features") != {"action": {"type": "ACTION", "shape": [7]}}
        or config.get("chunk_size") != action_horizon
        or config.get("n_action_steps") != 1
        or str(config.get("pretrained_path")) != base_model
        or not config.get("load_vlm_weights")
        or config.get("train_expert_only") != (not train_vlm)
        or config.get("freeze_vision_encoder") is not True
        or train_config.get("rename_map") != expected_mapping
    ):
        raise RuntimeError("Saved SmolVLA checkpoint does not preserve the audited target schema")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_model_sha256": sha256(checkpoint / "model.safetensors"),
        "checkpoint_input_features": expected_inputs,
        "checkpoint_output_features": {"action": {"type": "ACTION", "shape": [7]}},
    }


def main() -> int:
    args = parse_args()
    if args.action_horizon < 1:
        raise ValueError("--action-horizon must be positive")
    if args.batch_size < 1 or args.steps < 1 or args.gradient_accumulation < 1:
        raise ValueError("Batch size, steps and gradient accumulation must be positive")
    if not 0 < args.learning_rate < 1:
        raise ValueError("--learning-rate must be between zero and one")
    if args.target_balanced_sampling and args.initial_frame_sampling_probability is None:
        raise ValueError("Target-balanced sampling requires an explicit initial-frame probability")
    if not 0 <= args.paired_grounding_probability < 1 or (args.paired_grounding_probability and args.initial_frame_sampling_probability is None):
        raise ValueError("Grounding mass needs a valid probability and the explicit base sampler")
    if args.phase_balanced_sampling and (not args.target_balanced_sampling or args.initial_frame_sampling_probability is None or args.paired_grounding_probability):
        raise ValueError("Phase recipe requires target-balanced initial base and no t=0 pair mixture")
    optimizer_updates = 2 if args.mode == "smoke" else args.steps
    schedule = schedule_contract(optimizer_updates, args.gradient_accumulation,
                                 args.warmup_updates, args.decay_updates)
    lineage = validate_lineage(args)
    lineage["schedule_contract"] = schedule
    base_path = Path(args.base_model)
    if base_path.is_dir():
        from VLA.fingerprint_checkpoint import verify_fingerprint

        args.base_model = str(base_path.resolve())
        base_identity = verify_fingerprint(base_path.resolve())
        lineage["warm_start_content_sha256"] = base_identity["content_tree_sha256"]
        lineage["warm_start_optimizer_restored"] = False
    if args.mode == "validate":
        return 0
    if args.output is None:
        raise ValueError("--output is required for smoke/train")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    steps = optimizer_updates * args.gradient_accumulation
    save_freq = steps if args.mode == "smoke" else args.save_freq * args.gradient_accumulation
    rename_map = json.dumps(
        {
            "observation.images.front": "observation.images.camera1",
            "observation.images.wrist": "observation.images.camera2",
        },
        separators=(",", ":"),
    )
    repo_id = args.repo_id or f"local/{args.source.resolve().name}"
    safe_source_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.source.resolve().name)
    command = [
        sys.executable,
        "-m",
        "VLA.smolvla.train_phase_balanced" if args.phase_balanced_sampling else "VLA.smolvla.train_fp32",
        # Build the policy config from the target dataset schema (7D state and
        # exactly two cameras), then load the official pretrained weights.  The
        # shorter ``--policy.path=...`` form retains the base checkpoint's 6D,
        # three-camera schema in the saved config even though runtime tensors
        # happen to pass through, which is fragile and misstates model inputs.
        "--policy.type=smolvla",
        f"--policy.pretrained_path={args.base_model}",
        "--policy.load_vlm_weights=true",
        f"--policy.chunk_size={args.action_horizon}",
        "--policy.n_action_steps=1",
        "--policy.freeze_vision_encoder=true",
        f"--policy.train_expert_only={str(not args.train_vlm).lower()}",
        "--policy.train_state_proj=true",
        f"--policy.optimizer_lr={args.learning_rate}",
        f"--policy.scheduler_decay_lr={args.learning_rate * .025}",
        # This LeRobot revision advances the scheduler on microbatches.  Scale
        # the published preset with K so accumulation preserves the same
        # warmup/decay in optimizer-update units.
        f"--policy.scheduler_warmup_steps={schedule['warmup_microsteps']}",
        f"--policy.scheduler_decay_steps={schedule['decay_microsteps']}",
        # Newer LeRobot defaults pretrained policies to Hub upload and then
        # requires a remote repo_id even for a fully local run.  This benchmark
        # is intentionally offline/local and fingerprints checkpoints itself.
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={args.dataset_root.resolve()}",
        "--dataset.image_transforms.enable=false",
        f"--rename_map={rename_map}",
        f"--output_dir={output}",
        f"--job_name=smolvla_{safe_source_name}_h{args.action_horizon}",
        f"--steps={steps}",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        # The local constructor promotes trainable masters BEFORE loading.
        # BF16 below is AMP compute only: weights and Adam moments stay FP32.
        "--accelerator.mixed_precision=bf16",
        f"--accelerator.gradient_accumulation.steps={args.gradient_accumulation}",
        "--wandb.enable=false",
        "--env_eval_freq=0",
        "--eval_steps=0",
        f"--save_freq={save_freq}",
        "--save_checkpoint=true",
        "--seed=1000",
    ]
    environment = os.environ.copy()
    environment["HF_HOME"] = str(DEFAULT_HF_HOME)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["VLA_SMOLVLA_MASTER_AUDIT"] = str(output / "master_dtype_audit.json")
    environment["VLA_SMOLVLA_TRAIN_VISUAL_CONNECTOR"] = "1" if args.train_visual_connector else "0"
    if args.initial_frame_sampling_probability is not None:
        from VLA.smolvla.balanced_sampling import sampling_contract

        contract = sampling_contract(args.source, args.initial_frame_sampling_probability, args.target_balanced_sampling, args.paired_grounding_probability)
        if args.phase_balanced_sampling:
            from VLA.smolvla.phase_balanced_sampling import load_phase_recipe
            _, contract = load_phase_recipe(args.source, args.initial_frame_sampling_probability)
        environment["VLA_SMOLVLA_INITIAL_FRAME_PROB"] = str(args.initial_frame_sampling_probability)
        environment["VLA_SMOLVLA_SAMPLING_SOURCE"] = str(args.source.resolve())
        environment["VLA_SMOLVLA_TARGET_BALANCED"] = "1" if args.target_balanced_sampling else "0"
        environment["VLA_SMOLVLA_GROUNDING_PROB"] = str(args.paired_grounding_probability)
        lineage["sampling_contract"] = contract
    else:
        environment.pop("VLA_SMOLVLA_INITIAL_FRAME_PROB", None)
        environment.pop("VLA_SMOLVLA_SAMPLING_SOURCE", None)
        environment.pop("VLA_SMOLVLA_TARGET_BALANCED", None)
        environment.pop("VLA_SMOLVLA_GROUNDING_PROB", None)
    print("Launching:\n" + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
    info = json.loads((args.dataset_root.resolve() / "meta" / "info.json").read_text())
    checkpoint_schema = validate_checkpoint_schema(
        output, steps, info, args.action_horizon, args.base_model, args.train_vlm
    )
    master_audit = json.loads((output / "master_dtype_audit.json").read_text())
    from safetensors import safe_open

    checkpoint = Path(checkpoint_schema["checkpoint"])
    with safe_open(checkpoint / "model.safetensors", framework="np") as saved:
        if any(saved.get_slice(name).get_dtype() != "F32" for name in master_audit["trainable_parameter_names"]):
            raise RuntimeError("Saved trainable master weights are not FP32")
    with safe_open(checkpoint.parent / "training_state/optimizer_state.safetensors", framework="np") as saved:
        if any(saved.get_slice(name).get_dtype() != "F32" for name in saved.keys() if "exp_avg" in name):
            raise RuntimeError("Adam moments are not FP32")
    (checkpoint / "fp32_master_contract.json").write_text(json.dumps({
        **master_audit, "saved_weights_and_adam_moments_fp32": True,
        "constructor_sha256": sha256(Path(__file__).with_name("precision.py")),
        "trainer_sha256": sha256(Path(__file__).with_name("train_fp32.py")),
    }, indent=2) + "\n")
    (output / "training_lineage.json").write_text(
        json.dumps(
            {
                **lineage,
                **checkpoint_schema,
                "base_model": args.base_model,
                "steps": steps,
                "step_unit": "microbatches",
                "optimizer_updates": optimizer_updates,
                "gradient_accumulation": args.gradient_accumulation,
                "effective_batch_size": args.batch_size * args.gradient_accumulation,
                "batch_size": args.batch_size,
                "mode": args.mode,
                "train_vlm": args.train_vlm,
                "train_visual_connector": args.train_visual_connector,
                "vision_encoder_frozen": True,
                "learning_rate": args.learning_rate,
                "master_dtype": "float32",
                "compute_dtype": "bfloat16_autocast",
                "model_inputs": ["front_rgb", "wrist_rgb", "joint_state", "gripper", "text"],
                "native_action": "canonical 7D delta EEF plus Robotiq command",
            },
            indent=2,
        )
        + "\n"
    )
    # Store the final source/camera contract inside the checkpoint before it
    # is fingerprinted. Never mutate already-frozen legacy checkpoints.
    (checkpoint / "vla_checkpoint_manifest.json").write_text(
        (output / "training_lineage.json").read_text()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
