#!/usr/bin/env python3
"""QLoRA fine-tuning of official OpenVLA on the audited PyBullet split.

This is the official OpenVLA action-token objective with a small map-style
dataset wrapper, as suggested by upstream ``vla-scripts/finetune.py``.  It
avoids an unnecessary TensorFlow/RLDS conversion while preserving q01/q99
normalization, prompt construction, image transform, collator and action
tokenizer from OpenVLA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterator, Sized

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler


IGNORE_INDEX = -100
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_ROOT = PROJECT_ROOT / "VLA" / "third_party" / "openvla"
if str(OPENVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_ROOT))
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "VLA"
    / "datasets"
    / "pybullet_ur5e_robotiq_train_cam_v4_vla_oneshot"
)
DEFAULT_AUDIT = PROJECT_ROOT / "VLA" / "audits" / "camera_v4_vla_oneshot" / "audit.json"
DEFAULT_PREPROCESS_AUDIT = (
    PROJECT_ROOT
    / "VLA"
    / "audits"
    / "camera_v4_vla_oneshot"
    / "openvla_exact_worst_preprocess.json"
)
DEFAULT_ROUNDTRIP_AUDIT = (
    PROJECT_ROOT
    / "VLA"
    / "audits"
    / "camera_v4_vla_oneshot"
    / "openvla_roundtrip.json"
)
RESTART_FORMAT_VERSION = 1
RESUME_INVARIANT_KEYS = (
    "base_model",
    "dataset",
    "dataset_key",
    "source",
    "source_index_sha256",
    "source_manifest_sha256",
    "conversion_audit_sha256",
    "samples_sha256",
    "dataset_statistics_sha256",
    "preprocess_audit_sha256",
    "roundtrip_audit_sha256",
    "batch_size",
    "grad_accumulation",
    "learning_rate",
    "lora_rank",
    "seed",
    "image_input",
    "task_input",
    "native_action",
    "objective",
    "quantization",
    "optimizer",
    "max_grad_norm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--preprocess-report", type=Path, default=DEFAULT_PREPROCESS_AUDIT)
    parser.add_argument("--roundtrip-report", type=Path, default=DEFAULT_ROUNDTRIP_AUDIT)
    parser.add_argument("--base-model", default="openvla/openvla-7b")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Exact restartable checkpoint directory in OUTPUT/checkpoints/.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resume_invariants(config: dict) -> dict:
    missing = [key for key in RESUME_INVARIANT_KEYS if key not in config]
    if missing:
        raise RuntimeError(f"Training config lacks resume invariants: {missing}")
    return {key: config[key] for key in RESUME_INVARIANT_KEYS}


def resume_invariant_sha256(config: dict) -> str:
    return canonical_json_sha256(resume_invariants(config))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


class EpochRandomSampler(Sampler[int]):
    """Reconstructible per-epoch shuffle with an exact sample cursor."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        epoch: int,
        start_index: int = 0,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not 0 <= self.start_index <= len(self.data_source):
            raise ValueError("start_index must be within the dataset")

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) & ((1 << 63) - 1))
        permutation = torch.randperm(len(self.data_source), generator=generator).tolist()
        yield from permutation[self.start_index :]

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


def canonical_sampler_cursor(epoch: int, sample_offset: int, dataset_size: int) -> tuple[int, int]:
    if dataset_size < 1:
        raise ValueError("Cannot train on an empty dataset")
    if not 0 <= sample_offset <= dataset_size:
        raise ValueError("Sampler offset is outside the dataset")
    if sample_offset == dataset_size:
        return epoch + 1, 0
    return epoch, sample_offset


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError(f"Malformed RNG state: expected {sorted(required)}, got {sorted(state)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the saved restart state: "
                f"saved={len(cuda_states)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def checkpoint_artifacts(checkpoint: Path) -> dict[str, dict[str, int | str]]:
    artifacts: dict[str, dict[str, int | str]] = {}
    for path in sorted(checkpoint.rglob("*")):
        if path.is_file() and path.name != "checkpoint_manifest.json":
            relative = path.relative_to(checkpoint).as_posix()
            artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return artifacts


def write_checkpoint_manifest(checkpoint: Path, *, step: int, invariant_hash: str) -> dict:
    manifest = {
        "format_version": RESTART_FORMAT_VERSION,
        "optimizer_step": int(step),
        "resume_invariant_sha256": invariant_hash,
        "artifacts": checkpoint_artifacts(checkpoint),
    }
    atomic_write_json(checkpoint / "checkpoint_manifest.json", manifest)
    return manifest


def validate_checkpoint_manifest(checkpoint: Path) -> dict:
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Checkpoint {checkpoint} predates restart manifests and cannot be resumed exactly"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format_version") != RESTART_FORMAT_VERSION:
        raise RuntimeError(f"Unsupported restart manifest version in {manifest_path}")
    expected = manifest.get("artifacts")
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"Checkpoint manifest has no artifacts: {manifest_path}")
    actual_names = set(checkpoint_artifacts(checkpoint))
    if actual_names != set(expected):
        raise RuntimeError(
            f"Checkpoint artifact set changed: missing={sorted(set(expected) - actual_names)}, "
            f"extra={sorted(actual_names - set(expected))}"
        )
    for relative, record in expected.items():
        path = checkpoint / relative
        if path.stat().st_size != record.get("bytes") or sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Checkpoint artifact hash mismatch: {path}")
    return manifest


def validate_lineage(args: argparse.Namespace) -> dict:
    """Bind OpenVLA training to the audited camera-v4 source by hash."""

    source = args.source.resolve()
    dataset = args.dataset.resolve()
    index_path = source / "index.json"
    manifest_path = source / "dataset_manifest.json"
    conversion_path = dataset / "conversion_audit.json"
    samples_path = dataset / "samples.json"
    stats_path = dataset / "dataset_statistics.json"
    for path in (
        index_path,
        manifest_path,
        conversion_path,
        samples_path,
        stats_path,
        args.audit_report,
        args.preprocess_report,
        args.roundtrip_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    index_hash = sha256(index_path)
    manifest_hash = sha256(manifest_path)
    index = json.loads(index_path.read_text())
    if not index or any(not item.get("success") for item in index):
        raise RuntimeError("Source must contain only successful demonstrations")
    expected_frames = sum(int(item["steps"]) for item in index)

    audit = json.loads(args.audit_report.read_text())
    from VLA.common.source_payload_provenance import verify_source_payload_for_profile
    verify_source_payload_for_profile(audit, source)
    if (
        not audit.get("passed")
        or audit.get("failures")
        or Path(audit.get("dataset", "")).resolve() != source
        or audit.get("index_sha256") != index_hash
        or audit.get("manifest_sha256") != manifest_hash
        or audit.get("episodes") != len(index)
        or audit.get("frames") != expected_frames
    ):
        raise RuntimeError("Source camera/data audit is missing or stale")

    conversion = json.loads(conversion_path.read_text())
    samples = json.loads(samples_path.read_text())
    statistics = json.loads(stats_path.read_text())
    if (
        not conversion.get("passed")
        or Path(conversion.get("source", "")).resolve() != source
        or conversion.get("source_index_sha256") != index_hash
        or conversion.get("source_manifest_sha256") != manifest_hash
        or conversion.get("episodes") != len(index)
        or conversion.get("frames") != expected_frames
        or conversion.get("front_rgb_max_pixel_error") != 0
        or len(samples) != expected_frames
        or conversion.get("dataset_key") not in statistics
    ):
        raise RuntimeError("OpenVLA conversion is missing, stale, or incomplete")

    preprocess = json.loads(args.preprocess_report.read_text())
    if (
        not preprocess.get("passed")
        or preprocess.get("model") != args.base_model
        or Path(preprocess.get("dataset", "")).resolve() != dataset
        or Path(preprocess.get("source", "")).resolve() != source
        or any(not sample.get("finite") for sample in preprocess.get("samples", ()))
    ):
        raise RuntimeError("Exact OpenVLA image-preprocessing audit is missing or stale")

    roundtrip = json.loads(args.roundtrip_report.read_text())
    if (
        not roundtrip.get("passed")
        or roundtrip.get("failures")
        or Path(roundtrip.get("source", "")).resolve() != source
        or Path(roundtrip.get("dataset_root", "")).resolve() != dataset
        or roundtrip.get("source_index_sha256") != index_hash
        or roundtrip.get("source_manifest_sha256") != manifest_hash
        or roundtrip.get("conversion_audit_sha256") != sha256(conversion_path)
        or roundtrip.get("samples_sha256") != sha256(samples_path)
        or roundtrip.get("dataset_statistics_sha256") != sha256(stats_path)
        or roundtrip.get("episodes") != len(index)
        or roundtrip.get("frames") != expected_frames
        or roundtrip.get("action_max_abs_error") != 0.0
        or roundtrip.get("front_rgb_max_pixel_error") != 0
        or roundtrip.get("prompt_mismatches") != 0
        or roundtrip.get("index_mismatches") != 0
        or roundtrip.get("missing_images") != 0
        or roundtrip.get("statistics_max_abs_error", float("inf")) > 1e-12
    ):
        raise RuntimeError("OpenVLA source-to-training round trip is missing or stale")

    result = {
        "source": str(source),
        "dataset": str(dataset),
        "episodes": len(index),
        "frames": expected_frames,
        "source_index_sha256": index_hash,
        "source_manifest_sha256": manifest_hash,
        "conversion_audit_sha256": sha256(conversion_path),
        "samples_sha256": sha256(samples_path),
        "dataset_statistics_sha256": sha256(stats_path),
        "preprocess_audit_sha256": sha256(args.preprocess_report),
        "roundtrip_audit_sha256": sha256(args.roundtrip_report),
    }
    print("Validated OpenVLA lineage:\n" + json.dumps(result, indent=2), flush=True)
    return result


class PyBulletOpenVLADataset(Dataset):
    def __init__(
        self, root: Path, processor, action_tokenizer, prompt_builder_cls, dataset_key: str
    ) -> None:
        self.root = root
        self.samples = json.loads((root / "samples.json").read_text())
        statistics = json.loads((root / "dataset_statistics.json").read_text())
        self.dataset_key = dataset_key
        action_stats = statistics[dataset_key]["action"]
        self.low = np.asarray(action_stats["q01"], dtype=np.float32)
        self.high = np.asarray(action_stats["q99"], dtype=np.float32)
        self.mask = np.asarray(action_stats["mask"], dtype=bool)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.prompt_builder_cls = prompt_builder_cls

    def __len__(self) -> int:
        return len(self.samples)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        scaled = np.clip(2.0 * (action - self.low) / (self.high - self.low + 1e-8) - 1.0, -1.0, 1.0)
        normalized = np.where(self.mask, scaled, action)
        # A constant *continuous* channel has no scale and should encode
        # zero.  Do not overwrite mask=False channels: the gripper is kept
        # in native 1=open/0=closed units, even in an all-open touch subset.
        normalized = np.where(self.mask & (self.high == self.low), 0.0, normalized)
        return normalized.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(self.root / sample["image"]).convert("RGB")
        action = self.normalize_action(np.asarray(sample["action"], dtype=np.float32))
        prompt_builder = self.prompt_builder_cls("openvla")
        conversation = (
            {"from": "human", "value": f"What action should the robot take to {sample['prompt'].lower()}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        )
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])
        token_ids = self.processor.tokenizer(
            prompt_builder.get_prompt(), add_special_tokens=True
        ).input_ids
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        return {
            "pixel_values": self.processor.image_processor.apply_transform(image),
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": self.dataset_key,
        }


def to_device(value, device: torch.device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(device=device, dtype=torch.bfloat16)
        return value.to(device=device)
    return value


def save_checkpoint(
    model,
    processor,
    optimizer: torch.optim.Optimizer,
    dataset: Path,
    output: Path,
    *,
    step: int,
    total_micro_steps: int,
    epoch: int,
    sample_offset: int,
    dataset_size: int,
    config: dict,
) -> Path:
    """Atomically save everything required to continue at the exact next sample."""

    invariant_hash = resume_invariant_sha256(config)
    cursor_epoch, cursor_offset = canonical_sampler_cursor(epoch, sample_offset, dataset_size)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"{step:06d}"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{step:06d}.tmp-", dir=checkpoints))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        adapter_files = {
            path.name for path in temporary.iterdir() if path.is_file()
        }
        if "adapter_config.json" not in adapter_files or not {
            "adapter_model.safetensors",
            "adapter_model.bin",
        }.intersection(adapter_files):
            raise RuntimeError("OpenVLA checkpoint did not save a complete LoRA adapter")
        shutil.copy2(dataset / "dataset_statistics.json", temporary / "dataset_statistics.json")
        atomic_write_json(temporary / "train_config.json", config)
        training_state = {
            "format_version": RESTART_FORMAT_VERSION,
            "optimizer_step": int(step),
            "total_micro_steps": int(total_micro_steps),
            "epoch": int(cursor_epoch),
            "sample_offset": int(cursor_offset),
            "dataset_size": int(dataset_size),
            "gradient_accumulation_boundary": True,
            "scheduler": None,
            "resume_invariant_sha256": invariant_hash,
        }
        atomic_write_json(temporary / "training_state.json", training_state)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(capture_rng_state(), temporary / "rng_state.pt")
        write_checkpoint_manifest(
            temporary,
            step=step,
            invariant_hash=invariant_hash,
        )
        os.replace(temporary, checkpoint)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint


def validate_resume_checkpoint(
    *,
    resume_from: Path,
    output: Path,
    config: dict,
    dataset_size: int,
    target_steps: int,
) -> dict:
    checkpoint = resume_from.resolve()
    output = output.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.resolve() != output:
        raise RuntimeError("--resume-from must be an exact checkpoint inside OUTPUT/checkpoints/")
    if not output.is_dir():
        raise RuntimeError("Resume requires the existing output directory")

    manifest = validate_checkpoint_manifest(checkpoint)
    required = (
        checkpoint / "train_config.json",
        checkpoint / "training_state.json",
        checkpoint / "optimizer.pt",
        checkpoint / "rng_state.pt",
        checkpoint / "dataset_statistics.json",
        checkpoint / "adapter_config.json",
        output / "train_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Restart checkpoint is incomplete; missing {missing}")

    current_hash = resume_invariant_sha256(config)
    checkpoint_config = json.loads((checkpoint / "train_config.json").read_text())
    root_config = json.loads((output / "train_config.json").read_text())
    hashes = {
        current_hash,
        resume_invariant_sha256(checkpoint_config),
        resume_invariant_sha256(root_config),
        manifest.get("resume_invariant_sha256"),
    }
    if len(hashes) != 1:
        raise RuntimeError("Resume lineage/config differs from the checkpoint's training contract")
    if sha256(checkpoint / "dataset_statistics.json") != config["dataset_statistics_sha256"]:
        raise RuntimeError("Checkpoint action normalization statistics differ from the audited dataset")

    state = json.loads((checkpoint / "training_state.json").read_text())
    step = state.get("optimizer_step")
    if (
        state.get("format_version") != RESTART_FORMAT_VERSION
        or state.get("resume_invariant_sha256") != current_hash
        or state.get("dataset_size") != dataset_size
        or not state.get("gradient_accumulation_boundary")
        or not isinstance(step, int)
        or step < 1
        or state.get("total_micro_steps") != step * config["grad_accumulation"]
        or not isinstance(state.get("epoch"), int)
        or state["epoch"] < 0
        or not isinstance(state.get("sample_offset"), int)
        or not 0 <= state["sample_offset"] < dataset_size
    ):
        raise RuntimeError("Restart cursor is malformed or is not at an optimizer-step boundary")
    try:
        directory_step = int(checkpoint.name)
    except ValueError as error:
        raise RuntimeError(f"Checkpoint directory is not a numeric optimizer step: {checkpoint}") from error
    if directory_step != step or manifest.get("optimizer_step") != step:
        raise RuntimeError("Checkpoint directory, manifest and training cursor disagree on step")
    numeric_checkpoints = [
        int(path.name)
        for path in checkpoint.parent.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not numeric_checkpoints or max(numeric_checkpoints) != step:
        raise RuntimeError("Resume is only allowed from the latest checkpoint in an output lineage")
    if target_steps <= step:
        raise RuntimeError(f"--steps must exceed resumed optimizer step {step}")
    return {"checkpoint": checkpoint, **state}


def load_optimizer_and_rng(
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
) -> None:
    optimizer_state = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(optimizer_state)
    # Older PyTorch versions can leave Adam moments on CPU after loading into
    # CUDA parameters.  Move every tensor to its owning parameter explicitly.
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(parameter.device)
    rng_state = torch.load(checkpoint / "rng_state.pt", map_location="cpu", weights_only=False)
    restore_rng_state(rng_state)


def main() -> int:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
    from transformers import BitsAndBytesConfig

    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer

    args = parse_args()
    from VLA.common.source_training_gate import require_training_source
    # V9 qualification is model- and optimizer-budget-specific. Omitting
    # these fields rejects even a separately qualified OpenVLA pilot. Keep
    # the guard strict: Smol/RDT qualifications do NOT authorize this model.
    require_training_source(args.source, model="openvla", optimizer_updates=args.steps)
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.grad_accumulation < 1
        or args.save_freq < 1
        or args.lora_rank < 1
        or args.learning_rate <= 0
    ):
        raise ValueError("Training steps, batch sizes, save frequency, rank and learning rate must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("OpenVLA fine-tuning requires CUDA")
    lineage = validate_lineage(args)
    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    conversion = json.loads((args.dataset / "conversion_audit.json").read_text())
    dataset_key = conversion["dataset_key"]
    config = {
        "restart_format_version": RESTART_FORMAT_VERSION,
        "base_model": args.base_model,
        "dataset": str(args.dataset),
        "dataset_key": dataset_key,
        **lineage,
        "steps": args.steps,
        "save_freq": args.save_freq,
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "seed": args.seed,
        "image_input": "front_rgb",
        "task_input": "natural-language prompt",
        "native_action": "delta EEF + OpenVLA gripper 1=open/0=closed",
        "objective": "official OpenVLA action-token cross entropy",
        "quantization": "bitsandbytes NF4 4-bit, bf16 compute, double quantization",
        "optimizer": "torch.optim.AdamW",
        "max_grad_norm": 1.0,
    }
    config["resume_invariant_sha256"] = resume_invariant_sha256(config)
    resume_state = None
    if args.resume_from is None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite training output: {args.output}")
        args.output.mkdir(parents=True)
        config["resume_history"] = []
    else:
        resume_state = validate_resume_checkpoint(
            resume_from=args.resume_from,
            output=args.output,
            config=config,
            dataset_size=lineage["frames"],
            target_steps=args.steps,
        )
        previous_root = json.loads((args.output / "train_config.json").read_text())
        history = list(previous_root.get("resume_history", ()))
        history.append(
            {
                "checkpoint": str(resume_state["checkpoint"]),
                "optimizer_step": resume_state["optimizer_step"],
                "previous_target_steps": previous_root.get("steps"),
                "target_steps": args.steps,
            }
        )
        config["resume_history"] = history
        config["resume_from"] = str(resume_state["checkpoint"])
    atomic_write_json(args.output / "train_config.json", config)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    processor_source = resume_state["checkpoint"] if resume_state else args.base_model
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    from VLA.openvla.quantized_loading import quantized_loading_compatibility
    with quantized_loading_compatibility():
        model = AutoModelForVision2Seq.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, quantization_config=quantization,
            device_map={"": 0}, low_cpu_mem_usage=True,
            # The official classes above are registered locally. Passing an
            # explicit config and disabling dynamic model-code lookup keeps
            # complete cached weight snapshots usable offline even when HF
            # did not cache configuration_prismatic.py/modeling_prismatic.py.
            config=OpenVLAConfig.from_pretrained(args.base_model),
            trust_remote_code=False, attn_implementation="eager",
        )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    if resume_state:
        model = PeftModel.from_pretrained(
            model,
            resume_state["checkpoint"],
            is_trainable=True,
        )
    else:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    master_dtypes = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            dtype = str(parameter.dtype)
            master_dtypes[dtype] = master_dtypes.get(dtype, 0) + parameter.numel()
            if parameter.dtype != torch.float32:
                raise RuntimeError(f"OpenVLA LoRA master is not FP32: {name} ({dtype})")
    print(json.dumps({"trainable_master_dtype_counts": master_dtypes}), flush=True)

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    dataset = PyBulletOpenVLADataset(
        args.dataset, processor, action_tokenizer, PurePromptBuilder, dataset_key
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    if len(dataset) != lineage["frames"]:
        raise RuntimeError("Runtime OpenVLA dataset length differs from the audited lineage")
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)
    model.train()
    if resume_state:
        load_optimizer_and_rng(optimizer, resume_state["checkpoint"])
        step = resume_state["optimizer_step"]
        total_micro_steps = resume_state["total_micro_steps"]
        epoch = resume_state["epoch"]
        sample_offset = resume_state["sample_offset"]
        print(
            json.dumps(
                {
                    "resumed_from": str(resume_state["checkpoint"]),
                    "step": step,
                    "epoch": epoch,
                    "sample_offset": sample_offset,
                }
            ),
            flush=True,
        )
    else:
        step = 0
        total_micro_steps = 0
        epoch = 0
        sample_offset = 0
    optimizer.zero_grad(set_to_none=True)
    recent_losses: list[float] = []
    while step < args.steps:
        sampler = EpochRandomSampler(
            dataset,
            seed=args.seed,
            epoch=epoch,
            start_index=sample_offset,
        )
        # DataLoader creates an iterator base seed even with num_workers=0.
        # Keep that bookkeeping off the model's global RNG so rebuilding a
        # loader for a mid-epoch resume cannot change dropout/sampling state.
        loader_generator = torch.Generator()
        loader_generator.manual_seed((args.seed + epoch) & ((1 << 63) - 1))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            generator=loader_generator,
        )
        for batch in loader:
            batch_samples = int(batch["input_ids"].shape[0])
            sample_offset += batch_samples
            batch = to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch["pixel_values"],
                    labels=batch["labels"],
                )
                loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            (loss / args.grad_accumulation).backward()
            total_micro_steps += 1
            recent_losses.append(float(loss.detach().cpu()))
            if total_micro_steps % args.grad_accumulation:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}: {grad_norm.item()}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 10 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(np.mean(recent_losses[-10 * args.grad_accumulation :])),
                            "grad_norm": float(grad_norm.detach().cpu()),
                            "gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                        }
                    ),
                    flush=True,
                )
            if step % args.save_freq == 0 or step == args.steps:
                checkpoint = save_checkpoint(
                    model,
                    processor,
                    optimizer,
                    args.dataset,
                    args.output,
                    step=step,
                    total_micro_steps=total_micro_steps,
                    epoch=epoch,
                    sample_offset=sample_offset,
                    dataset_size=len(dataset),
                    config=config,
                )
                print(f"Saved {checkpoint}", flush=True)
            if step >= args.steps:
                break
        epoch += 1
        sample_offset = 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
