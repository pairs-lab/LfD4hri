#!/usr/bin/env python3
"""Precompute official T5-v1_1-XXL embeddings for an audited prompt set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, T5EncoderModel

from VLA.common.tasks import TASKS
from VLA.rdt.prompt_vocabulary import declared_prompt_vocabulary, vocabulary_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("VLA/rdt/prompt_embeddings.pt"))
    parser.add_argument("--model", default="google/t5-v1_1-xxl")
    parser.add_argument(
        "--dataset-index",
        type=Path,
        default=None,
        help="index.json whose complete prompt vocabulary will be encoded",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-task-vocabulary", action="store_true",
                        help="Encode all declared public task phrases so novel combinations cannot fail at the HTTP lookup")
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument("--load-in-8bit", action="store_true")
    quantization.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Quantize only the T5 encoder weights while precomputing embeddings",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(dataset_index: Path | None) -> tuple[list[str], str | None]:
    if dataset_index is None:
        return sorted({task.prompt for task in TASKS.values()}), None
    index = json.loads(dataset_index.read_text())
    if not isinstance(index, list) or not index:
        raise RuntimeError("Dataset index must be a non-empty JSON list")
    prompts = sorted({str(item["task"]["prompt"]) for item in index})
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise RuntimeError("Dataset index contains an empty prompt")
    return prompts, sha256(dataset_index)


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(args.output)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    prompts, source_index_hash = load_prompts(args.dataset_index)
    training_prompts = list(prompts)
    if args.include_task_vocabulary:
        universe = declared_prompt_vocabulary()
        if set(prompts) - set(universe):
            raise ValueError("Training prompts outside declared task grammar")
        prompts = universe
    tokenizer = AutoTokenizer.from_pretrained(args.model, model_max_length=1024)
    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.bfloat16,
    }
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = T5EncoderModel.from_pretrained(args.model, **model_kwargs).eval()
    # With Accelerate device maps, the first parameter may be a bookkeeping
    # tensor on CPU.  Input IDs must follow the token-embedding layer.
    device = model.get_input_embeddings().weight.device
    embeddings: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[start : start + args.batch_size]
            tokens = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            hidden = model(
                input_ids=tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state
            for row, prompt in enumerate(batch_prompts):
                length = int(tokens.attention_mask[row].sum().item())
                value = hidden[row : row + 1, :length].to(
                    dtype=torch.bfloat16, device="cpu"
                )
                embeddings[prompt] = value
                print(f"{len(embeddings)}/{len(prompts)} {prompt}: {tuple(value.shape)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "model": str(args.model),
        "source_index": None if args.dataset_index is None else str(args.dataset_index.resolve()),
        "source_index_sha256": source_index_hash,
        "prompt_count": len(embeddings),
        "training_prompt_count": len(training_prompts),
        "additional_prompts_scope": "declared_task_vocabulary_no_observations" if args.include_task_vocabulary else None,
        "declared_prompt_vocabulary_sha256": vocabulary_sha256() if args.include_task_vocabulary else None,
        "dtype": "torch.bfloat16",
        "embedding_dim": 4096,
        "encoder_weight_precision": (
            "nf4_4bit" if args.load_in_4bit
            else "int8" if args.load_in_8bit else "bfloat16"
        ),
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "embeddings_sha256": sha256(args.output),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
