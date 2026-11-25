"""Audit actual OpenVLA labels, prompt prefix, quantization and image tensors on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from VLA.openvla.finetune import IGNORE_INDEX, PyBulletOpenVLADataset, sha256


def main() -> int:
    from transformers import AutoProcessor
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="openvla/openvla-7b")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    conversion = json.loads((args.dataset / "conversion_audit.json").read_text())
    dataset = PyBulletOpenVLADataset(args.dataset, processor, tokenizer,
                                    PurePromptBuilder, conversion["dataset_key"])
    failures = []
    token_failures = []
    prefix_failures = []
    gripper_failures = []
    squared_error = np.zeros(7, dtype=np.float64)
    clipped = np.zeros(7, dtype=np.int64)
    image_samples = []
    selected = set(np.linspace(0, len(dataset) - 1, 96, dtype=int).tolist())
    for index, sample in enumerate(dataset.samples):
        native = np.asarray(sample["action"], dtype=np.float32)
        normalized = dataset.normalize_action(native)
        builder = PurePromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {sample['prompt'].lower()}?")
        builder.add_turn("gpt", tokenizer(normalized))
        ids = np.asarray(processor.tokenizer(builder.get_prompt(), add_special_tokens=True).input_ids)
        expected_tokens = processor.tokenizer.vocab_size - np.digitize(np.clip(normalized, -1, 1), tokenizer.bins)
        if ids[-1] != processor.tokenizer.eos_token_id or not np.array_equal(ids[-8:-1], expected_tokens):
            token_failures.append(index)
        inference_prompt = f"In: What action should the robot take to {sample['prompt'].lower()}?\nOut: "
        inference_ids = np.asarray(processor.tokenizer(inference_prompt, add_special_tokens=True).input_ids)
        if not np.array_equal(ids[:-8], inference_ids):
            prefix_failures.append(index)
        decoded = tokenizer.decode_token_ids_to_actions(expected_tokens)
        physical = np.where(dataset.mask, .5 * (decoded + 1) * (dataset.high - dataset.low) + dataset.low, decoded)
        squared_error += (physical - native) ** 2
        clipped += (dataset.mask & ((native < dataset.low) | (native > dataset.high)))
        if (float(native[6]) > .5) != (float(physical[6]) > .5):
            gripper_failures.append(index)
        if index in selected:
            actual = dataset[index]
            labels = actual["labels"].numpy()
            if not np.all(labels[:-8] == IGNORE_INDEX) or not np.array_equal(labels[-8:], ids[-8:]):
                failures.append(f"wrong actual loss mask at {index}")
            # Use the inference processor, not a second hand-written transform.
            from PIL import Image
            image = Image.open(args.dataset / sample["image"]).convert("RGB")
            inference_pixels = processor(inference_prompt, image, return_tensors="pt")["pixel_values"][0]
            pixels = actual["pixel_values"]
            if not torch.equal(pixels, inference_pixels) or not bool(torch.isfinite(pixels).all()):
                failures.append(f"train/infer image mismatch at {index}")
            image_samples.append({"index": index, "shape": list(pixels.shape),
                                  "pixel_max_error": float((pixels - inference_pixels).abs().max())})
    if token_failures:
        failures.append(f"action-token/EOS reconstruction failed on {len(token_failures)} frames")
    if prefix_failures:
        failures.append(f"training/inference prompt prefixes differ on {len(prefix_failures)} frames")
    if gripper_failures:
        failures.append(f"gripper quantization changes open/closed on {len(gripper_failures)} frames")
    report = {
        "passed": not failures, "failures": failures,
        "model": args.model, "dataset": str(args.dataset.resolve()),
        "samples_sha256": sha256(args.dataset / "samples.json"),
        "frames": len(dataset), "supervised_tokens_per_frame": 8,
        "token_failures": token_failures[:20], "prefix_failures": prefix_failures[:20],
        "gripper_failures": gripper_failures[:20],
        "decoded_physical_rmse_per_dim": np.sqrt(squared_error / len(dataset)).tolist(),
        "q01_q99_clipped_fraction_per_dim": (clipped / len(dataset)).tolist(),
        "actual_dataset_label_and_train_infer_image_samples": image_samples,
        "scope": "CPU data contract only; not model predictions or task success rate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "actual_dataset_label_and_train_infer_image_samples"}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
