"""Real batched native OpenVLA token/image/label preflight; no neural weights."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer

from VLA.audit_privileged_native_inputs import SNAPSHOT
from VLA.common.provenance import sha256_file
from VLA.openvla.train_privileged_prompt import NativePromptDataset
from VLA.privileged_prompt import augment_prompt, parse_feedback


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    processor = AutoProcessor.from_pretrained(str(SNAPSHOT), trust_remote_code=True)
    tokenizer = ActionTokenizer(processor.tokenizer)
    key = json.loads((args.dataset / "conversion_audit.json").read_text())["dataset_key"]
    dataset = NativePromptDataset(args.dataset, processor, tokenizer, PurePromptBuilder, key)
    ids = np.random.default_rng(20261014).choice(len(dataset), 128, replace=False).tolist()
    collator = PaddedCollatorForActionPrediction(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right")
    loader = DataLoader(dataset, batch_size=2, sampler=ids, collate_fn=collator, num_workers=0)
    errors = {"pixel_values": 0., "input_ids": 0., "labels": 0., "attention_mask": 0.}
    processed = 0; maximum = 0
    for batch in loader:
        expected = collator([dataset[index] for index in ids[processed:processed+2]])
        for field in errors:
            errors[field] = max(errors[field], float(torch.max(torch.abs(batch[field].float() - expected[field].float()))))
        for index in ids[processed:processed+2]:
            raw = dataset.raw[index]; item = dataset[index]
            expected_prompt = augment_prompt(raw["original_prompt"], raw["pre_action_state18"])
            if expected_prompt != raw["prompt"] or np.max(np.abs(parse_feedback(raw["prompt"]) - raw["pre_action_state18"])) > .000051:
                raise ValueError("Actual loader privileged feedback differs")
            ids_infer = processor.tokenizer("In: What action should the robot take to " + raw["prompt"].lower() + "?\nOut: ", truncation=False).input_ids
            train_ids = item["input_ids"].tolist(); common = 0
            for a, b in zip(train_ids, ids_infer):
                if a != b: break
                common += 1
            if common < len(ids_infer) - 2 or int((item["labels"] != -100).sum()) != 8:
                raise ValueError("Training/inference language or supervised action suffix differs")
            pixels_infer = processor.image_processor.apply_transform(Image.fromarray(raw["image"]))
            if not torch.equal(pixels_infer, item["pixel_values"]): raise ValueError("Native image prep differs")
            maximum = max(maximum, len(train_ids))
        processed += len(batch["input_ids"])
    if processed != 128 or any(errors.values()): raise ValueError("Actual native batched loader parity failed")
    proof = {"passed": True, "actual_loader_batches": 64, "samples": processed, "errors": errors,
        "native_train_inference_image_and_instruction_parity": True,
        "supervised_suffix_tokens": 8, "future_feedback_conditioning": False,
        "maximum_actual_training_tokens": maximum, "neural_weights_loaded": False,
        "source_index_sha256": dataset.raw.audit["source_index_sha256"],
        "dataset_contract_sha256": sha256_file(args.dataset / "privileged_dataset.json"),
        "trainer_sha256": sha256_file(Path(__file__).with_name("train_privileged_prompt.py"))}
    args.output.write_text(json.dumps(proof, indent=2) + "\n"); print(json.dumps(proof), flush=True)


if __name__ == "__main__": main()
