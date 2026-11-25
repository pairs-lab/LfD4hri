"""Bounded OpenVLA QLoRA using full-source causal feedback LANGUAGE.

Uses the existing official action-token objective, native image transform,
FP32 LoRA/Adam and exact native restart format. Only the dataset/lineage hooks
are replaced, explicitly, by this separate source-view contract. Never run a
long stage on SmolVLA/RDT-only qualifications or quietly change their files.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from VLA.common.provenance import sha256_file
from VLA.privileged_native_dataset import PrivilegedNativeDataset
from VLA.privileged_prompt import PROMPT_SCHEMA
from VLA.openvla import finetune as native


class NativePromptDataset(native.PyBulletOpenVLADataset):
    def __init__(self, root, processor, action_tokenizer, prompt_builder_cls, dataset_key):
        self.root = Path(root)
        self.contract = json.loads((self.root / "privileged_dataset.json").read_text())
        self.raw = PrivilegedNativeDataset(self.contract["projection"], 1)
        self.processor, self.action_tokenizer = processor, action_tokenizer
        self.prompt_builder_cls, self.dataset_key = prompt_builder_cls, dataset_key
        stats = json.loads((self.root / "dataset_statistics.json").read_text())[dataset_key]["action"]
        self.low, self.high, self.mask = (np.asarray(stats[k], dtype=bool if k == "mask" else np.float32)
                                          for k in ("q01", "q99", "mask"))

    def __len__(self): return len(self.raw)

    def __getitem__(self, index):
        raw = self.raw[index]
        action = self.normalize_action(raw["language_actions"][0])
        builder = self.prompt_builder_cls("openvla")
        builder.add_turn("human", f"What action should the robot take to {raw['prompt'].lower()}?")
        builder.add_turn("gpt", self.action_tokenizer(action))
        ids = self.processor.tokenizer(builder.get_prompt(), add_special_tokens=True, truncation=False).input_ids
        if len(ids) + 256 > min(2048, self.processor.tokenizer.model_max_length):
            raise ValueError("Native feedback/action token truncation forbidden")
        inputs = torch.tensor(ids, dtype=torch.long); labels = inputs.clone()
        labels[:-8] = native.IGNORE_INDEX
        return {"pixel_values": self.processor.image_processor.apply_transform(Image.fromarray(raw["image"])),
                "input_ids": inputs, "labels": labels, "dataset_name": self.dataset_key}

    def __getitems__(self, indices): return [self[int(i)] for i in indices]


def validate_lineage(args):
    root = args.dataset.resolve(); contract_path = root / "privileged_dataset.json"
    contract = json.loads(contract_path.read_text())
    if (contract.get("schema") != PROMPT_SCHEMA or contract.get("model") != "openvla"
            or contract.get("candidate_only") is not False or contract.get("max_optimizer_updates") != 1000
            or Path(contract["qualified_source"]).resolve() != args.source.resolve()):
        raise ValueError("OpenVLA needs its own explicit bounded privileged source")
    for role, item in contract["dependencies"].items():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Privileged dataset dependency changed: " + role)
    from VLA.common.source_training_gate import require_training_source
    require_training_source(args.source, model="openvla", optimizer_updates=args.steps)
    # The newly hard-linked view MUST have the exact parent index and NPZ
    # bytes; its changed qualification metadata is not falsely reused as an
    # old camera replay report. Full parent replay remains a separate proof.
    original = Path(contract["original_source"])
    rows = json.loads((args.source / "index.json").read_text())
    if sha256_file(args.source / "index.json") != contract["original_index_sha256"]:
        raise ValueError("Pilot index differs from qualified parent")
    for row in rows:
        child_stat, parent_stat = (args.source / row["file"]).stat(), (original / row["file"]).stat()
        if (child_stat.st_dev, child_stat.st_ino) != (parent_stat.st_dev, parent_stat.st_ino):
            if sha256_file(args.source / row["file"]) != sha256_file(original / row["file"]):
                raise ValueError("Native labels/source payload changed")
    return {"source": str(args.source.resolve()), "dataset": str(root), "episodes": len(rows),
        "frames": contract["frames"], "source_index_sha256": sha256_file(args.source / "index.json"),
        "source_manifest_sha256": sha256_file(args.source / "dataset_manifest.json"),
        "conversion_audit_sha256": sha256_file(contract_path), "samples_sha256": sha256_file(root / "samples.json"),
        "dataset_statistics_sha256": sha256_file(root / "dataset_statistics.json"),
        "preprocess_audit_sha256": contract["dependencies"]["native_preflight"]["sha256"],
        "roundtrip_audit_sha256": contract["dependencies"]["causal_projection"]["sha256"],
        "privileged_model_inputs": True, "privileged_prompt_schema": PROMPT_SCHEMA,
        "privileged_dataset_contract_sha256": sha256_file(contract_path),
        "privileged_conditioning_is_not_an_oracle_controller": True,
        "fine_tune_recipe_is_bounded_pilot_NOT_data_sufficiency_certification": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True); p.add_argument("--grad-accumulation", type=int, default=4)
    p.add_argument("--resume-from", type=Path); p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=20261014); p.add_argument("--save-freq", type=int, default=250)
    p.add_argument("--base-model", required=True)
    p.add_argument("--smoke-qualification", type=Path); args = p.parse_args()
    contract = json.loads((args.dataset / "privileged_dataset.json").read_text())
    if args.steps > 2:
        if args.smoke_qualification is None: raise ValueError("Actual new-input two-update/reload/robot qualification required")
        proof = json.loads(args.smoke_qualification.read_text())
        if (not proof.get("passed") or proof.get("optimizer_updates") != 2
                or proof.get("dataset_contract_sha256") != sha256_file(args.dataset / "privileged_dataset.json")
                or proof.get("trainer_sha256") != sha256_file(Path(__file__))
                or proof.get("grad_accumulation") != args.grad_accumulation
                or proof.get("learning_rate") != args.learning_rate or proof.get("seed") != args.seed
                or proof.get("actual_neural_robot_controls") != 8):
            raise ValueError("Actual smoke recipe/input proof differs")
        for item in proof["dependencies"]:
            if sha256_file(item["path"]) != item["sha256"]: raise ValueError("Actual smoke proof dependency changed")
    original_args = native.parse_args
    def forwarded():
        return SimpleNamespace(**vars(args), source=Path(contract["qualified_source"]), batch_size=1, lora_rank=16,
                               audit_report=None, preprocess_report=None, roundtrip_report=None)
    native.parse_args = forwarded
    native.validate_lineage = validate_lineage
    native.PyBulletOpenVLADataset = NativePromptDataset
    # A different input contract is part of exact restart identity, not just
    # a human-readable note. Ordinary adapters cannot masquerade as this fit.
    native.RESUME_INVARIANT_KEYS += ("privileged_prompt_schema", "privileged_dataset_contract_sha256")
    return native.main()


if __name__ == "__main__": raise SystemExit(main())
