"""Actual official CogACT constructor, tokenizer and image transform on CPU.

Read-only mmap/empty-init loading preserves every published tensor's storage;
NO GPU, neural prediction, synthetic cognition features or robot actions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from VLA.audit_privileged_native_inputs import selected_indices
from VLA.common.provenance import sha256_file
from VLA.cross_vla120_server import load_cogact
from VLA.privileged_native_dataset import PrivilegedNativeDataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("VLA/checkpoints/cogact-small/checkpoints/CogACT-Small.pt"))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    dataset = PrivilegedNativeDataset(args.projection)
    model, load = load_cogact(args.checkpoint, args.output / "official_constructor", meta_only=True)
    processor = model.vlm.vision_backbone.image_transform
    tokenizer = model.vlm.llm_backbone.tokenizer
    samples = []; cells = []; seen = {}
    for index in selected_indices(dataset):
        raw = dataset[index]; task = dataset.rows[raw["episode_index"]]["task"]["name"]
        image = Image.fromarray(raw["image"]).resize((224, 224), Image.Resampling.LANCZOS)
        pixels = processor(image)
        if not isinstance(pixels, dict) or not pixels:
            raise ValueError("Expected actual native fused-vision transform")
        for name, tensor in pixels.items():
            if tuple(tensor.shape) != (3, 224, 224) or not bool(tensor.isfinite().all()):
                raise ValueError("CogACT actual vision tensor shape/finite failure")
        builder = model.vlm.get_prompt_builder()
        builder.add_turn(role="human", message=f"What action should the robot take to {raw['prompt'].lower()}?")
        text = builder.get_prompt()
        untruncated = tokenizer(text, truncation=False).input_ids + [29871, 2]
        truncated = tokenizer(text, truncation=True).input_ids + [29871, 2]
        if untruncated != truncated or len(untruncated) + 256 > min(2048, tokenizer.model_max_length):
            raise ValueError("Actual CogACT instruction would truncate")
        samples.append({"index": index, "task": task, "actual_image_shapes": {k: list(v.shape) for k, v in pixels.items()},
            "actual_native_instruction_tokens": len(untruncated), "native_truncation_did_not_change_tokens": True,
            "actual_prompt": text, "pre_action_state18": raw["pre_action_state18"].tolist()})
        count = seen.get(task, 0); seen[task] = count + 1
        if count < 3:
            # Native transform here uses each published backbone's own mean/
            # std. Display the original post-server resize alongside the
            # exact normalized tensors' arrays in separate .npy files; don't
            # guess their normalization just to produce a pretty image.
            cell = Image.new("RGB", (224, 248), "black"); cell.paste(image, (0, 24))
            ImageDraw.Draw(cell).text((3, 3), f"{task} idx{index}", fill="white"); cells.append(cell)
            for name, tensor in pixels.items():
                np.save(args.output / f"native_{index}_{name}.npy", tensor.detach().cpu().numpy())
    sheet = Image.new("RGB", (224 * 3, 248 * 4), "black")
    for i, cell in enumerate(cells): sheet.paste(cell, ((i % 3) * 224, (i // 3) * 248))
    sheet.save(args.output / "actual_server_front_resize_contact_sheet.png")
    report = {"passed": True, "model": "cogact", "native_samples": len(samples), "samples": samples,
        "source_index_sha256": dataset.audit["source_index_sha256"], "source_manifest_sha256": dataset.audit["source_manifest_sha256"],
        "actual_official_tensor_storage_assignment_verified": load["all_checkpoint_tensor_shapes_and_storage_assignment_exact"],
        "official_weights_not_replaced": True, "neural_prediction_or_success_rate_established": False,
        "native_cognition_feature_or_action_prediction_parity_still_pending": True,
        "GPU_used": False, "oracle_or_IK_used": False,
        "constructor_source_sha256": sha256_file(Path(__file__).parents[1] / "cross_vla120_server.py")}
    (args.output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)


if __name__ == "__main__": main()
