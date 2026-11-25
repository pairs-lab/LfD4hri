"""Verify ACTUAL saved privileged QLoRA masters, Adam counts and moments."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from VLA.common.provenance import sha256_file
from VLA.openvla.finetune import validate_checkpoint_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--updates", type=int, required=True); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); manifest = validate_checkpoint_manifest(args.checkpoint)
    if args.output.exists(): raise FileExistsError(args.output)
    adapter = load_file(str(args.checkpoint / "adapter_model.safetensors"), device="cpu")
    if not adapter or any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in adapter.values()):
        raise ValueError("Saved native LoRA masters are not all finite FP32")
    if not any("lora_B" in k and bool(torch.any(t != 0)) for k, t in adapter.items()):
        raise ValueError("Zero-initialized learned B matrices did not change")
    optimizer = torch.load(args.checkpoint / "optimizer.pt", map_location="cpu", weights_only=False)
    counts = []; moment_values = 0
    for state in optimizer["state"].values():
        counts.append(float(state["step"]))
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = state[key]
            if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise ValueError("Saved actual Adam moments are not finite FP32")
            moment_values += tensor.numel()
    if not counts or set(counts) != {float(args.updates)} or manifest["optimizer_step"] != args.updates:
        raise ValueError("Saved ACTUAL optimizer updates differ")
    report = {"passed": True, "checkpoint": str(args.checkpoint.resolve()), "optimizer_updates": args.updates,
        "optimizer_step_counts": sorted(set(counts)), "master_dtype": "float32", "moment_dtype": "float32",
        "adapter_values": sum(t.numel() for t in adapter.values()), "adam_moment_values": moment_values,
        "zero_initialized_lora_B_changed": True, "checkpoint_manifest_sha256": sha256_file(args.checkpoint / "checkpoint_manifest.json"),
        "neural_task_success_certified": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
