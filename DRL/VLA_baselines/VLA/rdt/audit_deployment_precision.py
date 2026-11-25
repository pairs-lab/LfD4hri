"""Verify every loaded target-domain RDT value against its stored FP32 tensor."""
import argparse
import json
from pathlib import Path

import torch

from VLA.rdt import server  # Establish the pinned upstream import path.
from VLA.rdt.checkpoint_loading import load_checkpoint
from VLA.common.provenance import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    model, _ = load_checkpoint(checkpoint, config["pred_horizon"], dtype=torch.float32)
    saved = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    current = model.state_dict()
    if set(current) != set(saved):
        raise RuntimeError("RDT loaded/stored tensor keys differ")
    mismatches, compared, bf16_changed, maximum_bf16_rounding = 0, 0, 0, 0.
    for name, value in saved.items():
        if value.is_floating_point():
            if value.dtype != torch.float32 or current[name].dtype != torch.float32:
                raise RuntimeError("Expected FP32 stored and loaded values: " + name)
            compared += value.numel()
            mismatches += int((value != current[name]).sum())
            rounded = value.to(torch.bfloat16).float()
            bf16_changed += int((value != rounded).sum())
            maximum_bf16_rounding = max(maximum_bf16_rounding, float((value - rounded).abs().max()))
    report = {"passed": mismatches == 0, "checkpoint": str(checkpoint),
        "compared_float_values": compared, "lossless_loader_mismatched_values": mismatches,
        "values_changed_by_legacy_bf16_cast": bf16_changed,
        "legacy_bf16_rounding_max_abs_error": maximum_bf16_rounding,
        "server_sha256": sha256_file(Path(server.__file__)),
        "loader_sha256": sha256_file(Path(__file__).with_name("checkpoint_loading.py")),
        "scope": "CPU checkpoint reload/rounding evidence only; not proof of changed task success"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("RDT reload changed stored fine-tuned parameters")


if __name__ == "__main__":
    main()
