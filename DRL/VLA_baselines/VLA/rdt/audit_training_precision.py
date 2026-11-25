"""Audit locally generated RDT optimizer updates and FP32 Adam/master storage."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    weights = torch.load(args.checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
    floating = [value for value in weights.values() if value.is_floating_point()]
    if not floating or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in floating):
        raise RuntimeError("RDT saved master weights are nonfinite or not FP32")
    # This is a trusted checkpoint just produced by our local trainer; its RNG
    # state includes NumPy objects. Do NOT use this loader on third-party files.
    state = torch.load(args.checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state["step"] != args.updates:
        raise RuntimeError("Wrong real optimizer update count")
    moments = []
    steps = set()
    for item in state["optimizer"]["state"].values():
        steps.add(float(item["step"]))
        moments.extend([item["exp_avg"], item["exp_avg_sq"]])
    if (not moments or steps != {float(args.updates)}
            or any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in moments)):
        raise RuntimeError("RDT Adam moments/steps failed FP32 update audit")
    report = {"passed": True, "checkpoint": str(args.checkpoint.resolve()), "real_optimizer_updates": args.updates,
              "master_weight_values": sum(value.numel() for value in floating),
              "adam_moment_values": sum(value.numel() for value in moments), "optimizer_steps": sorted(steps),
              "master_dtype": "float32", "adam_dtype": "float32", "neural_success_claimed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
