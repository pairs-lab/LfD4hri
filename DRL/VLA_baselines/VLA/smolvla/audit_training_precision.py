"""Inspect stored master/moment dtypes without loading a model or using GPU."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text())
    contract_path = checkpoint / "fp32_master_contract.json"
    contract = json.loads(contract_path.read_text()) if contract_path.exists() else None
    if not contract and not config["train_expert_only"]:
        raise RuntimeError("VLM-unfrozen checkpoints require a recorded trainable-parameter contract")
    weights, moments, steps = defaultdict(int), defaultdict(int), set()
    with safe_open(checkpoint / "model.safetensors", framework="np") as saved:
        names = contract["trainable_parameter_names"] if contract else [
            name for name in saved.keys() if ".vlm." not in name
        ]
        for name in names:
            tensor = saved.get_slice(name)
            count = 1
            for dimension in tensor.get_shape():
                count *= dimension
            weights[tensor.get_dtype()] += count
    with safe_open(checkpoint.parent / "training_state/optimizer_state.safetensors", framework="np") as saved:
        for name in saved.keys():
            if "exp_avg" in name:
                tensor = saved.get_slice(name)
                count = 1
                for dimension in tensor.get_shape():
                    count *= dimension
                moments[tensor.get_dtype()] += count
            if name.endswith("/step"):
                steps.add(float(saved.get_tensor(name)))
    report = {
        "passed": set(weights) == {"F32"} and set(moments) == {"F32"},
        "checkpoint": str(checkpoint), "trainable_master_value_counts_by_dtype": dict(weights),
        "adam_moment_value_counts_by_dtype": dict(moments), "optimizer_steps": sorted(steps),
        "interpretation": "Optimizer step counters alone do not establish adequate numerical update precision",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
