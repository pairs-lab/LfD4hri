"""CPU-only, bounded tensor slices: learned weights versus the cached base.

No model construction or inference. This detects unchanged learned weights;
it does not establish convergence, functional correctness, or success rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file

PROBES = {
    "action_head": "model.action_out_proj.weight",
    "action_encoder": "model.action_in_proj.weight",
    "proprio_encoder": "model.state_proj.weight",
    "visual_connector": "model.vlm_with_expert.vlm.model.connector.modality_projection.proj.weight",
    "language_layer": "model.vlm_with_expert.vlm.model.text_model.layers.0.self_attn.q_proj.weight",
    "action_expert": "model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj.weight",
    "frozen_vision": "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding.weight",
}


def difference_metrics(original, learned):
    original, learned = np.asarray(original), np.asarray(learned)
    if original.shape != learned.shape or not original.size:
        raise ValueError("Weight probe shapes differ or are empty")
    if not np.isfinite(original).all() or not np.isfinite(learned).all():
        raise ValueError("Nonfinite weight probe")
    delta = learned.astype(np.float64) - original.astype(np.float64)
    return {"probed_values": int(delta.size), "changed_values": int(np.count_nonzero(delta)),
            "changed_fraction": float(np.mean(delta != 0)),
            "max_abs_change": float(np.abs(delta).max()),
            "rms_change": float(np.sqrt(np.mean(delta**2)))}


def main():
    from safetensors import safe_open
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    saved = checkpoint / "model.safetensors"
    contract = json.loads((checkpoint / "fp32_master_contract.json").read_text())
    trainable = set(contract["trainable_parameter_names"])
    if not contract.get("passed"):
        raise ValueError("Failed saved precision contract")
    before = {"base": sha256_file(args.base_weights), "learned": sha256_file(saved)}
    rows, failures = {}, []
    # Read no more than 64 first-axis rows per probe, not whole model tensors.
    with safe_open(saved, framework="pt", device="cpu") as new, \
            safe_open(args.base_weights, framework="pt", device="cpu") as base:
        for name, key in PROBES.items():
            old_slice, new_slice = base.get_slice(key), new.get_slice(key)
            shape = old_slice.get_shape()
            if shape != new_slice.get_shape():
                raise ValueError("Base/checkpoint tensor schema differs: " + key)
            old = old_slice[:min(64, shape[0])].float().numpy()
            actual = new_slice[:min(64, shape[0])].float().numpy()
            metrics = difference_metrics(old, actual)
            expected_trainable = name != "frozen_vision"
            if (key in trainable) != expected_trainable:
                failures.append(name + ": saved trainability differs from this recipe")
            if expected_trainable and not metrics["changed_values"]:
                failures.append(name + ": learned probe did not change")
            if not expected_trainable and metrics["changed_values"]:
                failures.append(name + ": frozen vision probe changed; base identity or freeze must be investigated")
            rows[name] = {"tensor": key, "full_shape": shape, "saved_dtype": new_slice.get_dtype(),
                          "expected_trainable": expected_trainable, **metrics}
    if before != {"base": sha256_file(args.base_weights), "learned": sha256_file(saved)}:
        raise RuntimeError("Weights changed during audit")
    report = {"passed": not failures, "failures": failures,
        "scope": "Bounded weight-slice update check, NOT convergence or learned success evidence",
        "checkpoint": str(checkpoint), "base_weights": str(args.base_weights.resolve()),
        "weights_sha256": before, "max_first_axis_rows_per_probe": 64,
        "model_loaded_or_inferred": False, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
