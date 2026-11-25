"""CPU/meta-model audit that language/connector actually received Adam updates.

Unfreezing a config is not evidence that loss gradients reached that branch.
Maps stored optimizer IDs using the exact official parameter order, asserts
that the constructed trainable name list equals the saved master contract,
and checks actual nonzero first/second moments per parameter family.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from VLA.common.provenance import sha256_file


def family(name):
    if ".vlm.model.text_model.embed_tokens." in name:
        return "language_token_embeddings"
    if ".vlm.model.text_model." in name:
        return "vlm_language_attention_mlp"
    if ".vlm.model.connector." in name:
        return "visual_connector"
    if ".lm_expert." in name:
        return "action_expert"
    if "state_proj" in name:
        return "proprioception_projection"
    return "action_time_projections"


def main():
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from VLA.smolvla.precision import fp32_policy_class

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    contract = json.loads((checkpoint/"fp32_master_contract.json").read_text())
    cfg = SmolVLAConfig.from_pretrained(str(checkpoint))
    cfg.load_vlm_weights = False
    cfg.device = "cpu"
    with init_empty_weights():
        policy = fp32_policy_class(contract.get("train_visual_connector", False))(cfg)
    names = list(policy.named_parameters())
    if [name for name, value in names if value.requires_grad] != contract["trainable_parameter_names"]:
        raise ValueError("Architecture/parameter order differs from actual training contract")
    # Stock SmolVLA returns self.parameters(), including frozen parameters.
    if [id(p) for p in policy.get_optim_params()] != [id(p) for _, p in names]:
        raise ValueError("Unsupported nontrivial optimizer parameter-group ordering")
    state = checkpoint.parent/"training_state"
    groups = json.loads((state/"optimizer_param_groups.json").read_text())
    if len(groups) != 1 or groups[0]["params"] != list(range(len(names))):
        raise ValueError("Stored optimizer IDs differ from exact model ordering")
    totals = defaultdict(lambda: {"parameter_tensors": 0, "nonzero_first_moment_tensors": 0,
        "nonzero_second_moment_tensors": 0, "parameter_values": 0, "first_moment_l2_squared": 0.,
        "second_moment_sum": 0.})
    failures, steps = [], set()
    with safe_open(state/"optimizer_state.safetensors", framework="np") as saved:
        keys = set(saved.keys())
        for number, (name, value) in enumerate(names):
            if not value.requires_grad:
                continue
            prefix = f"state/{number}/"
            expected = {prefix+k for k in ("exp_avg", "exp_avg_sq", "step")}
            if not expected <= keys:
                failures.append("Missing Adam state for "+name); continue
            first, second = saved.get_tensor(prefix+"exp_avg"), saved.get_tensor(prefix+"exp_avg_sq")
            if tuple(first.shape) != tuple(value.shape) or tuple(second.shape) != tuple(value.shape):
                raise ValueError("Optimizer tensor shape/order mismatch for "+name)
            if not np.isfinite(first).all() or not np.isfinite(second).all() or (second < 0).any():
                raise ValueError("Invalid Adam moments for "+name)
            steps.add(float(saved.get_tensor(prefix+"step")))
            summary = totals[family(name)]
            summary["parameter_tensors"] += 1
            summary["parameter_values"] += first.size
            summary["nonzero_first_moment_tensors"] += int(np.any(first != 0))
            summary["nonzero_second_moment_tensors"] += int(np.any(second != 0))
            summary["first_moment_l2_squared"] += float(np.square(first.astype(np.float64)).sum())
            summary["second_moment_sum"] += float(second.astype(np.float64).sum())
    required = {"language_token_embeddings", "vlm_language_attention_mlp", "visual_connector", "action_expert"}
    if not required <= set(totals):
        failures.append("Missing a required language/visual/action family")
    for key in required & set(totals):
        if totals[key]["nonzero_second_moment_tensors"] != totals[key]["parameter_tensors"]:
            failures.append("Zero second-moment gradient path in "+key)
    report = {"passed": not failures, "checkpoint": str(checkpoint), "failures": failures,
        "optimizer_steps": sorted(steps), "exact_parameter_order_verified": True,
        "optimizer_state_sha256": sha256_file(state/"optimizer_state.safetensors"), "families": dict(totals),
        "scope": "actual_stored_Adam_gradient_path_diagnostic_NOT_correct_grounding_or_success",
        "policy_weights_loaded": False, "gpu_used": False, "training_performed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise ValueError("A required conditioning branch did not receive optimizer updates")


if __name__ == "__main__":
    main()
